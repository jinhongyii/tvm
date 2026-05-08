# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Static tile scheduler for megakernel."""

from typing import Literal

import tvm
from tvm.megakernel.utils.base import SemaphoreBase, TileSchedulerBase
from tvm.megakernel.utils.config import JobType, KernelConfig
from tvm.megakernel.utils.utils import any_sync, atomic_add_int32, gt, unpack_from_32bit
from tvm.script import tirx as Tx


class Semaphore(SemaphoreBase):
    def __init__(self, buffer):
        self.sem = buffer
        self.state = Tx.alloc_buffer([1], "int32", scope="local", align=4)

    @Tx.inline
    def semaphore_wait(self, *coord, level: Literal["cta", "warp"] = "cta", mask=0xFFFFFFFF):
        if level == "cta":
            with Tx.thread():
                while 1:
                    Tx.ptx.ld_global_acquire(
                        self.state[0],
                        self.sem.access_ptr("r", offset=self.sem.elem_offset_of(coord)),
                    )
                    if Tx.cuda.syncthreads_and(self.state[0] == 0):
                        break
                    Tx.cuda.nano_sleep(40)
        elif level == "warp":
            with Tx.thread():
                warp_id = Tx.warp_id([KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER])
                lane_id = Tx.lane_id([32])
                if (mask >> warp_id) & 1 == 1:
                    self.state[0] = -1
                    while 1:
                        if lane_id == 0:
                            Tx.ptx.ld_global_acquire(
                                self.state[0],
                                self.sem.access_ptr("r", offset=self.sem.elem_offset_of(coord)),
                            )
                        if any_sync(0xFFFFFFFF, self.state[0] == 0):
                            break
                        Tx.cuda.nano_sleep(40)
        else:
            assert False

    @Tx.inline
    def semaphore_notify(self, *coord, rank=-1, release=False):
        # wg is synced
        self.state[0] = atomic_add_int32(
            self.sem.ptr_to(coord), -(self.base + 1), rank, release=release
        )
        if self.state[0] <= 0:
            while 1:
                Tx.ptx.ld_global_acquire(self.state[0], self.sem.ptr_to(coord))
                if gt(self.state[0], 0):
                    atomic_add_int32(
                        self.sem.ptr_to(coord), -(self.base + 1), rank, release=release
                    )
                    break
                Tx.cuda.nano_sleep(40)


class StaticTileScheduler(TileSchedulerBase):
    MAX_TASKS = 128

    def __init__(self, prefix: str, exec_queue, smem_manager, debug=False):
        super().__init__()
        self.exec_queue = exec_queue
        self.debug = debug
        self.prefix = prefix
        self.smem_manager = smem_manager

    @Tx.inline
    def _update_current_m_n_idx(self):
        unpack_from_32bit(
            self.queue_smem[self.tile_idx],
            Tx.address_of(self.task_type),
            Tx.address_of(self.m_idx),
            Tx.address_of(self.n_idx),
            Tx.address_of(self.k_idx),
        )

    def _alloc(self):
        self.m_idx = Tx.local_scalar("int32")
        self.n_idx = Tx.local_scalar("int32")
        self.k_idx = Tx.local_scalar("int32")
        self.task_type = Tx.local_scalar("int32")
        self.tile_idx = Tx.local_scalar("int32")
        self.queue_smem = self.smem_manager.alloc(
            (self.MAX_TASKS,), "int32", align=16, method="persistent"
        )

    @Tx.inline
    def init(self):
        self._alloc()
        with Tx.cta():
            bx = Tx.cta_id([KernelConfig.SM_NUMBER])
            tid = Tx.thread_id([KernelConfig.NUM_THREADS])
            self.tile_idx = 0
            for k in Tx.serial(Tx.ceildiv(self.MAX_TASKS, KernelConfig.NUM_THREADS)):
                idx = Tx.meta_var(k * KernelConfig.NUM_THREADS + tid)
                if idx < self.MAX_TASKS:
                    self.queue_smem[idx] = self.exec_queue[bx, idx]
            Tx.tvm_storage_sync("shared")
            self._update_current_m_n_idx()

    def get_idx_and_task_type(self):
        return [self.m_idx, self.n_idx, self.k_idx], self.task_type

    @Tx.inline
    def next_tile(self):
        self.tile_idx += 1
        self._update_current_m_n_idx()

    @Tx.inline
    def wait(
        self, evt: Semaphore, *coord, wait_level: Literal["cta", "warp"] = "cta", mask=0xFFFFFFFF
    ):
        evt.semaphore_wait(*coord, level=wait_level, mask=mask)

    @Tx.inline
    def notify(
        self,
        evt: Semaphore,
        func_notify,
        scope: Literal["thread", "warp", "warpgroup", "cta"] = "thread",
        scope_id=0,
        release=False,
    ):
        # Notes: Here each thread will notify only at most one time,
        #        and the tids of the threads involved among scope in the notification process start from 0 and increment sequentially.  # noqa: E501
        # Notes: (num, rank, coord) = func_notify(notify_idx), rank=-1 for the local rank
        # Notes: scope_id = -1 represents that each scope will separately notify

        max_notify_num_map = Tx.meta_var(
            {
                "thread": 1,
                "warp": 32,
                "warpgroup": KernelConfig.NUM_THREADS // KernelConfig.WG_NUMBER,
                "cta": KernelConfig.NUM_THREADS,
            }
        )
        max_scope_id_map = Tx.meta_var(
            {
                "thread": KernelConfig.NUM_THREADS,
                "warp": KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER,
                "warpgroup": KernelConfig.WG_NUMBER,
                "cta": 1,
            }
        )

        @Tx.inline
        def sync(scope: Literal["thread", "warp", "warpgroup", "cta"], scope_id=0):
            if scope == "thread":
                pass
            elif scope == "warp":
                Tx.cuda.warp_sync()
            elif scope == "warpgroup":
                Tx.ptx.bar.sync(6 + scope_id, 128)
            elif scope == "cta":
                Tx.tvm_storage_sync("shared")

        with Tx.cta():
            wg_id = Tx.warpgroup_id([KernelConfig.WG_NUMBER])
            warp_id = Tx.warp_id([KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER])
            tid = Tx.thread_id([KernelConfig.NUM_THREADS])
            tid_in_wg = Tx.thread_id_in_wg([KernelConfig.NUM_THREADS // KernelConfig.WG_NUMBER])
            lane_id = Tx.lane_id([32])
            idx_map = Tx.meta_var(
                {
                    "thread": (tid, 0),
                    "warp": (warp_id, lane_id),
                    "warpgroup": (wg_id, tid_in_wg),
                    "cta": (0, tid),
                }
            )
            idx = Tx.meta_var(idx_map[scope])
            if self.debug:
                Tx.cuda.trap_when_assert_failed(
                    scope_id == -1 or scope_id < max_scope_id_map[scope]
                )
            if scope_id == -1 or idx[0] == scope_id:
                sync(scope, scope_id)
                # `func_notify` can emit side effects while constructing dependency exprs.
                # Evaluate it once and then unpack to avoid duplicated rewrites.
                notify_info = Tx.meta_var(func_notify(idx[1]))
                notify_num = Tx.meta_var(notify_info[0])
                rank = Tx.meta_var(notify_info[1])
                coord = Tx.meta_var(notify_info[2:])
                if self.debug:
                    Tx.cuda.trap_when_assert_failed(notify_num <= max_notify_num_map[scope])
                if idx[1] < notify_num:
                    evt.semaphore_notify(*coord, rank=rank, release=release)

    def valid(self):
        return tvm.tirx.all(self.tile_idx < self.MAX_TASKS, self.task_type != JobType.END.value)
