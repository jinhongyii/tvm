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

"""Supporting methods for megakernel testing."""

import threading
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tvm_ffi

import tvm
from tvm.megakernel.kernels import (
    AllreduceTile,
    GateUpSiluTile,
    GemmTile,
    GroupGEMMTileSM100,
    SiluMultiplyTile,
    SplitKReduceTile,
)
from tvm.megakernel.utils.config import JobType, KernelConfig, event_type_names
from tvm.megakernel.utils.dynamic_scheduler import DynamicTileScheduler, MPMCQueueHost
from tvm.megakernel.utils.static_scheduler import StaticTileScheduler
from tvm.megakernel.utils.utils import ceildiv, pack_into_32bit
from tvm.script import tirx as Tx
from tvm.tirx.bench import export_to_perfetto_trace


def get_inverse_plan_info(batch_size, kv_head_num, q_indptr, kv_head_idx, attn_task_num):
    """For layer testing use."""
    MAX_TOTAL_NUM_WORKERS = 1025
    inverse_info = [[] for _ in range(batch_size * kv_head_num)]
    for m in range(attn_task_num):
        bs_idx = q_indptr[m]
        kv_idx = kv_head_idx[m]
        inverse_info[kv_idx * batch_size + bs_idx].append(
            (m // KernelConfig.WG_NUMBER) % KernelConfig.SM_NUMBER
        )
    for m in range(
        attn_task_num, ceildiv(attn_task_num, KernelConfig.WG_NUMBER) * KernelConfig.WG_NUMBER
    ):
        inverse_info[(m - attn_task_num) % (batch_size * kv_head_num)].append(
            (m // KernelConfig.WG_NUMBER) % KernelConfig.SM_NUMBER
        )  # align attn_task_num
    inverse_indptr = [0 for _ in range(MAX_TOTAL_NUM_WORKERS)]
    inverse_indices = [0 for _ in range(MAX_TOTAL_NUM_WORKERS)]
    for i in range(batch_size * kv_head_num):
        inverse_indptr[i + 1] = inverse_indptr[i] + len(inverse_info[i])
        for j in range(len(inverse_info[i])):
            inverse_indices[inverse_indptr[i] + j] = inverse_info[i][j]
    assert (
        inverse_indptr[batch_size * kv_head_num]
        == ceildiv(attn_task_num, KernelConfig.WG_NUMBER) * KernelConfig.WG_NUMBER
    )

    DEV = tvm.cuda(0)
    return (
        tvm.runtime.tensor(np.array(inverse_indptr, dtype=np.int32), DEV),
        tvm.runtime.tensor(np.array(inverse_indices, dtype=np.int32), DEV),
    )


def push_moe_tasks(
    central_queue, batch_size, config, insert_wait_etensor_init=False, use_cp_async_input=False
):
    MOE_BLK_M = 128
    gating_blk_m = 128
    for m_idx in range(ceildiv(batch_size, gating_blk_m)):
        for k_idx in range(config["GATING_SPLIT_K_FACTOR"]):
            central_queue.append((m_idx, 0, k_idx, JobType.MOE_GATING.value))
    if insert_wait_etensor_init:
        for i in range(KernelConfig.SM_NUMBER):
            central_queue.append((i, 0, 0, JobType.WAIT_ETENSOR_INIT.value))
    for m_idx in range(KernelConfig.SM_NUMBER):
        central_queue.append((m_idx, 0, 0, JobType.MOE_TOPK_SOFTMAX.value))
    if use_cp_async_input:
        central_queue.append((0, 0, 0, JobType.MOE_TOKEN_DISPATCH.value))
    else:
        central_queue.append((0, 0, 0, JobType.MOE_ALIGN.value))
        for m_idx in range(KernelConfig.SM_NUMBER):
            central_queue.append((m_idx, 0, 0, JobType.MOE_COUNT_AND_SORT.value))
    max_num_tokens_padded = get_max_num_tokens_padded(
        batch_size, config["NUM_EXPERTS_PER_TOK"], config["NUM_EXPERTS"], MOE_BLK_M
    )
    for m_idx in range(max_num_tokens_padded // MOE_BLK_M):
        for n_idx in range(config["INTERMEDIATE_SIZE"] * 2 // GroupGEMMTileSM100.BLK_N):
            central_queue.append((m_idx, n_idx, 0, JobType.MOE_GROUP_GEMM_GATE_UP_SILU.value))
    for m_idx in range(max_num_tokens_padded // MOE_BLK_M):
        for n_idx in range(config["HIDDEN_SIZE"] // GroupGEMMTileSM100.BLK_N):
            central_queue.append((m_idx, n_idx, 0, JobType.MOE_GROUP_GEMM_DOWN.value))


def generate_exec_queue(
    batch_size,
    attn_task_num,
    config,
    WORLD_SIZE,
    etensor_num,
    scheduler: Literal["static", "dynamic"],
):
    """The execution queue generation function for layer testing use."""
    INTERMEDIATE_SIZE = config["INTERMEDIATE_SIZE"] // WORLD_SIZE
    NUM_ATTENTION_HEADS = config["NUM_ATTENTION_HEADS"] // WORLD_SIZE
    NUM_KEY_VALUE_HEADS = config["NUM_KEY_VALUE_HEADS"] // WORLD_SIZE
    is_moe = "NUM_EXPERTS" in config
    torch.cuda.nvtx.range_push("generate_exec_queue")
    if scheduler == "static":
        exec_queue = np.zeros(
            (KernelConfig.SM_NUMBER, StaticTileScheduler.MAX_TASKS), dtype=np.int32
        )
        central_queue = []

        # init etensor
        for i in range(etensor_num):
            central_queue.append((i, 0, 0, JobType.INIT_ETENSOR.value))

        # qkv projection
        for n_idx in range(
            ceildiv(
                (NUM_ATTENTION_HEADS + 2 * NUM_KEY_VALUE_HEADS) * config["HEAD_DIM"], GemmTile.BLK_N
            )
        ):
            for k_idx in range(config["SPLIT_QKV_PROJECT_DICT"][WORLD_SIZE]):
                central_queue.append((0, n_idx, k_idx, JobType.GEMM_QKV_PROJ.value))

        for i in range(KernelConfig.SM_NUMBER):
            central_queue.append((i, 0, 0, JobType.WAIT_ETENSOR_INIT.value))

        m_split = ceildiv(KernelConfig.SM_NUMBER, NUM_ATTENTION_HEADS + 2 * NUM_KEY_VALUE_HEADS)
        m_tile = ceildiv(batch_size, m_split)
        m_split = ceildiv(batch_size, m_tile)

        # q reduce + rmsnorm + rope
        for m_idx in range(m_split):
            for n_idx in range(NUM_ATTENTION_HEADS):
                central_queue.append((m_idx, n_idx, -1, JobType.Q_REDUCE_RMS_ROPE.value))

        # k reduce + rmsnorm + rope + append
        for m_idx in range(m_split):
            for n_idx in range(NUM_KEY_VALUE_HEADS):
                central_queue.append((m_idx, n_idx, -1, JobType.K_REDUCE_RMS_ROPE_APPEND.value))

        # v reduce + append
        for m_idx in range(m_split):
            for n_idx in range(NUM_KEY_VALUE_HEADS):
                central_queue.append((m_idx, n_idx, -1, JobType.V_REDUCE_APPEND.value))

        # attention
        attn_tile_num = ceildiv(attn_task_num, KernelConfig.WG_NUMBER)
        for m_idx in range(min(KernelConfig.SM_NUMBER, attn_tile_num)):
            central_queue.append((m_idx, -1, -1, JobType.BATCH_ATTENTION.value))

        if attn_task_num > NUM_KEY_VALUE_HEADS * batch_size:
            # merge
            for m_idx in range(
                ceildiv(
                    batch_size * NUM_ATTENTION_HEADS,
                    KernelConfig.WG_NUMBER * KernelConfig.WARP_NUMBER,
                )
            ):
                central_queue.append((m_idx, -1, -1, JobType.BATCH_ATTENTION_MERGE.value))

        # o projection
        for n_idx in range(ceildiv(config["HIDDEN_SIZE"], GemmTile.BLK_N)):
            for k_idx in range(config["SPLIT_O_PROJECT_DICT"][WORLD_SIZE]):
                central_queue.append((0, n_idx, k_idx, JobType.GEMM_O_PROJ.value))

        if WORLD_SIZE > 1:
            # o reduction
            m_split_o_proj_reduce = min(
                batch_size,
                KernelConfig.SM_NUMBER // (config["HIDDEN_SIZE"] // SplitKReduceTile.N_UNIT),
            )
            n_tile_o_proj_reduce = (
                ceildiv(SplitKReduceTile.N_REPEAT, ceildiv(batch_size, m_split_o_proj_reduce))
                * SplitKReduceTile.N_UNIT
            )
            m_tile_o_proj_reduce = ceildiv(batch_size, m_split_o_proj_reduce)
            m_split_o_proj_reduce = ceildiv(batch_size, m_tile_o_proj_reduce)
            for n_idx in range(ceildiv(config["HIDDEN_SIZE"], n_tile_o_proj_reduce)):
                for m_idx in range(m_split_o_proj_reduce):
                    central_queue.append((m_idx, n_idx, 0, JobType.GEMM_O_REDUCE.value))

            # o allreduce
            for m_idx in range(ceildiv(batch_size, AllreduceTile.M_TILE)):
                for n_idx in range(
                    ceildiv(config["HIDDEN_SIZE"] // WORLD_SIZE, AllreduceTile.N_TILE)
                ):
                    central_queue.append((m_idx, n_idx, 0, JobType.O_ALLREDUCE.value))

        # add rmsnorm
        for m_idx in range(batch_size):
            central_queue.append((m_idx, -1, -1, JobType.ATTN_ADD_RMS_NORM.value))

        if is_moe:
            push_moe_tasks(central_queue, batch_size, config)
        else:
            if config["GATE_UP_PROJ_SPLIT_K_FACTOR_DICT"][WORLD_SIZE] == 1:
                # gate_up_silu
                assert INTERMEDIATE_SIZE % GateUpSiluTile.BLK_N == 0
                for n_idx in range(INTERMEDIATE_SIZE * 2 // GateUpSiluTile.BLK_N):
                    central_queue.append((0, n_idx, 0, JobType.GATE_UP_SILU.value))
            else:
                # gate_up_proj
                assert INTERMEDIATE_SIZE % GemmTile.BLK_N == 0
                for n_idx in range(INTERMEDIATE_SIZE // GemmTile.BLK_N):
                    for k_idx in range(config["GATE_UP_PROJ_SPLIT_K_FACTOR_DICT"][WORLD_SIZE]):
                        central_queue.append((0, n_idx, k_idx, JobType.GEMM_GATE_UP_PROJ.value))
                        central_queue.append(
                            (
                                0,
                                n_idx + INTERMEDIATE_SIZE // GemmTile.BLK_N,
                                k_idx,
                                JobType.GEMM_GATE_UP_PROJ.value,
                            )
                        )

                # gate_up reduce
                m_split_gate_up_proj_reduce = min(
                    batch_size,
                    KernelConfig.SM_NUMBER // (INTERMEDIATE_SIZE * 2 // SplitKReduceTile.N_UNIT),
                )
                m_tile_gate_up_proj_reduce = ceildiv(batch_size, m_split_gate_up_proj_reduce)
                m_split_gate_up_proj_reduce = ceildiv(batch_size, m_tile_gate_up_proj_reduce)
                for m_idx in range(m_split_gate_up_proj_reduce):
                    for n_idx in range(ceildiv(INTERMEDIATE_SIZE * 2, SplitKReduceTile.N_UNIT)):
                        central_queue.append((m_idx, n_idx, 0, JobType.GATE_UP_PROJ_REDUCE.value))

                # split_silu_multiply
                for m_idx in range(ceildiv(INTERMEDIATE_SIZE, SiluMultiplyTile.TILE_SIZE)):
                    central_queue.append((m_idx, 0, 0, JobType.SPLIT_SILU_MULTIPLY.value))

            # gemm_down_proj
            for n_idx in range(ceildiv(config["HIDDEN_SIZE"], GemmTile.BLK_N)):
                for k_idx in range(config["DOWN_PROJ_SPLIT_K_FACTOR_DICT"][WORLD_SIZE]):
                    central_queue.append((0, n_idx, k_idx, JobType.GEMM_DOWN_PROJ.value))

            if WORLD_SIZE > 1:
                # down_proj_reduce
                m_split_down_proj_reduce = min(
                    KernelConfig.SM_NUMBER // (config["HIDDEN_SIZE"] // SplitKReduceTile.N_UNIT),
                    batch_size,
                )
                n_tile = (
                    ceildiv(
                        SplitKReduceTile.N_REPEAT, ceildiv(batch_size, m_split_down_proj_reduce)
                    )
                    * SplitKReduceTile.N_UNIT
                )
                m_tile_down_proj_reduce = ceildiv(batch_size, m_split_down_proj_reduce)
                m_split_down_proj_reduce = ceildiv(batch_size, m_tile_down_proj_reduce)
                for m_idx in range(m_split_down_proj_reduce):
                    for n_idx in range(ceildiv(config["HIDDEN_SIZE"], n_tile)):
                        central_queue.append((m_idx, n_idx, 0, JobType.DOWN_PROJ_REDUCE.value))

                # down_proj_allreduce
                for m_idx in range(ceildiv(batch_size, AllreduceTile.M_TILE)):
                    for n_idx in range(
                        ceildiv(config["HIDDEN_SIZE"] // WORLD_SIZE, AllreduceTile.N_TILE)
                    ):
                        central_queue.append((m_idx, n_idx, 0, JobType.DOWN_PROJ_ALLREDUCE.value))

        # add_rms_norm
        for m_idx in range(batch_size):
            central_queue.append((m_idx, 0, 0, JobType.MLP_ADD_RMS_NORM.value))

        tile_idx = 0
        while len(central_queue) > 0:
            for bx in range(KernelConfig.SM_NUMBER):
                if len(central_queue) > 0:
                    exec_queue[bx, tile_idx] = pack_into_32bit(*central_queue.pop(0))
                else:
                    exec_queue[bx, tile_idx] = pack_into_32bit(-1, -1, -1, JobType.END.value)
            tile_idx += 1
        for bx in range(KernelConfig.SM_NUMBER):
            exec_queue[bx, tile_idx] = pack_into_32bit(-1, -1, -1, JobType.END.value)
        DEV = tvm.cuda(0)
        ret = tvm.runtime.tensor(exec_queue, device=DEV)
        torch.cuda.nvtx.range_pop()
        return ret
    elif scheduler == "dynamic":
        exec_queue = MPMCQueueHost(DynamicTileScheduler.MAX_TASKS)
        # init etensor
        for i in range(etensor_num):
            exec_queue.enqueue(JobType.INIT_ETENSOR.value, i, 0, 0)
        for n in reversed(
            range(
                ceildiv(
                    (NUM_ATTENTION_HEADS + 2 * NUM_KEY_VALUE_HEADS) * config["HEAD_DIM"],
                    GemmTile.BLK_N,
                )
            )
        ):
            for k in range(config["SPLIT_QKV_PROJECT_DICT"][WORLD_SIZE]):
                exec_queue.enqueue(JobType.GEMM_QKV_PROJ.value, 0, n, k)
        torch.cuda.nvtx.range_pop()
        return exec_queue
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler}")


@tvm_ffi.register_global_func("tirx.megakernel.get_max_num_tokens_padded")
def get_max_num_tokens_padded(batch_size, topk, num_experts, moe_blk_m):
    if isinstance(batch_size, int):
        if batch_size * topk < num_experts:
            return batch_size * topk * moe_blk_m
        else:
            return (num_experts + ceildiv(batch_size * topk - num_experts, moe_blk_m)) * moe_blk_m
    else:
        return Tx.if_then_else(
            batch_size * topk < num_experts,
            batch_size * topk * moe_blk_m,
            (num_experts + ceildiv(batch_size * topk - num_experts, moe_blk_m)) * moe_blk_m,
        )


def get_max_blocks_padded_relaxed(batch_size, topk, num_experts, moe_blk_m):
    return batch_size * topk // moe_blk_m + (num_experts + 1)


def generate_etensor_unmatched_dim(i, dim_len, in_par_size, out_par_size):
    start_out_par = i * out_par_size
    end_out_par = min(dim_len, (i + 1) * out_par_size)
    return (end_out_par - 1) // in_par_size - start_out_par // in_par_size + 1


def generate_exec_queue_moe(
    batch_size,
    config,
    etensor_num,
    scheduler: Literal["static", "dynamic"],
    use_cp_async_input=False,
):
    torch.cuda.nvtx.range_push("generate_exec_queue")
    if scheduler == "static":
        exec_queue = np.zeros(
            (KernelConfig.SM_NUMBER, StaticTileScheduler.MAX_TASKS), dtype=np.int32
        )
        central_queue = []
        # init etensor
        for i in range(etensor_num):
            central_queue.append((i, 0, 0, JobType.INIT_ETENSOR.value))
        push_moe_tasks(
            central_queue,
            batch_size,
            config,
            insert_wait_etensor_init=True,
            use_cp_async_input=use_cp_async_input,
        )
        tile_idx = 0

        while len(central_queue) > 0:
            for bx in range(KernelConfig.SM_NUMBER):
                if len(central_queue) > 0:
                    exec_queue[bx, tile_idx] = pack_into_32bit(*central_queue.pop(0))
                else:
                    exec_queue[bx, tile_idx] = pack_into_32bit(-1, -1, -1, JobType.END.value)
            tile_idx += 1
        for bx in range(KernelConfig.SM_NUMBER):
            exec_queue[bx, tile_idx] = pack_into_32bit(-1, -1, -1, JobType.END.value)
        DEV = tvm.cuda(0)
        ret = tvm.runtime.tensor(exec_queue, device=DEV)
        torch.cuda.nvtx.range_pop()
        return ret
    elif scheduler == "dynamic":
        exec_queue = MPMCQueueHost(DynamicTileScheduler.MAX_TASKS)
        gating_blk_m = 128
        # init etensor
        for i in range(etensor_num):
            exec_queue.enqueue(JobType.INIT_ETENSOR.value, i, 0, 0)
        for m in range(ceildiv(batch_size, gating_blk_m)):
            for k in range(config["GATING_SPLIT_K_FACTOR"]):
                exec_queue.enqueue(JobType.MOE_GATING.value, m, 0, k)
        torch.cuda.nvtx.range_pop()
        return exec_queue


class ProfilerHandler:
    def __init__(self):
        self.counter = 0
        self.lock = threading.Lock()
        # TODO: make this configurable if needed
        self.trigger_count = 167
        self.profiler_layer_id = [3, 47]
        self.dir_path = Path("~/qwen3-mg-debug").expanduser()
        self.file_name = "qwen3-model-mega"

    def export_trace(self, profiler_buffer, rank):
        with self.lock:
            self.counter += 1
            current_run = self.counter

        if current_run == self.trigger_count:
            for layer_id in self.profiler_layer_id:
                if rank == -1:
                    file_name = f"{self.dir_path}/{self.file_name}-layer{layer_id}.perfetto-trace"
                else:
                    file_name = f"{self.dir_path}/{self.file_name}-layer{layer_id}-rank{rank}.perfetto-trace"  # noqa: E501
                export_to_perfetto_trace(
                    profiler_buffer[layer_id].numpy(), file_name, event_type_names
                )
                print(f"Exported layer {layer_id} to {file_name}")


profiler_handler = ProfilerHandler()


@tvm_ffi.register_global_func("megakernel.export_trace")
def export_trace(profiler_buffer, rank):
    profiler_handler.export_trace(profiler_buffer, rank)
