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
import pytest

import tvm
import tvm.testing
from tvm.script import tirx as Tx

from .utils import run_on_remote_and_check_correct

target = tvm.target.Target("aws/trn1/trn1.2xlarge")


@pytest.mark.dependency(depends=["ssh_success"])
def test_gemm2(ssh_client):
    K = 4096
    M = 4096
    N = 4096
    BLOCK_M = 1024
    BLOCK_N = 1024
    BLOCK_K = 1024
    NUM_BLOCK_M = M // BLOCK_M
    NUM_BLOCK_N = N // BLOCK_N
    NUM_BLOCK_K = K // BLOCK_K
    TILE_M = 128
    TILE_N = 512
    NUM_TILE_M = BLOCK_M // TILE_M
    NUM_TILE_N = BLOCK_N // TILE_N
    dtype = "float16"

    # fmt: off
    @Tx.inline
    def mm(iter_id, a_sbuf, b_sbuf, c_psum, result_tiles):
        n = Tx.meta_var(iter_id // (NUM_BLOCK_M * NUM_BLOCK_K))
        k = Tx.meta_var(iter_id % NUM_BLOCK_K)
        m = Tx.meta_var(iter_id // NUM_BLOCK_K % NUM_BLOCK_M)
        for mi, ni in Tx.grid(NUM_TILE_M, NUM_TILE_N):
            psum_bank = Tx.meta_var((mi*NUM_TILE_N+ni)%8)
            TILE_N_START = Tx.meta_var(ni * TILE_N)
            TILE_M_START = Tx.meta_var(mi * TILE_M)
            # FIXME: currently nki has a bug for psum initialization
            # fix this when NKI exposes psum initialization API
            Tx.gemm(c_psum[psum_bank], a_sbuf[iter_id%3, :, TILE_M_START:TILE_M_START+TILE_M], b_sbuf[iter_id%3, :, TILE_N_START:TILE_N_START+TILE_N], c_psum[psum_bank], transpose_A=True)  # noqa: E501
            if k == 0:
                Tx.copy(result_tiles[(n*NUM_BLOCK_K+m)%2,
                    TILE_M_START : TILE_M_START+TILE_M,
                    TILE_N_START : TILE_N_START + TILE_N,
                ], c_psum[psum_bank])
            else:
                Tx.add(
                    result_tiles[(n*NUM_BLOCK_K+m)%2,
                        TILE_M_START : TILE_M_START+TILE_M,
                        TILE_N_START : TILE_N_START + TILE_N,
                    ],
                    result_tiles[(n*NUM_BLOCK_K+m)%2,
                        TILE_M_START : TILE_M_START+TILE_M,
                        TILE_N_START : TILE_N_START + TILE_N,
                    ],
                    c_psum[psum_bank],
                )

    @Tx.inline
    def load_A(iter_id, a_sbuf, A):
        m = Tx.meta_var(iter_id // NUM_BLOCK_K % NUM_BLOCK_M)
        k = Tx.meta_var(iter_id % NUM_BLOCK_K)
        Tx.copy(
            a_sbuf[iter_id % 3], A[ k * BLOCK_K : (k + 1) * BLOCK_K, (m) * BLOCK_M : (m + 1) * BLOCK_M]  # noqa: E501
        )

    @Tx.inline
    def load_B(iter_id, b_sbuf, B):
        n = Tx.meta_var(iter_id // (NUM_BLOCK_M * NUM_BLOCK_K))
        k = Tx.meta_var(iter_id % NUM_BLOCK_K)
        Tx.copy(
            b_sbuf[iter_id % 3],
            B[(k) * BLOCK_K : (k + 1) * BLOCK_K, n * BLOCK_N : (n + 1) * BLOCK_N],
        )
    # do not reuse memory load, and keep result_tiles small
    @Tx.prim_func(tirx=True)
    def matmul2(A_ptr: Tx.handle, B_ptr: Tx.handle, C_ptr: Tx.handle):
        Tx.func_attr({"num_inputs": 2})
        A = Tx.match_buffer(A_ptr, (K, M), dtype)
        B = Tx.match_buffer(B_ptr, (K, N), dtype)
        C = Tx.match_buffer(C_ptr, (M, N), dtype)
        with Tx.kernel():
            result_tiles = Tx.alloc_buffer((2, BLOCK_M, BLOCK_N), "float32", scope="trn.sbuf", layout="FPF")  # noqa: E501
            b_sbuf = Tx.alloc_buffer((3, BLOCK_K, BLOCK_N), dtype, scope="trn.sbuf", layout="FPF")
            a_sbuf = Tx.alloc_buffer((3, BLOCK_K, BLOCK_M), dtype, scope="trn.sbuf", layout="FPF")
            c_psum = Tx.alloc_buffer((8, TILE_M, TILE_N), "float32", scope="trn.psum", layout="FPF", allocated_addr=(0, 0))  # noqa: E501
            load_A(0, a_sbuf, A)
            load_B(0, b_sbuf, B)
            for iter_id in Tx.serial(NUM_BLOCK_M * NUM_BLOCK_K * NUM_BLOCK_N-1):
                m = Tx.meta_var(iter_id // NUM_BLOCK_K % NUM_BLOCK_M)
                k = Tx.meta_var(iter_id % NUM_BLOCK_K)
                n = Tx.meta_var(iter_id // (NUM_BLOCK_M * NUM_BLOCK_K))
                load_A(iter_id+1, a_sbuf, A)
                load_B(iter_id+1, b_sbuf, B)
                mm(iter_id, a_sbuf, b_sbuf, c_psum, result_tiles)
                if k == NUM_BLOCK_K-1:
                    Tx.copy(C[m*BLOCK_M:(m+1)*BLOCK_M, n*BLOCK_N:(n+1)*BLOCK_N], result_tiles[(n*NUM_BLOCK_K+m)%2])  # noqa: E501
            mm(NUM_BLOCK_M * NUM_BLOCK_K * NUM_BLOCK_N-1, a_sbuf, b_sbuf, c_psum, result_tiles)
            Tx.copy(C[(NUM_BLOCK_M-1)*BLOCK_M:(NUM_BLOCK_M-1)*BLOCK_M+BLOCK_M, (NUM_BLOCK_N-1)*BLOCK_N:(NUM_BLOCK_N-1)*BLOCK_N+BLOCK_N], result_tiles[(NUM_BLOCK_M * NUM_BLOCK_N-1)%2])  # noqa: E501
    # fmt: on
    with target:
        mod = tvm.IRModule({"main": matmul2})
        func = mod["main"]
        run_on_remote_and_check_correct(func, lambda x, y: (x.reshape(K, M).T @ y,), target)


if __name__ == "__main__":
    test_gemm2()
