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

"""Configuration for megakernel."""

from enum import Enum
from typing import Any

import tvm

F8_BYTES = 1
F16_BYTES = 2
F32_BYTES = 4
F128_BYTES = 16

# Paged kv-cache config
KV_LAYOUT = "HND"


class KernelConfig:
    # global constant
    M_CLUSTER = 1
    N_CLUSTER = 1
    WG_NUMBER = 2
    WARP_NUMBER = 4
    NUM_THREADS = (32 * WARP_NUMBER) * WG_NUMBER
    SM_NUMBER = 148
    CTA_GROUP = M_CLUSTER
    MAX_SMEM_SIZE = 232448


class JobType(Enum):
    V_REDUCE_APPEND = 0
    K_REDUCE_RMS_ROPE_APPEND = 1
    Q_REDUCE_RMS_ROPE = 2
    BATCH_ATTENTION = 3
    BATCH_ATTENTION_MERGE = 4
    GATE_UP_PROJ_REDUCE = 5
    DOWN_PROJ_ALLREDUCE = 6
    O_ALLREDUCE = 7
    ATTN_ADD_RMS_NORM = 8
    GEMM_O_REDUCE = 9
    GEMM_O_PROJ = 10
    GEMM_QKV_PROJ = 11
    MLP_ADD_RMS_NORM = 12
    DOWN_PROJ_REDUCE = 13
    GEMM_DOWN_PROJ = 14
    SPLIT_SILU_MULTIPLY = 15
    GEMM_GATE_UP_PROJ = 16
    GATE_UP_SILU = 17
    MOE_GATING = 18
    MOE_TOPK_SOFTMAX = 19
    MOE_ALIGN = 20
    MOE_TOKEN_DISPATCH = 21
    MOE_COUNT_AND_SORT = 22
    MOE_GROUP_GEMM_GATE_UP = 23
    MOE_SILU_MULTIPLY = 24
    MOE_GROUP_GEMM_DOWN = 25
    MOE_TOPK_REDUCE = 26
    MOE_GROUP_GEMM_GATE_UP_SILU = 27
    INIT_ETENSOR = 28
    WAIT_ETENSOR_INIT = 29

    # end
    END = 31


class ProfileEventType(Enum):
    GEMM_GATE_UP_PROJ = 0
    SPLIT_SILU_MULTIPLY = 1
    GEMM_DOWN_PROJ = 2
    DOWN_PROJ_REDUCE = 3
    MLP_ADD_RMS_NORM = 4
    FETCH = 5
    GEMM_QKV_PROJ = 6
    GEMM_QKV_REDUCE = 7
    RMSNORM = 8
    ROPE = 9
    APPEND_KV = 10
    BATCH_DECODE_NO_SPLIT = 11
    BATCH_DECODE_SPLIT = 12
    DECODE_MERGE = 13
    GEMM_O_PROJ = 14
    GEMM_O_REDUCE = 15
    ATTN_ADD_RMS_NORM = 16
    Q_RMSNORM_ROPE = 17
    K_RMSNORM_ROPE_APPEND_KV = 18
    V_APPEND_KV = 19
    PUSH = 20
    O_ALLREDUCE = 21
    DOWN_PROJ_ALLREDUCE = 22
    GATE_UP_PROJ_REDUCE = 23
    BATCH_ATTENTION = 24
    BATCH_ATTENTION_MERGE = 25
    PREFETCH = 26
    TMA = 27
    MMA = 28
    ATTN_INIT = 29
    ATTN_LOAD_Q = 30
    ATTN_LOOP_BODY = 31
    ATTN_COMPUTE_QKV = 32
    ATTN_WRITE_BACK = 33
    Q_REDUCE_RMSNORM_ROPE = 34
    K_REDUCE_RMSNORM_ROPE_APPEND = 35
    V_REDUCE_APPEND = 36
    GATE_UP_SILU = 37
    MOE_GATING = 38
    TOPK_SOFTMAX = 39
    MOE_ALIGN = 40
    COUNT_AND_SORT = 41
    GROUP_GEMM_GATE_UP = 42
    SILU_MUL = 43
    GROUP_GEMM_DOWN = 44
    TOPK_REDUCE = 45
    EP_DISPATCH_PRECOMPUTE = 46
    EP_DISPATCH_SEND = 47
    EP_DISPATCH_RECV = 48
    EP_COMBINE_SEND = 49
    EP_COMBINE_RECV = 50
    GROUP_GEMM_GATE_UP_SILU = 51
    INIT_ETENSOR = 52
    WAIT_ETENSOR_INIT = 53
    END = 54


map_job_type_to_profile_event_type = {
    JobType.GEMM_GATE_UP_PROJ.value: ProfileEventType.GEMM_GATE_UP_PROJ,
    JobType.SPLIT_SILU_MULTIPLY.value: ProfileEventType.SPLIT_SILU_MULTIPLY,
    JobType.GEMM_DOWN_PROJ.value: ProfileEventType.GEMM_DOWN_PROJ,
    JobType.DOWN_PROJ_REDUCE.value: ProfileEventType.DOWN_PROJ_REDUCE,
    JobType.MLP_ADD_RMS_NORM.value: ProfileEventType.MLP_ADD_RMS_NORM,
    JobType.GEMM_QKV_PROJ.value: ProfileEventType.GEMM_QKV_PROJ,
    JobType.GEMM_O_PROJ.value: ProfileEventType.GEMM_O_PROJ,
    JobType.GEMM_O_REDUCE.value: ProfileEventType.GEMM_O_REDUCE,
    JobType.ATTN_ADD_RMS_NORM.value: ProfileEventType.ATTN_ADD_RMS_NORM,
    JobType.O_ALLREDUCE.value: ProfileEventType.O_ALLREDUCE,
    JobType.DOWN_PROJ_ALLREDUCE.value: ProfileEventType.DOWN_PROJ_ALLREDUCE,
    JobType.GATE_UP_PROJ_REDUCE.value: ProfileEventType.GATE_UP_PROJ_REDUCE,
    JobType.BATCH_ATTENTION.value: ProfileEventType.BATCH_ATTENTION,
    JobType.BATCH_ATTENTION_MERGE.value: ProfileEventType.BATCH_ATTENTION_MERGE,
    JobType.Q_REDUCE_RMS_ROPE.value: ProfileEventType.Q_REDUCE_RMSNORM_ROPE,
    JobType.K_REDUCE_RMS_ROPE_APPEND.value: ProfileEventType.K_REDUCE_RMSNORM_ROPE_APPEND,
    JobType.V_REDUCE_APPEND.value: ProfileEventType.V_REDUCE_APPEND,
    JobType.GATE_UP_SILU.value: ProfileEventType.GATE_UP_SILU,
    JobType.END.value: ProfileEventType.END,
}

event_type_names = [
    "GEMM_GATE_UP_PROJ",
    "SPLIT_SILU_MULTIPLY",
    "GEMM_DOWN_PROJ",
    "DOWN_PROJ_REDUCE",
    "MLP_ADD_RMS_NORM",
    "FETCH",
    "GEMM_QKV_PROJ",
    "GEMM_QKV_REDUCE",
    "RMSNORM",
    "ROPE",
    "APPEND_KV",
    "BATCH_DECODE_NO_SPLIT",
    "BATCH_DECODE_SPLIT",
    "DECODE_MERGE",
    "GEMM_O_PROJ",
    "GEMM_O_REDUCE",
    "ATTN_ADD_RMS_NORM",
    "Q_RMSNORM_ROPE",
    "K_RMSNORM_ROPE_APPEND_KV",
    "V_APPEND_KV",
    "PUSH",
    "O_ALLREDUCE",
    "DOWN_PROJ_ALLREDUCE",
    "GATE_UP_PROJ_REDUCE",
    "BATCH_ATTENTION",
    "BATCH_ATTENTION_MERGE",
    "PREFETCH",
    "TMA",
    "MMA",
    "ATTN_INIT",
    "ATTN_LOAD_Q",
    "ATTN_LOOP_BODY",
    "ATTN_COMPUTE_QKV",
    "ATTN_WRITE_BACK",
    "Q_REDUCE_RMSNORM_ROPE",
    "K_REDUCE_RMSNORM_ROPE_APPEND",
    "V_REDUCE_APPEND",
    "GATE_UP_SILU",
    "MOE_GATING",
    "TOPK_SOFTMAX",
    "MOE_ALIGN",
    "COUNT_AND_SORT",
    "GROUP_GEMM_GATE_UP",
    "SILU_MUL",
    "GROUP_GEMM_DOWN",
    "TOPK_REDUCE",
    "EP_DISPATCH_PRECOMPUTE",
    "EP_DISPATCH_SEND",
    "EP_DISPATCH_RECV",
    "EP_COMBINE_SEND",
    "EP_COMBINE_RECV",
    "GROUP_GEMM_GATE_UP_SILU",
    "INIT_ETENSOR",
    "WAIT_ETENSOR_INIT",
]

qwen3_30b_a3b_config = {
    "MODEL_NAME": "qwen3_30b_a3b",
    "TIE_WORD_EMBEDDINGS": False,
    "VOCAB_SIZE": 151936,
    "MAX_POSITION_EMBEDDINGS": 40960,
    "HIDDEN_SIZE": 2048,
    "INTERMEDIATE_SIZE": 768,
    "NUM_HIDDEN_LAYERS": 48,
    "NUM_ATTENTION_HEADS": 32,
    "NUM_KEY_VALUE_HEADS": 4,
    "HEAD_DIM": 128,
    "RMS_NORM_EPS": 1e-6,
    "ROPE_THETA": 1000000,
    "NUM_EXPERTS": 128,
    "NUM_EXPERTS_PER_TOK": 8,
    "GATING_SPLIT_K_FACTOR": 4,
    "SPLIT_QKV_PROJECT_DICT": {1: 3, 4: 4, 8: 4},
    "SPLIT_O_PROJECT_DICT": {1: 5, 4: 3, 8: 2},
}

qwen3_32b_config = {
    "MODEL_NAME": "qwen3_32b",
    "TIE_WORD_EMBEDDINGS": False,
    "VOCAB_SIZE": 151936,
    "MAX_POSITION_EMBEDDINGS": 40960,
    "HIDDEN_SIZE": 5120,
    "INTERMEDIATE_SIZE": 25600,
    "NUM_HIDDEN_LAYERS": 64,
    "NUM_ATTENTION_HEADS": 64,
    "NUM_KEY_VALUE_HEADS": 8,
    "HEAD_DIM": 128,
    "RMS_NORM_EPS": 1e-6,
    "ROPE_THETA": 1000000,
    "SPLIT_QKV_PROJECT_DICT": {1: 3, 4: 4, 8: 4},
    "SPLIT_O_PROJECT_DICT": {1: 3, 4: 3, 8: 2},
    "GATE_UP_PROJ_SPLIT_K_FACTOR_DICT": {1: 1, 4: 1, 8: 2},
    "DOWN_PROJ_SPLIT_K_FACTOR_DICT": {1: 10, 4: 3, 8: 3},
}

llama3_1b_config = {
    "MODEL_NAME": "llama3_1b",
    "TIE_WORD_EMBEDDINGS": True,
    "VOCAB_SIZE": 128256,
    "MAX_POSITION_EMBEDDINGS": 131072,
    "HIDDEN_SIZE": 2048,
    "INTERMEDIATE_SIZE": 8192,
    "NUM_HIDDEN_LAYERS": 16,
    "NUM_ATTENTION_HEADS": 32,
    "NUM_KEY_VALUE_HEADS": 8,
    "HEAD_DIM": 64,
    "RMS_NORM_EPS": 1e-5,
    "ROPE_THETA": 500000.0,
    "ROPE_SCALING": {
        "FACTOR": 32.0,
        "HIGH_FREQ_FACTOR": 4.0,
        "LOW_FREQ_FACTOR": 1.0,
        "ORIGINAL_MAX_POSITION_EMBEDDINGS": 8192,
        "ROPE_TYPE": "llama3",
    },
    "SPLIT_QKV_PROJECT_DICT": {1: 8, 4: -1, 8: -1},
    "SPLIT_O_PROJECT_DICT": {1: 8, 4: -1, 8: -1},
    "GATE_UP_PROJ_SPLIT_K_FACTOR_DICT": {1: 1, 4: -1, 8: -1},
    "DOWN_PROJ_SPLIT_K_FACTOR_DICT": {1: 9, 4: -1, 8: -1},
}


@tvm.register_global_func("tirx.megakernel.get_model_config")
def get_model_config(model_name: str) -> dict[str, Any]:
    """Get model config by model name.

    Parameters
    ----------
    model_name : str
        The model name.

    Returns
    -------
    Dict[str, Any]
        The model config.
    """
    if model_name == "qwen3_30b_a3b" or model_name == "qwen3_30b_a3b_unfused":
        return qwen3_30b_a3b_config
    elif model_name == "qwen3_32b":
        return qwen3_32b_config
    elif model_name == "llama3_1b":
        return llama3_1b_config
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
