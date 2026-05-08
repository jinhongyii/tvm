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

from typing import Literal

from tvm.megakernel.utils.base import MegaKernelWrapper
from tvm.megakernel.utils.dynamic_scheduler import DynamicTileScheduler
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tirx as Tx

NUM_HIDDEN_LAYERS = 64


def get_qwen3_megakernel_relax_mod(
    mk: MegaKernelWrapper, scheduler: Literal["static", "dynamic"], TP_SIZE: int, PROFILER_ON: bool
):
    assert not mk.TIE_WORD_EMBEDDINGS, "Qwen3-32B does not support tie word embeddings"

    def static_mod():
        # fmt: off
        @R.macro(hygienic=False)
        def call_qwen3_layer(input0, input1, layer_id):
            with R.dataflow():
                # 8i+1, 8i+2, 8i+3, 8i+4, 8i+5, 8i+6, 8i+8, 8i+15 if i<num_hidden_layers-1 else 8i+9
                model_layers_0_self_attn_c_attn_weight1: R.Tensor(((mk.NUM_ATTENTION_HEADS + mk.NUM_KEY_VALUE_HEADS * 2) * mk.HEAD_DIM, mk.HIDDEN_SIZE), dtype="float16") = packed_params[8*layer_id+1]  # noqa: E501, F821
                model_layers_0_self_attn_o_proj_weight1: R.Tensor((mk.HIDDEN_SIZE, mk.NUM_ATTENTION_HEADS * mk.HEAD_DIM), dtype="float16") = packed_params[8*layer_id+2]  # noqa: E501, F821
                model_layers_0_self_attn_q_norm_weight1: R.Tensor((mk.HEAD_DIM,), dtype="float16") = packed_params[8*layer_id+3]  # noqa: E501, F821
                model_layers_0_self_attn_k_norm_weight1: R.Tensor((mk.HEAD_DIM,), dtype="float16") = packed_params[8*layer_id+4]  # noqa: E501, F821
                model_layers_0_mlp_gate_up_proj_weight1: R.Tensor((2 * mk.INTERMEDIATE_SIZE, mk.HIDDEN_SIZE), dtype="float16") = packed_params[8*layer_id+5]  # noqa: E501, F821
                model_layers_0_mlp_down_proj_weight1: R.Tensor((mk.HIDDEN_SIZE, mk.INTERMEDIATE_SIZE), dtype="float16") = packed_params[8*layer_id+6]  # noqa: E501, F821
                model_layers_0_post_attention_layernorm_weight1: R.Tensor((mk.HIDDEN_SIZE,), dtype="float16") = packed_params[8*layer_id+8]  # noqa: E501, F821
                model_norm_weight1: R.Tensor((mk.HIDDEN_SIZE,), dtype="float16") = packed_params[8*layer_id+15 if layer_id < NUM_HIDDEN_LAYERS-1 else 8*layer_id+9]  # noqa: E501, F821


                default_device = R.call_pure_packed("runtime.disco.device", sinfo_args=[R.Object])
                partital_qkv = R.builtin.alloc_tensor(R.shape([mk.SPLIT_QKV_PROJECT, batch_size, (mk.NUM_ATTENTION_HEADS + mk.NUM_KEY_VALUE_HEADS * 2) * mk.HEAD_DIM]), dtype="float32", runtime_device_index=0)  # noqa: E501, F821
                qkv = R.builtin.alloc_tensor(R.shape([batch_size, mk.NUM_ATTENTION_HEADS + mk.NUM_KEY_VALUE_HEADS * 2, mk.HEAD_DIM]), dtype="float16", runtime_device_index=0)  # noqa: E501, F821
                o = R.builtin.alloc_tensor(R.shape([batch_size, mk.NUM_ATTENTION_HEADS, mk.HEAD_DIM]), dtype="float16", runtime_device_index=0)  # noqa: E501, F821
                o_partial_attn = R.builtin.alloc_tensor(R.shape([mk.MAX_NUM_KV_SPLITS * 8 // TP_SIZE * mk.HEAD_DIM]), dtype="float32", runtime_device_index=0)  # noqa: E501
                lse_partial = R.builtin.alloc_tensor(R.shape([mk.MAX_NUM_KV_SPLITS * 8 // TP_SIZE]), dtype="float32", runtime_device_index=0)  # noqa: E501
                partial_o = R.builtin.alloc_tensor(R.shape([mk.SPLIT_O_PROJECT, batch_size, mk.HIDDEN_SIZE]), dtype="float32", runtime_device_index=0)  # noqa: E501, F821
                before_o_allreduce = R.call_pure_packed("runtime.disco.nvshmem.empty", R.shape([batch_size, mk.HIDDEN_SIZE]), R.dtype("float16"), default_device, sinfo_args=[R.Tensor((batch_size, mk.HIDDEN_SIZE), "float16")]) if TP_SIZE > 1 else R.builtin.alloc_tensor(R.shape([batch_size, mk.HIDDEN_SIZE]), dtype="float16", runtime_device_index=0)  # noqa: E501, F821
                hidden_state_attn_mlp = R.call_pure_packed("runtime.disco.nvshmem.empty", R.shape([batch_size, mk.HIDDEN_SIZE]), R.dtype("float16"), default_device, sinfo_args=[R.Tensor((batch_size, mk.HIDDEN_SIZE), "float16")]) if TP_SIZE > 1 else R.builtin.alloc_tensor(R.shape([batch_size, mk.HIDDEN_SIZE]), dtype="float16", runtime_device_index=0)  # noqa: E501, F821
                partial_out_gate_up_proj = R.builtin.alloc_tensor(R.shape([mk.GATE_UP_PROJ_SPLIT_K_FACTOR, batch_size, mk.INTERMEDIATE_SIZE * 2]), dtype="float32", runtime_device_index=0)  # noqa: E501, F821
                out_gate_up_proj = R.builtin.alloc_tensor(R.shape([batch_size, 2 * mk.INTERMEDIATE_SIZE]), dtype="float16", runtime_device_index=0)  # noqa: E501, F821
                out_silu_multiply = R.builtin.alloc_tensor(R.shape([batch_size, mk.INTERMEDIATE_SIZE]), dtype="float16", runtime_device_index=0)  # noqa: E501, F821
                partial_sum_down_proj = R.builtin.alloc_tensor(R.shape([mk.DOWN_PROJ_SPLIT_K_FACTOR, batch_size, mk.HIDDEN_SIZE]), dtype="float32", runtime_device_index=0)  # noqa: E501, F821
                before_down_proj_allreduce = R.call_pure_packed("runtime.disco.nvshmem.empty", R.shape([batch_size, mk.HIDDEN_SIZE]), R.dtype("float16"), default_device, sinfo_args=[R.Tensor((batch_size, mk.HIDDEN_SIZE), "float16")]) if TP_SIZE > 1 else R.builtin.alloc_tensor(R.shape([batch_size, mk.HIDDEN_SIZE]), dtype="float16", runtime_device_index=0)  # noqa: E501, F821
                output_tensor = R.call_pure_packed("runtime.disco.nvshmem.empty", R.shape([batch_size, mk.HIDDEN_SIZE]), R.dtype("float16"), default_device, sinfo_args=[R.Tensor((batch_size, mk.HIDDEN_SIZE), "float16")]) if TP_SIZE > 1 else R.builtin.alloc_tensor(R.shape([batch_size, mk.HIDDEN_SIZE]), dtype="float16", runtime_device_index=0)  # noqa: E501, F821
                profiler_buffer = R.builtin.alloc_tensor(R.shape([mk.PROFILER_BUFFER_SIZE]), dtype="uint64", runtime_device_index=0)  # noqa: E501

                layer_res = R.call_tir_inplace(
                    cls.layer_kernel,  # noqa: F821
                    (
                        # input and output
                        input0, input1, output_tensor,
                        # weights
                        model_layers_0_self_attn_c_attn_weight1, model_layers_0_self_attn_o_proj_weight1,  # noqa: E501
                        model_layers_0_self_attn_q_norm_weight1, model_layers_0_self_attn_k_norm_weight1,  # noqa: E501
                        model_layers_0_mlp_gate_up_proj_weight1, model_layers_0_mlp_down_proj_weight1,  # noqa: E501
                        model_layers_0_post_attention_layernorm_weight1, model_norm_weight1,
                        # page cache, cos_sin cache and plan info
                        cos_sin_cache, rope_pos, kv_data[layer_id], append_pos, q_indptr, kv_indptr,  # noqa: F821
                        partial_indptr, page_kv_indices, q_len, kv_len, q_start, kv_start, kv_end, kv_head_idx,  # noqa: E501, F821
                        work_indptr, len_kv_chunk, num_qo_len, merge_indptr, merge_o_indices, inverse_indptr, inverse_indices,  # noqa: E501, F821
                        # intermediate buffer
                        partital_qkv, qkv, o, o_partial_attn, lse_partial, partial_o, before_o_allreduce,  # noqa: E501
                        hidden_state_attn_mlp, partial_out_gate_up_proj, out_gate_up_proj, out_silu_multiply,  # noqa: E501
                        partial_sum_down_proj, before_down_proj_allreduce,
                        # event tensor
                        etensor_workspace,  # noqa: F821
                        # execution queue
                        exec_queue, profiler_buffer  # noqa: F821
                    ),
                    [2, 1, 47],
                    out_sinfo=[
                        R.Tensor((batch_size, mk.HIDDEN_SIZE), dtype="float16"), # output  # noqa: E501, F821
                        R.Tensor((batch_size, mk.HIDDEN_SIZE), dtype="float32" if mk.tp_size == 1 else "float16"), # residual  # noqa: E501, F821
                        R.Tensor((mk.PROFILER_BUFFER_SIZE,), dtype="uint64"), # profiler
                    ]
                )
                R.output(layer_res)
            return layer_res

        @I.ir_module
        class Module:
            @Tx.prim_func
            def rms_norm(input_ptr: Tx.handle, weight_ptr: Tx.handle, out_ptr: Tx.handle):
                pass

            @Tx.prim_func(private=True)
            def cast(var_lv4: Tx.handle, var_compute: Tx.handle):
                Tx.func_attr({"op_pattern": 0, "tirx.noalias": True})
                batch_size = Tx.int64()
                lv4 = Tx.match_buffer(var_lv4, (batch_size, Tx.int64(1), Tx.int64(mk.VOCAB_SIZE)), "float16")  # noqa: E501
                compute = Tx.match_buffer(var_compute, (batch_size, Tx.int64(1), Tx.int64(mk.VOCAB_SIZE)))  # noqa: E501
                # with Tx.sblock("root"):
                for i0, i1, i2 in Tx.grid(batch_size, Tx.int64(1), Tx.int64(mk.VOCAB_SIZE)):
                    with Tx.sblock("compute"):
                        v_i0, v_i1, v_i2 = Tx.axis.remap("SSS", [i0, i1, i2])
                        Tx.reads(lv4[v_i0, v_i1, v_i2])
                        Tx.writes(compute[v_i0, v_i1, v_i2])
                        compute[v_i0, v_i1, v_i2] = Tx.Cast("float32", lv4[v_i0, v_i1, v_i2])

            @Tx.prim_func(private=True)
            def cast_res(var_res: Tx.handle, var_compute: Tx.handle):
                Tx.func_attr({"op_pattern": 0, "tirx.noalias": True})
                batch_size = Tx.int64()
                res = Tx.match_buffer(var_res, (batch_size, Tx.int64(mk.HIDDEN_SIZE)), "float16")
                compute = Tx.match_buffer(var_compute, (batch_size, Tx.int64(mk.HIDDEN_SIZE)))
                # with Tx.sblock("root"):
                for i0, i1 in Tx.grid(batch_size, Tx.int64(mk.HIDDEN_SIZE)):
                    with Tx.sblock("compute"):
                        v_i0, v_i1 = Tx.axis.remap("SS", [i0, i1])
                        Tx.reads(res[v_i0, v_i1])
                        Tx.writes(compute[v_i0, v_i1])
                        compute[v_i0, v_i1] = Tx.Cast("float32", res[v_i0, v_i1])

            @Tx.prim_func(private=True)
            def hgemm(A_ptr: Tx.handle, B_ptr: Tx.handle, out_ptr: Tx.handle):
                pass

            @Tx.prim_func(private=True)
            def cos_sin_cache(cos_sin_cache: Tx.handle):
                pass

            @R.function
            def cos_sin_cache_func(max_seq_len_: R.Shape(["max_seq_len"])):
                max_seq_len = Tx.int64()
                cls = Module
                with R.dataflow():
                    cache: R.Tensor((max_seq_len, mk.HEAD_DIM), dtype="float32") = R.call_tir(cls.cos_sin_cache, [], out_sinfo=R.Tensor((max_seq_len, mk.HEAD_DIM), dtype="float32") )  # noqa: E501
                    R.output(cache)
                return cache

            @Tx.prim_func(private=True)
            def layer_kernel(
                # input and output
                hidden_state_ptr: Tx.handle, # input: read-only
                residual_ptr: Tx.handle, # input & output: inplace update
                output_ptr: Tx.handle, # output

                # weight
                qkv_proj_weight_ptr: Tx.handle, # read-only
                o_proj_weight_ptr: Tx.handle, # read-only
                q_rms_weight_ptr: Tx.handle, # read-only
                k_rms_weight_ptr: Tx.handle, # read-only
                gate_up_weight_ptr: Tx.handle, # read-only
                down_weight_ptr: Tx.handle, # read-only
                attn_add_rms_weight_ptr: Tx.handle, # read-only
                mlp_add_rms_weight_ptr: Tx.handle, # read-only

                # page cache, cos_sin cache and plan info
                cos_sin_cache_ptr: Tx.handle, # read-only
                rope_pos_ptr: Tx.handle, # read-only
                kv_cache_ptr: Tx.handle, # inplace update
                append_pos_ptr: Tx.handle, # read-only
                q_indptr_ptr : Tx.handle, # read-only
                kv_indptr_ptr : Tx.handle, # read-only
                partial_indptr_ptr : Tx.handle, # read-only
                kv_indices_ptr : Tx.handle, # read-only
                q_len_ptr : Tx.handle, # read-only
                kv_len_ptr : Tx.handle, # read-only
                q_start_ptr : Tx.handle, # read-only
                kv_start_ptr : Tx.handle, # read-only
                kv_end_ptr : Tx.handle, # read-only
                kv_head_idx_ptr : Tx.handle, # read-only
                work_indptr_ptr : Tx.handle, # read-only
                len_kv_chunk_ptr : Tx.handle, # read-only
                num_qo_len_ptr: Tx.handle, # read-only
                merge_indptr_ptr: Tx.handle, # read-only
                merge_o_indices_ptr: Tx.handle, # read-only
                inverse_indptr_ptr: Tx.handle, # read-only
                inverse_indices_ptr: Tx.handle, # read-only

                # intermediate buffer
                partital_qkv_ptr: Tx.handle, # intermediate
                qkv_ptr: Tx.handle,  # intermediate
                o_ptr: Tx.handle, # intermediate
                o_partial_attn_ptr: Tx.handle, # intermediate
                lse_partial_attn_ptr: Tx.handle, # intermediate
                partial_o_ptr: Tx.handle, # intermediate
                before_o_allreduce_ptr: Tx.handle, # intermediate
                hidden_state_attn_mlp_ptr: Tx.handle, # intermediate
                partial_out_gate_up_proj_ptr: Tx.handle, # intermediate
                out_gate_up_proj_ptr: Tx.handle, # intermediate
                out_silu_multiply_ptr: Tx.handle, # intermediate
                partial_sum_down_proj_ptr: Tx.handle, # intermediate
                before_down_proj_allreduce_ptr: Tx.handle, # intermediate

                # event tensor
                etensor_workspace_ptr: Tx.handle,

                # execution queue
                exec_queue_ptr: Tx.handle,
                profiler_buffer: Tx.Buffer((mk.PROFILER_BUFFER_SIZE,), "uint64")
            ):
                pass


            @R.function(pure=True)
            def batch_decode_inner(
                input_embeds: R.Tensor(("batch_size", 1, mk.HIDDEN_SIZE), dtype="float16"),
                paged_kv_cache: R.Object,
                # rope
                cos_sin_cache: R.Tensor(("max_seq_len", mk.HEAD_DIM), dtype="float32"),
                packed_params: R.Tuple(
                    R.Tensor((mk.VOCAB_SIZE, mk.HIDDEN_SIZE), dtype="float16"),
                    #
                    *([R.Tensor(((mk.NUM_ATTENTION_HEADS + mk.NUM_KEY_VALUE_HEADS * 2) * mk.HEAD_DIM, mk.HIDDEN_SIZE), dtype="float16"),  # noqa: E501
                    R.Tensor((mk.HIDDEN_SIZE, mk.NUM_ATTENTION_HEADS * mk.HEAD_DIM), dtype="float16"),  # noqa: E501
                    R.Tensor((mk.HEAD_DIM,), dtype="float16"),
                    R.Tensor((mk.HEAD_DIM,), dtype="float16"),
                    R.Tensor((2 * mk.INTERMEDIATE_SIZE, mk.HIDDEN_SIZE), dtype="float16"),
                    R.Tensor((mk.HIDDEN_SIZE, mk.INTERMEDIATE_SIZE), dtype="float16"),
                    R.Tensor((mk.HIDDEN_SIZE,), dtype="float16"),
                    R.Tensor((mk.HIDDEN_SIZE,), dtype="float16"),] * NUM_HIDDEN_LAYERS),
                    R.Tensor((mk.HIDDEN_SIZE,), dtype="float16"),
                    R.Tensor((mk.VOCAB_SIZE, mk.HIDDEN_SIZE), dtype="float16"),
                ),
            ):
                batch_size = Tx.int64()
                Tx.int64()
                Tx.int64()
                Tx.int64()

                cls = Module
                with R.dataflow():
                    res0 = R.call_pure_packed(
                        "vm.builtin.paged_attention_kv_cache_tensor_retrieve",
                        paged_kv_cache, R.prim_value(1),
                        sinfo_args=[
                            R.Tuple([R.Tensor(None, dtype="float16")] * NUM_HIDDEN_LAYERS),
                            R.Tensor((batch_size + 1,), dtype="int32"),
                            R.Tensor(None, dtype="int32"),
                            R.Tensor((batch_size,), dtype="int32"),
                            R.Tensor((batch_size,), dtype="int32"),
                            R.Tensor((batch_size,), dtype="int32"),
                            R.Tuple([R.Prim("int64")] * 2 + [R.Tuple([R.Tensor(None, dtype="int32")] * 13)] * 2 + [R.Tensor(None, dtype="int32")] * 5),  # noqa: E501
                            R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"),
                            R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"),
                            R.Tensor(None, dtype="int32"),
                            R.Prim("int64"),
                        ],
                    )
                    (
                        _kv_data,
                        _page_kv_indptr,
                        _page_kv_indices,
                        _page_kv_last_page_len,
                        _append_pos,
                        _rope_pos,
                        attn_plan_results,
                        _inverse_indptr,
                        _inverse_indices,
                        _etensor_workspace,
                        attn_task_num,
                    ) = (
                        res0[0],
                        res0[1],
                        res0[2],
                        res0[3],
                        res0[4],
                        res0[5],
                        res0[6],
                        res0[7],
                        res0[8],
                        res0[9],
                        res0[10],
                    )
                    # kv_data = R.match_cast(kv_data_, R.Tuple([R.Tensor((max_page_num, 2, 8 // TP_SIZE, page_size, mk.HEAD_DIM), dtype="float16")] * NUM_HIDDEN_LAYERS))  # noqa: E501
                    # page_kv_indices = R.match_cast(page_kv_indices_, R.Tensor((total_page_num,), dtype="int32"))  # noqa: E501
                    task, _len_kv_chunk, _merge_indptr, _merge_o_indices, _num_qo_len = attn_plan_results[3], attn_plan_results[4], attn_plan_results[5], attn_plan_results[6], attn_plan_results[7]  # noqa: E501
                    _q_indptr, _kv_indptr, _partial_indptr, _q_len, _kv_len, _q_start, _kv_start, _kv_end, _kv_head_idx, _work_indptr = task[0], task[1], task[2], task[3], task[4], task[5], task[6], task[7], task[8], task[9]  # noqa: E501
                    # len_kv_chunk = R.match_cast(len_kv_chunk_, R.Tensor((2,), dtype="int32"))
                    # merge_indptr = R.match_cast(merge_indptr_, R.Tensor((mk.MAX_NUM_KV_SPLITS,), dtype="int32"))  # noqa: E501
                    # merge_o_indices = R.match_cast(merge_o_indices_, R.Tensor((mk.MAX_NUM_KV_SPLITS,), dtype="int32"))  # noqa: E501
                    # num_qo_len = R.match_cast(num_qo_len_, R.Tensor((1,), dtype="int32"))
                    # q_indptr = R.match_cast(q_indptr_, R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"))  # noqa: E501
                    # kv_indptr = R.match_cast(kv_indptr_, R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"))  # noqa: E501
                    # partial_indptr = R.match_cast(partial_indptr_, R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"))  # noqa: E501
                    # q_len = R.match_cast(q_len_, R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"))  # noqa: E501
                    # kv_len = R.match_cast(kv_len_, R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"))  # noqa: E501
                    # q_start = R.match_cast(q_start_, R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"))  # noqa: E501
                    # kv_start = R.match_cast(kv_start_, R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"))  # noqa: E501
                    # kv_end = R.match_cast(kv_end_, R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"))  # noqa: E501
                    # kv_head_idx = R.match_cast(kv_head_idx_, R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"))  # noqa: E501
                    # work_indptr = R.match_cast(work_indptr_, R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"))  # noqa: E501
                    R.call_pure_packed(
                        "vm.builtin.paged_attention_kv_cache_get_exec_queue",
                        paged_kv_cache,
                        R.prim_value(batch_size),
                        attn_task_num,
                        R.prim_value(-1),
                        sinfo_args=[
                            R.Tensor((148, mk.HEAD_DIM), dtype="int32"),
                        ]
                    )

                    model_layers_0_input_layernorm_weight1: R.Tensor((mk.HIDDEN_SIZE,), dtype="float16") = packed_params[7]  # noqa: E501
                    lm_head_weight1: R.Tensor((mk.VOCAB_SIZE, mk.HIDDEN_SIZE), dtype="float16") = packed_params[NUM_HIDDEN_LAYERS*8+2] # num_hidden_layers*8+2  # noqa: E501

                    rs0_ = R.reshape(input_embeds, (batch_size, mk.HIDDEN_SIZE))
                    rms_norm = R.call_tir(cls.rms_norm, (rs0_, model_layers_0_input_layernorm_weight1), out_sinfo=R.Tensor((batch_size, mk.HIDDEN_SIZE), dtype="float16"))  # noqa: E501
                    rs0 = R.call_tir(cls.cast_res, (rs0_,), out_sinfo=R.Tensor((batch_size, mk.HIDDEN_SIZE), dtype="float32")) if mk.tp_size == 1 else rs0_  # noqa: E501

                    o_layer0 = call_qwen3_layer(rms_norm, rs0, 0)
                    o_layer1 = call_qwen3_layer(o_layer0[0], o_layer0[1], 1)
                    o_layer2 = call_qwen3_layer(o_layer1[0], o_layer1[1], 2)
                    o_layer3 = call_qwen3_layer(o_layer2[0], o_layer2[1], 3)
                    o_layer4 = call_qwen3_layer(o_layer3[0], o_layer3[1], 4)
                    o_layer5 = call_qwen3_layer(o_layer4[0], o_layer4[1], 5)
                    o_layer6 = call_qwen3_layer(o_layer5[0], o_layer5[1], 6)
                    o_layer7 = call_qwen3_layer(o_layer6[0], o_layer6[1], 7)
                    o_layer8 = call_qwen3_layer(o_layer7[0], o_layer7[1], 8)
                    o_layer9 = call_qwen3_layer(o_layer8[0], o_layer8[1], 9)
                    o_layer10 = call_qwen3_layer(o_layer9[0], o_layer9[1], 10)
                    o_layer11 = call_qwen3_layer(o_layer10[0], o_layer10[1], 11)
                    o_layer12 = call_qwen3_layer(o_layer11[0], o_layer11[1], 12)
                    o_layer13 = call_qwen3_layer(o_layer12[0], o_layer12[1], 13)
                    o_layer14 = call_qwen3_layer(o_layer13[0], o_layer13[1], 14)
                    o_layer15 = call_qwen3_layer(o_layer14[0], o_layer14[1], 15)
                    o_layer16 = call_qwen3_layer(o_layer15[0], o_layer15[1], 16)
                    o_layer17 = call_qwen3_layer(o_layer16[0], o_layer16[1], 17)
                    o_layer18 = call_qwen3_layer(o_layer17[0], o_layer17[1], 18)
                    o_layer19 = call_qwen3_layer(o_layer18[0], o_layer18[1], 19)
                    o_layer20 = call_qwen3_layer(o_layer19[0], o_layer19[1], 20)
                    o_layer21 = call_qwen3_layer(o_layer20[0], o_layer20[1], 21)
                    o_layer22 = call_qwen3_layer(o_layer21[0], o_layer21[1], 22)
                    o_layer23 = call_qwen3_layer(o_layer22[0], o_layer22[1], 23)
                    o_layer24 = call_qwen3_layer(o_layer23[0], o_layer23[1], 24)
                    o_layer25 = call_qwen3_layer(o_layer24[0], o_layer24[1], 25)
                    o_layer26 = call_qwen3_layer(o_layer25[0], o_layer25[1], 26)
                    o_layer27 = call_qwen3_layer(o_layer26[0], o_layer26[1], 27)
                    o_layer28 = call_qwen3_layer(o_layer27[0], o_layer27[1], 28)
                    o_layer29 = call_qwen3_layer(o_layer28[0], o_layer28[1], 29)
                    o_layer30 = call_qwen3_layer(o_layer29[0], o_layer29[1], 30)
                    o_layer31 = call_qwen3_layer(o_layer30[0], o_layer30[1], 31)
                    o_layer32 = call_qwen3_layer(o_layer31[0], o_layer31[1], 32)
                    o_layer33 = call_qwen3_layer(o_layer32[0], o_layer32[1], 33)
                    o_layer34 = call_qwen3_layer(o_layer33[0], o_layer33[1], 34)
                    o_layer35 = call_qwen3_layer(o_layer34[0], o_layer34[1], 35)
                    o_layer36 = call_qwen3_layer(o_layer35[0], o_layer35[1], 36)
                    o_layer37 = call_qwen3_layer(o_layer36[0], o_layer36[1], 37)
                    o_layer38 = call_qwen3_layer(o_layer37[0], o_layer37[1], 38)
                    o_layer39 = call_qwen3_layer(o_layer38[0], o_layer38[1], 39)
                    o_layer40 = call_qwen3_layer(o_layer39[0], o_layer39[1], 40)
                    o_layer41 = call_qwen3_layer(o_layer40[0], o_layer40[1], 41)
                    o_layer42 = call_qwen3_layer(o_layer41[0], o_layer41[1], 42)
                    o_layer43 = call_qwen3_layer(o_layer42[0], o_layer42[1], 43)
                    o_layer44 = call_qwen3_layer(o_layer43[0], o_layer43[1], 44)
                    o_layer45 = call_qwen3_layer(o_layer44[0], o_layer44[1], 45)
                    o_layer46 = call_qwen3_layer(o_layer45[0], o_layer45[1], 46)
                    o_layer47 = call_qwen3_layer(o_layer46[0], o_layer46[1], 47)
                    o_layer48 = call_qwen3_layer(o_layer47[0], o_layer47[1], 48)
                    o_layer49 = call_qwen3_layer(o_layer48[0], o_layer48[1], 49)
                    o_layer50 = call_qwen3_layer(o_layer49[0], o_layer49[1], 50)
                    o_layer51 = call_qwen3_layer(o_layer50[0], o_layer50[1], 51)
                    o_layer52 = call_qwen3_layer(o_layer51[0], o_layer51[1], 52)
                    o_layer53 = call_qwen3_layer(o_layer52[0], o_layer52[1], 53)
                    o_layer54 = call_qwen3_layer(o_layer53[0], o_layer53[1], 54)
                    o_layer55 = call_qwen3_layer(o_layer54[0], o_layer54[1], 55)
                    o_layer56 = call_qwen3_layer(o_layer55[0], o_layer55[1], 56)
                    o_layer57 = call_qwen3_layer(o_layer56[0], o_layer56[1], 57)
                    o_layer58 = call_qwen3_layer(o_layer57[0], o_layer57[1], 58)
                    o_layer59 = call_qwen3_layer(o_layer58[0], o_layer58[1], 59)
                    o_layer60 = call_qwen3_layer(o_layer59[0], o_layer59[1], 60)
                    o_layer61 = call_qwen3_layer(o_layer60[0], o_layer60[1], 61)
                    o_layer62 = call_qwen3_layer(o_layer61[0], o_layer61[1], 62)
                    o_layer63 = call_qwen3_layer(o_layer62[0], o_layer62[1], 63)

                    # permute_dims4 = R.permute_dims(lm_head_weight1, axes=None)
                    # lv4: R.Tensor((batch_size, mk.VOCAB_SIZE), dtype="float16") = R.matmul(rms_norm4, permute_dims4, out_dtype="void")  # noqa: E501
                    lv4 = R.call_tir(cls.hgemm, (o_layer63[0], lm_head_weight1), out_sinfo=R.Tensor((batch_size, mk.VOCAB_SIZE), dtype="float16"))  # noqa: E501
                    lv4_rs = R.reshape(lv4, (batch_size, 1, mk.VOCAB_SIZE))
                    astype = R.call_tir(cls.cast, (lv4_rs,), out_sinfo=R.Tensor((batch_size, 1, mk.VOCAB_SIZE), dtype="float32"))  # noqa: E501

                    profiler = (
                        o_layer0[2], o_layer1[2], o_layer2[2], o_layer3[2], o_layer4[2], o_layer5[2],  # noqa: E501
                        o_layer6[2], o_layer7[2], o_layer8[2], o_layer9[2], o_layer10[2], o_layer11[2],  # noqa: E501
                        o_layer12[2], o_layer13[2], o_layer14[2], o_layer15[2], o_layer16[2], o_layer17[2],  # noqa: E501
                        o_layer18[2], o_layer19[2], o_layer20[2], o_layer21[2], o_layer22[2], o_layer23[2],  # noqa: E501
                        o_layer24[2], o_layer25[2], o_layer26[2], o_layer27[2], o_layer28[2], o_layer29[2],  # noqa: E501
                        o_layer30[2], o_layer31[2], o_layer32[2], o_layer33[2], o_layer34[2], o_layer35[2],  # noqa: E501
                        o_layer36[2], o_layer37[2], o_layer38[2], o_layer39[2], o_layer40[2], o_layer41[2],  # noqa: E501
                        o_layer42[2], o_layer43[2], o_layer44[2], o_layer45[2], o_layer46[2], o_layer47[2],  # noqa: E501
                        o_layer48[2], o_layer49[2], o_layer50[2], o_layer51[2], o_layer52[2], o_layer53[2],  # noqa: E501
                        o_layer54[2], o_layer55[2], o_layer56[2], o_layer57[2], o_layer58[2], o_layer59[2],  # noqa: E501
                        o_layer60[2], o_layer61[2], o_layer62[2], o_layer63[2]
                    )

                    gv1 = astype, paged_kv_cache, profiler
                    R.output(gv1)
                return gv1


            @R.function(pure=False)
            def batch_decode(
                input_embeds: R.Tensor(("batch_size", 1, mk.HIDDEN_SIZE), dtype="float16"),
                paged_kv_cache: R.Object,
                # rope
                cos_sin_cache: R.Tensor(("max_seq_len", mk.HEAD_DIM), dtype="float32"),
                packed_params: R.Tuple(
                    R.Tensor((mk.VOCAB_SIZE, mk.HIDDEN_SIZE), dtype="float16"),
                    #
                    *([R.Tensor(((mk.NUM_ATTENTION_HEADS + mk.NUM_KEY_VALUE_HEADS * 2) * mk.HEAD_DIM, mk.HIDDEN_SIZE), dtype="float16"),  # noqa: E501
                    R.Tensor((mk.HIDDEN_SIZE, mk.NUM_ATTENTION_HEADS * mk.HEAD_DIM), dtype="float16"),  # noqa: E501
                    R.Tensor((mk.HEAD_DIM,), dtype="float16"),
                    R.Tensor((mk.HEAD_DIM,), dtype="float16"),
                    R.Tensor((2 * mk.INTERMEDIATE_SIZE, mk.HIDDEN_SIZE), dtype="float16"),
                    R.Tensor((mk.HIDDEN_SIZE, mk.INTERMEDIATE_SIZE), dtype="float16"),
                    R.Tensor((mk.HIDDEN_SIZE,), dtype="float16"),
                    R.Tensor((mk.HIDDEN_SIZE,), dtype="float16"),] * NUM_HIDDEN_LAYERS),
                    R.Tensor((mk.HIDDEN_SIZE,), dtype="float16"),
                    R.Tensor((mk.VOCAB_SIZE, mk.HIDDEN_SIZE), dtype="float16"),
                ),
            ):
                cls = Module
                model_output = cls.batch_decode_inner(
                    input_embeds,
                    paged_kv_cache,
                    cos_sin_cache,
                    packed_params,
                )
                if PROFILER_ON:
                    rank = R.call_packed("runtime.disco.worker_id", sinfo_args=[R.Shape()]) if TP_SIZE > 1 else R.prim_value(-1)  # noqa: E501
                    _ = R.call_packed(
                        "megakernel.export_trace",
                        model_output[2],
                        rank,
                        sinfo_args=[],
                    )
                else:
                    # there must be some dummy thing in else statement to avoid error
                    _ = R.prim_value(-1)

                res = model_output[0], model_output[1]
                return res

        # fmt: on
        return Module

    def dynamic_mod():
        # fmt: off
        @R.macro(hygienic=False)
        def call_qwen3_layer(input0, input1, layer_id):
            with R.dataflow():
                # 8i+1, 8i+2, 8i+3, 8i+4, 8i+5, 8i+6, 8i+8, 8i+15 if i<num_hidden_layers-1 else 8i+9
                model_layers_0_self_attn_c_attn_weight1: R.Tensor(((mk.NUM_ATTENTION_HEADS + mk.NUM_KEY_VALUE_HEADS * 2) * mk.HEAD_DIM, mk.HIDDEN_SIZE), dtype="float16") = packed_params[8*layer_id+1]  # noqa: E501, F821
                model_layers_0_self_attn_o_proj_weight1: R.Tensor((mk.HIDDEN_SIZE, mk.NUM_ATTENTION_HEADS * mk.HEAD_DIM), dtype="float16") = packed_params[8*layer_id+2]  # noqa: E501, F821
                model_layers_0_self_attn_q_norm_weight1: R.Tensor((mk.HEAD_DIM,), dtype="float16") = packed_params[8*layer_id+3]  # noqa: E501, F821
                model_layers_0_self_attn_k_norm_weight1: R.Tensor((mk.HEAD_DIM,), dtype="float16") = packed_params[8*layer_id+4]  # noqa: E501, F821
                model_layers_0_mlp_gate_up_proj_weight1: R.Tensor((2 * mk.INTERMEDIATE_SIZE, mk.HIDDEN_SIZE), dtype="float16") = packed_params[8*layer_id+5]  # noqa: E501, F821
                model_layers_0_mlp_down_proj_weight1: R.Tensor((mk.HIDDEN_SIZE, mk.INTERMEDIATE_SIZE), dtype="float16") = packed_params[8*layer_id+6]  # noqa: E501, F821
                model_layers_0_post_attention_layernorm_weight1: R.Tensor((mk.HIDDEN_SIZE,), dtype="float16") = packed_params[8*layer_id+8]  # noqa: E501, F821
                model_norm_weight1: R.Tensor((mk.HIDDEN_SIZE,), dtype="float16") = packed_params[8*layer_id+15 if layer_id < NUM_HIDDEN_LAYERS-1 else 8*layer_id+9]  # noqa: E501, F821


                default_device = R.call_pure_packed("runtime.disco.device", sinfo_args=[R.Object])
                partital_qkv = R.builtin.alloc_tensor(R.shape([mk.SPLIT_QKV_PROJECT, batch_size, (mk.NUM_ATTENTION_HEADS + mk.NUM_KEY_VALUE_HEADS * 2) * mk.HEAD_DIM]), dtype="float32", runtime_device_index=0)  # noqa: E501, F821
                qkv = R.builtin.alloc_tensor(R.shape([batch_size, mk.NUM_ATTENTION_HEADS + mk.NUM_KEY_VALUE_HEADS * 2, mk.HEAD_DIM]), dtype="float16", runtime_device_index=0)  # noqa: E501, F821
                o = R.builtin.alloc_tensor(R.shape([batch_size, mk.NUM_ATTENTION_HEADS, mk.HEAD_DIM]), dtype="float16", runtime_device_index=0)  # noqa: E501, F821
                o_partial_attn = R.builtin.alloc_tensor(R.shape([mk.MAX_NUM_KV_SPLITS * 8 // TP_SIZE * mk.HEAD_DIM]), dtype="float32", runtime_device_index=0)  # noqa: E501
                lse_partial = R.builtin.alloc_tensor(R.shape([mk.MAX_NUM_KV_SPLITS * 8 // TP_SIZE]), dtype="float32", runtime_device_index=0)  # noqa: E501
                partial_o = R.builtin.alloc_tensor(R.shape([mk.SPLIT_O_PROJECT, batch_size, mk.HIDDEN_SIZE]), dtype="float32", runtime_device_index=0)  # noqa: E501, F821
                before_o_allreduce = R.call_pure_packed("runtime.disco.nvshmem.empty", R.shape([batch_size, mk.HIDDEN_SIZE]), R.dtype("float16"), default_device, sinfo_args=[R.Tensor((batch_size, mk.HIDDEN_SIZE), "float16")]) if TP_SIZE > 1 else R.builtin.alloc_tensor(R.shape([batch_size, mk.HIDDEN_SIZE]), dtype="float16", runtime_device_index=0)  # noqa: E501, F821
                hidden_state_attn_mlp = R.call_pure_packed("runtime.disco.nvshmem.empty", R.shape([batch_size, mk.HIDDEN_SIZE]), R.dtype("float16"), default_device, sinfo_args=[R.Tensor((batch_size, mk.HIDDEN_SIZE), "float16")]) if TP_SIZE > 1 else R.builtin.alloc_tensor(R.shape([batch_size, mk.HIDDEN_SIZE]), dtype="float16", runtime_device_index=0)  # noqa: E501, F821
                partial_out_gate_up_proj = R.builtin.alloc_tensor(R.shape([mk.GATE_UP_PROJ_SPLIT_K_FACTOR, batch_size, mk.INTERMEDIATE_SIZE * 2]), dtype="float32", runtime_device_index=0)  # noqa: E501, F821
                out_gate_up_proj = R.builtin.alloc_tensor(R.shape([batch_size, 2 * mk.INTERMEDIATE_SIZE]), dtype="float16", runtime_device_index=0)  # noqa: E501, F821
                out_silu_multiply = R.builtin.alloc_tensor(R.shape([batch_size, mk.INTERMEDIATE_SIZE]), dtype="float16", runtime_device_index=0)  # noqa: E501, F821
                partial_sum_down_proj = R.builtin.alloc_tensor(R.shape([mk.DOWN_PROJ_SPLIT_K_FACTOR, batch_size, mk.HIDDEN_SIZE]), dtype="float32", runtime_device_index=0)  # noqa: E501, F821
                before_down_proj_allreduce = R.call_pure_packed("runtime.disco.nvshmem.empty", R.shape([batch_size, mk.HIDDEN_SIZE]), R.dtype("float16"), default_device, sinfo_args=[R.Tensor((batch_size, mk.HIDDEN_SIZE), "float16")]) if TP_SIZE > 1 else R.builtin.alloc_tensor(R.shape([batch_size, mk.HIDDEN_SIZE]), dtype="float16", runtime_device_index=0)  # noqa: E501, F821
                output_tensor = R.call_pure_packed("runtime.disco.nvshmem.empty", R.shape([batch_size, mk.HIDDEN_SIZE]), R.dtype("float16"), default_device, sinfo_args=[R.Tensor((batch_size, mk.HIDDEN_SIZE), "float16")]) if TP_SIZE > 1 else R.builtin.alloc_tensor(R.shape([batch_size, mk.HIDDEN_SIZE]), dtype="float16", runtime_device_index=0)  # noqa: E501, F821
                profiler_buffer = R.builtin.alloc_tensor(R.shape([mk.PROFILER_BUFFER_SIZE]), dtype="uint64", runtime_device_index=0)  # noqa: E501
                exec_queue = R.call_pure_packed(
                    "vm.builtin.paged_attention_kv_cache_get_exec_queue",
                    paged_kv_cache,  # noqa: F821
                    R.prim_value(batch_size),  # noqa: F821
                    attn_task_num,  # noqa: F821
                    R.prim_value(layer_id),
                    sinfo_args=[
                        R.Tensor((DynamicTileScheduler.MAX_TASKS,), dtype="int32"),
                        R.Tensor((1,), dtype="int32"),
                        R.Tensor((1,), dtype="int32"),
                    ]
                )
                queue_tasks, queue_head, queue_tail = exec_queue[0], exec_queue[1], exec_queue[2]


                layer_res = R.call_tir_inplace(
                    cls.layer_kernel,  # noqa: F821
                    (
                        # input and output
                        input0, input1, output_tensor,
                        # weights
                        model_layers_0_self_attn_c_attn_weight1, model_layers_0_self_attn_o_proj_weight1,  # noqa: E501
                        model_layers_0_self_attn_q_norm_weight1, model_layers_0_self_attn_k_norm_weight1,  # noqa: E501
                        model_layers_0_mlp_gate_up_proj_weight1, model_layers_0_mlp_down_proj_weight1,  # noqa: E501
                        model_layers_0_post_attention_layernorm_weight1, model_norm_weight1,
                        # page cache, cos_sin cache and plan info
                        cos_sin_cache, rope_pos, kv_data[layer_id], append_pos, q_indptr, kv_indptr,  # noqa: F821
                        partial_indptr, page_kv_indices, q_len, kv_len, q_start, kv_start, kv_end, kv_head_idx,  # noqa: E501, F821
                        work_indptr, len_kv_chunk, num_qo_len, merge_indptr, merge_o_indices, inverse_indptr, inverse_indices,  # noqa: E501, F821
                        # intermediate buffer
                        partital_qkv, qkv, o, o_partial_attn, lse_partial, partial_o, before_o_allreduce,  # noqa: E501
                        hidden_state_attn_mlp, partial_out_gate_up_proj, out_gate_up_proj, out_silu_multiply,  # noqa: E501
                        partial_sum_down_proj, before_down_proj_allreduce,
                        # event tensor
                        etensor_workspace,  # noqa: F821
                        # execution queue
                        queue_tasks, queue_head, queue_tail, profiler_buffer
                    ),
                    [2, 1, 49],
                    out_sinfo=[
                        R.Tensor((batch_size, mk.HIDDEN_SIZE), dtype="float16"), # output  # noqa: E501, F821
                        R.Tensor((batch_size, mk.HIDDEN_SIZE), dtype="float32" if mk.tp_size == 1 else "float16"), # residual  # noqa: E501, F821
                        R.Tensor((mk.PROFILER_BUFFER_SIZE,), dtype="uint64"), # profiler
                    ]
                )
                R.output(layer_res)
            return layer_res

        @I.ir_module
        class Module:
            @Tx.prim_func
            def rms_norm(input_ptr: Tx.handle, weight_ptr: Tx.handle, out_ptr: Tx.handle):
                pass

            @Tx.prim_func(private=True)
            def cast(var_lv4: Tx.handle, var_compute: Tx.handle):
                Tx.func_attr({"op_pattern": 0, "tirx.noalias": True})
                batch_size = Tx.int64()
                lv4 = Tx.match_buffer(var_lv4, (batch_size, Tx.int64(1), Tx.int64(mk.VOCAB_SIZE)), "float16")  # noqa: E501
                compute = Tx.match_buffer(var_compute, (batch_size, Tx.int64(1), Tx.int64(mk.VOCAB_SIZE)))  # noqa: E501
                # with Tx.sblock("root"):
                for i0, i1, i2 in Tx.grid(batch_size, Tx.int64(1), Tx.int64(mk.VOCAB_SIZE)):
                    with Tx.sblock("compute"):
                        v_i0, v_i1, v_i2 = Tx.axis.remap("SSS", [i0, i1, i2])
                        Tx.reads(lv4[v_i0, v_i1, v_i2])
                        Tx.writes(compute[v_i0, v_i1, v_i2])
                        compute[v_i0, v_i1, v_i2] = Tx.Cast("float32", lv4[v_i0, v_i1, v_i2])

            @Tx.prim_func(private=True)
            def cast_res(var_res: Tx.handle, var_compute: Tx.handle):
                Tx.func_attr({"op_pattern": 0, "tirx.noalias": True})
                batch_size = Tx.int64()
                res = Tx.match_buffer(var_res, (batch_size, Tx.int64(mk.HIDDEN_SIZE)), "float16")
                compute = Tx.match_buffer(var_compute, (batch_size, Tx.int64(mk.HIDDEN_SIZE)))
                # with Tx.sblock("root"):
                for i0, i1 in Tx.grid(batch_size, Tx.int64(mk.HIDDEN_SIZE)):
                    with Tx.sblock("compute"):
                        v_i0, v_i1 = Tx.axis.remap("SS", [i0, i1])
                        Tx.reads(res[v_i0, v_i1])
                        Tx.writes(compute[v_i0, v_i1])
                        compute[v_i0, v_i1] = Tx.Cast("float32", res[v_i0, v_i1])

            @Tx.prim_func(private=True)
            def hgemm(A_ptr: Tx.handle, B_ptr: Tx.handle, out_ptr: Tx.handle):
                pass

            @Tx.prim_func(private=True)
            def cos_sin_cache(cos_sin_cache: Tx.handle):
                pass

            @R.function
            def cos_sin_cache_func(max_seq_len_: R.Shape(["max_seq_len"])):
                max_seq_len = Tx.int64()
                cls = Module
                with R.dataflow():
                    cache: R.Tensor((max_seq_len, mk.HEAD_DIM), dtype="float32") = R.call_tir(cls.cos_sin_cache, [], out_sinfo=R.Tensor((max_seq_len, mk.HEAD_DIM), dtype="float32") )  # noqa: E501
                    R.output(cache)
                return cache

            @Tx.prim_func(private=True)
            def layer_kernel(
                # input and output
                hidden_state_ptr: Tx.handle, # input: read-only
                residual_ptr: Tx.handle, # input & output: inplace update
                output_ptr: Tx.handle, # output

                # weight
                qkv_proj_weight_ptr: Tx.handle, # read-only
                o_proj_weight_ptr: Tx.handle, # read-only
                q_rms_weight_ptr: Tx.handle, # read-only
                k_rms_weight_ptr: Tx.handle, # read-only
                gate_up_weight_ptr: Tx.handle, # read-only
                down_weight_ptr: Tx.handle, # read-only
                attn_add_rms_weight_ptr: Tx.handle, # read-only
                mlp_add_rms_weight_ptr: Tx.handle, # read-only

                # page cache, cos_sin cache and plan info
                cos_sin_cache_ptr: Tx.handle, # read-only
                rope_pos_ptr: Tx.handle, # read-only
                kv_cache_ptr: Tx.handle, # inplace update
                append_pos_ptr: Tx.handle, # read-only
                q_indptr_ptr : Tx.handle, # read-only
                kv_indptr_ptr : Tx.handle, # read-only
                partial_indptr_ptr : Tx.handle, # read-only
                kv_indices_ptr : Tx.handle, # read-only
                q_len_ptr : Tx.handle, # read-only
                kv_len_ptr : Tx.handle, # read-only
                q_start_ptr : Tx.handle, # read-only
                kv_start_ptr : Tx.handle, # read-only
                kv_end_ptr : Tx.handle, # read-only
                kv_head_idx_ptr : Tx.handle, # read-only
                work_indptr_ptr : Tx.handle, # read-only
                len_kv_chunk_ptr : Tx.handle, # read-only
                num_qo_len_ptr: Tx.handle, # read-only
                merge_indptr_ptr: Tx.handle, # read-only
                merge_o_indices_ptr: Tx.handle, # read-only
                inverse_indptr_ptr: Tx.handle, # read-only
                inverse_indices_ptr: Tx.handle, # read-only

                # intermediate buffer
                partital_qkv_ptr: Tx.handle, # intermediate
                qkv_ptr: Tx.handle,  # intermediate
                o_ptr: Tx.handle, # intermediate
                o_partial_attn_ptr: Tx.handle, # intermediate
                lse_partial_attn_ptr: Tx.handle, # intermediate
                partial_o_ptr: Tx.handle, # intermediate
                before_o_allreduce_ptr: Tx.handle, # intermediate
                hidden_state_attn_mlp_ptr: Tx.handle, # intermediate
                partial_out_gate_up_proj_ptr: Tx.handle, # intermediate
                out_gate_up_proj_ptr: Tx.handle, # intermediate
                out_silu_multiply_ptr: Tx.handle, # intermediate
                partial_sum_down_proj_ptr: Tx.handle, # intermediate
                before_down_proj_allreduce_ptr: Tx.handle, # intermediate

                # event tensor
                etensor_workspace_ptr: Tx.handle,

                # execution queue
                queue_task_ptr: Tx.handle,
                queue_head_ptr: Tx.handle,
                queue_tail_ptr: Tx.handle,
                profiler_buffer: Tx.Buffer((mk.PROFILER_BUFFER_SIZE,), "uint64")
            ):
                pass


            @R.function(pure=True)
            def batch_decode_inner(
                input_embeds: R.Tensor(("batch_size", 1, mk.HIDDEN_SIZE), dtype="float16"),
                paged_kv_cache: R.Object,
                # rope
                cos_sin_cache: R.Tensor(("max_seq_len", mk.HEAD_DIM), dtype="float32"),
                packed_params: R.Tuple(
                    R.Tensor((mk.VOCAB_SIZE, mk.HIDDEN_SIZE), dtype="float16"),
                    #
                    *([R.Tensor(((mk.NUM_ATTENTION_HEADS + mk.NUM_KEY_VALUE_HEADS * 2) * mk.HEAD_DIM, mk.HIDDEN_SIZE), dtype="float16"),  # noqa: E501
                    R.Tensor((mk.HIDDEN_SIZE, mk.NUM_ATTENTION_HEADS * mk.HEAD_DIM), dtype="float16"),  # noqa: E501
                    R.Tensor((mk.HEAD_DIM,), dtype="float16"),
                    R.Tensor((mk.HEAD_DIM,), dtype="float16"),
                    R.Tensor((2 * mk.INTERMEDIATE_SIZE, mk.HIDDEN_SIZE), dtype="float16"),
                    R.Tensor((mk.HIDDEN_SIZE, mk.INTERMEDIATE_SIZE), dtype="float16"),
                    R.Tensor((mk.HIDDEN_SIZE,), dtype="float16"),
                    R.Tensor((mk.HIDDEN_SIZE,), dtype="float16"),] * NUM_HIDDEN_LAYERS),
                    R.Tensor((mk.HIDDEN_SIZE,), dtype="float16"),
                    R.Tensor((mk.VOCAB_SIZE, mk.HIDDEN_SIZE), dtype="float16"),
                ),
            ):
                batch_size = Tx.int64()
                total_page_num = Tx.int64()
                max_page_num = Tx.int64()
                page_size = Tx.int64()

                cls = Module
                with R.dataflow():
                    res0 = R.call_pure_packed(
                        "vm.builtin.paged_attention_kv_cache_tensor_retrieve",
                        paged_kv_cache, R.prim_value(1),
                        sinfo_args=[
                            R.Tuple([R.Tensor(None, dtype="float16")] * NUM_HIDDEN_LAYERS),
                            R.Tensor((batch_size + 1,), dtype="int32"),
                            R.Tensor(None, dtype="int32"),
                            R.Tensor((batch_size,), dtype="int32"),
                            R.Tensor((batch_size,), dtype="int32"),
                            R.Tensor((batch_size,), dtype="int32"),
                            R.Tuple([R.Prim("int64")] * 2 + [R.Tuple([R.Tensor(None, dtype="int32")] * 13)] * 2 + [R.Tensor(None, dtype="int32")] * 5),  # noqa: E501
                            R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"),
                            R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"),
                            R.Tensor(None, dtype="int32"),
                            R.Prim("int64"),
                        ],
                    )
                    (
                        kv_data_,
                        _page_kv_indptr,
                        page_kv_indices_,
                        _page_kv_last_page_len,
                        _append_pos,
                        _rope_pos,
                        attn_plan_results,
                        _inverse_indptr,
                        _inverse_indices,
                        _etensor_workspace,
                        _attn_task_num,
                    ) = (
                        res0[0],
                        res0[1],
                        res0[2],
                        res0[3],
                        res0[4],
                        res0[5],
                        res0[6],
                        res0[7],
                        res0[8],
                        res0[9],
                        res0[10],
                    )
                    R.match_cast(kv_data_, R.Tuple([R.Tensor((max_page_num, 2, 8 // TP_SIZE, page_size, mk.HEAD_DIM), dtype="float16")] * NUM_HIDDEN_LAYERS))  # noqa: E501
                    R.match_cast(page_kv_indices_, R.Tensor((total_page_num,), dtype="int32"))
                    task, len_kv_chunk_, merge_indptr_, merge_o_indices_, num_qo_len_ = attn_plan_results[3], attn_plan_results[4], attn_plan_results[5], attn_plan_results[6], attn_plan_results[7]  # noqa: E501
                    q_indptr_, kv_indptr_, partial_indptr_, q_len_, kv_len_, q_start_, kv_start_, kv_end_, kv_head_idx_, work_indptr_ = task[0], task[1], task[2], task[3], task[4], task[5], task[6], task[7], task[8], task[9]  # noqa: E501
                    R.match_cast(len_kv_chunk_, R.Tensor((2,), dtype="int32"))
                    R.match_cast(merge_indptr_, R.Tensor((mk.MAX_NUM_KV_SPLITS,), dtype="int32"))
                    R.match_cast(merge_o_indices_, R.Tensor((mk.MAX_NUM_KV_SPLITS,), dtype="int32"))
                    R.match_cast(num_qo_len_, R.Tensor((1,), dtype="int32"))
                    R.match_cast(q_indptr_, R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"))
                    R.match_cast(kv_indptr_, R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"))
                    R.match_cast(partial_indptr_, R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"))  # noqa: E501
                    R.match_cast(q_len_, R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"))
                    R.match_cast(kv_len_, R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"))
                    R.match_cast(q_start_, R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"))
                    R.match_cast(kv_start_, R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"))
                    R.match_cast(kv_end_, R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"))
                    R.match_cast(kv_head_idx_, R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"))
                    R.match_cast(work_indptr_, R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"))
                    model_layers_0_input_layernorm_weight1: R.Tensor((mk.HIDDEN_SIZE,), dtype="float16") = packed_params[7]  # noqa: E501
                    lm_head_weight1: R.Tensor((mk.VOCAB_SIZE, mk.HIDDEN_SIZE), dtype="float16") = packed_params[NUM_HIDDEN_LAYERS*8+2] # num_hidden_layers*8+2  # noqa: E501

                    rs0_ = R.reshape(input_embeds, (batch_size, mk.HIDDEN_SIZE))
                    rms_norm = R.call_tir(cls.rms_norm, (rs0_, model_layers_0_input_layernorm_weight1), out_sinfo=R.Tensor((batch_size, mk.HIDDEN_SIZE), dtype="float16"))  # noqa: E501
                    rs0 = R.call_tir(cls.cast_res, (rs0_,), out_sinfo=R.Tensor((batch_size, mk.HIDDEN_SIZE), dtype="float32")) if mk.tp_size == 1 else rs0_  # noqa: E501

                    o_layer0 = call_qwen3_layer(rms_norm, rs0, 0)
                    o_layer1 = call_qwen3_layer(o_layer0[0], o_layer0[1], 1)
                    o_layer2 = call_qwen3_layer(o_layer1[0], o_layer1[1], 2)
                    o_layer3 = call_qwen3_layer(o_layer2[0], o_layer2[1], 3)
                    o_layer4 = call_qwen3_layer(o_layer3[0], o_layer3[1], 4)
                    o_layer5 = call_qwen3_layer(o_layer4[0], o_layer4[1], 5)
                    o_layer6 = call_qwen3_layer(o_layer5[0], o_layer5[1], 6)
                    o_layer7 = call_qwen3_layer(o_layer6[0], o_layer6[1], 7)
                    o_layer8 = call_qwen3_layer(o_layer7[0], o_layer7[1], 8)
                    o_layer9 = call_qwen3_layer(o_layer8[0], o_layer8[1], 9)
                    o_layer10 = call_qwen3_layer(o_layer9[0], o_layer9[1], 10)
                    o_layer11 = call_qwen3_layer(o_layer10[0], o_layer10[1], 11)
                    o_layer12 = call_qwen3_layer(o_layer11[0], o_layer11[1], 12)
                    o_layer13 = call_qwen3_layer(o_layer12[0], o_layer12[1], 13)
                    o_layer14 = call_qwen3_layer(o_layer13[0], o_layer13[1], 14)
                    o_layer15 = call_qwen3_layer(o_layer14[0], o_layer14[1], 15)
                    o_layer16 = call_qwen3_layer(o_layer15[0], o_layer15[1], 16)
                    o_layer17 = call_qwen3_layer(o_layer16[0], o_layer16[1], 17)
                    o_layer18 = call_qwen3_layer(o_layer17[0], o_layer17[1], 18)
                    o_layer19 = call_qwen3_layer(o_layer18[0], o_layer18[1], 19)
                    o_layer20 = call_qwen3_layer(o_layer19[0], o_layer19[1], 20)
                    o_layer21 = call_qwen3_layer(o_layer20[0], o_layer20[1], 21)
                    o_layer22 = call_qwen3_layer(o_layer21[0], o_layer21[1], 22)
                    o_layer23 = call_qwen3_layer(o_layer22[0], o_layer22[1], 23)
                    o_layer24 = call_qwen3_layer(o_layer23[0], o_layer23[1], 24)
                    o_layer25 = call_qwen3_layer(o_layer24[0], o_layer24[1], 25)
                    o_layer26 = call_qwen3_layer(o_layer25[0], o_layer25[1], 26)
                    o_layer27 = call_qwen3_layer(o_layer26[0], o_layer26[1], 27)
                    o_layer28 = call_qwen3_layer(o_layer27[0], o_layer27[1], 28)
                    o_layer29 = call_qwen3_layer(o_layer28[0], o_layer28[1], 29)
                    o_layer30 = call_qwen3_layer(o_layer29[0], o_layer29[1], 30)
                    o_layer31 = call_qwen3_layer(o_layer30[0], o_layer30[1], 31)
                    o_layer32 = call_qwen3_layer(o_layer31[0], o_layer31[1], 32)
                    o_layer33 = call_qwen3_layer(o_layer32[0], o_layer32[1], 33)
                    o_layer34 = call_qwen3_layer(o_layer33[0], o_layer33[1], 34)
                    o_layer35 = call_qwen3_layer(o_layer34[0], o_layer34[1], 35)
                    o_layer36 = call_qwen3_layer(o_layer35[0], o_layer35[1], 36)
                    o_layer37 = call_qwen3_layer(o_layer36[0], o_layer36[1], 37)
                    o_layer38 = call_qwen3_layer(o_layer37[0], o_layer37[1], 38)
                    o_layer39 = call_qwen3_layer(o_layer38[0], o_layer38[1], 39)
                    o_layer40 = call_qwen3_layer(o_layer39[0], o_layer39[1], 40)
                    o_layer41 = call_qwen3_layer(o_layer40[0], o_layer40[1], 41)
                    o_layer42 = call_qwen3_layer(o_layer41[0], o_layer41[1], 42)
                    o_layer43 = call_qwen3_layer(o_layer42[0], o_layer42[1], 43)
                    o_layer44 = call_qwen3_layer(o_layer43[0], o_layer43[1], 44)
                    o_layer45 = call_qwen3_layer(o_layer44[0], o_layer44[1], 45)
                    o_layer46 = call_qwen3_layer(o_layer45[0], o_layer45[1], 46)
                    o_layer47 = call_qwen3_layer(o_layer46[0], o_layer46[1], 47)
                    o_layer48 = call_qwen3_layer(o_layer47[0], o_layer47[1], 48)
                    o_layer49 = call_qwen3_layer(o_layer48[0], o_layer48[1], 49)
                    o_layer50 = call_qwen3_layer(o_layer49[0], o_layer49[1], 50)
                    o_layer51 = call_qwen3_layer(o_layer50[0], o_layer50[1], 51)
                    o_layer52 = call_qwen3_layer(o_layer51[0], o_layer51[1], 52)
                    o_layer53 = call_qwen3_layer(o_layer52[0], o_layer52[1], 53)
                    o_layer54 = call_qwen3_layer(o_layer53[0], o_layer53[1], 54)
                    o_layer55 = call_qwen3_layer(o_layer54[0], o_layer54[1], 55)
                    o_layer56 = call_qwen3_layer(o_layer55[0], o_layer55[1], 56)
                    o_layer57 = call_qwen3_layer(o_layer56[0], o_layer56[1], 57)
                    o_layer58 = call_qwen3_layer(o_layer57[0], o_layer57[1], 58)
                    o_layer59 = call_qwen3_layer(o_layer58[0], o_layer58[1], 59)
                    o_layer60 = call_qwen3_layer(o_layer59[0], o_layer59[1], 60)
                    o_layer61 = call_qwen3_layer(o_layer60[0], o_layer60[1], 61)
                    o_layer62 = call_qwen3_layer(o_layer61[0], o_layer61[1], 62)
                    o_layer63 = call_qwen3_layer(o_layer62[0], o_layer62[1], 63)

                    # permute_dims4 = R.permute_dims(lm_head_weight1, axes=None)
                    # lv4: R.Tensor((batch_size, mk.VOCAB_SIZE), dtype="float16") = R.matmul(rms_norm4, permute_dims4, out_dtype="void")  # noqa: E501
                    lv4 = R.call_tir(cls.hgemm, (o_layer63[0], lm_head_weight1), out_sinfo=R.Tensor((batch_size, mk.VOCAB_SIZE), dtype="float16"))  # noqa: E501
                    lv4_rs = R.reshape(lv4, (batch_size, 1, mk.VOCAB_SIZE))
                    astype = R.call_tir(cls.cast, (lv4_rs,), out_sinfo=R.Tensor((batch_size, 1, mk.VOCAB_SIZE), dtype="float32"))  # noqa: E501

                    profiler = (
                        o_layer0[2], o_layer1[2], o_layer2[2], o_layer3[2], o_layer4[2], o_layer5[2],  # noqa: E501
                        o_layer6[2], o_layer7[2], o_layer8[2], o_layer9[2], o_layer10[2], o_layer11[2],  # noqa: E501
                        o_layer12[2], o_layer13[2], o_layer14[2], o_layer15[2], o_layer16[2], o_layer17[2],  # noqa: E501
                        o_layer18[2], o_layer19[2], o_layer20[2], o_layer21[2], o_layer22[2], o_layer23[2],  # noqa: E501
                        o_layer24[2], o_layer25[2], o_layer26[2], o_layer27[2], o_layer28[2], o_layer29[2],  # noqa: E501
                        o_layer30[2], o_layer31[2], o_layer32[2], o_layer33[2], o_layer34[2], o_layer35[2],  # noqa: E501
                        o_layer36[2], o_layer37[2], o_layer38[2], o_layer39[2], o_layer40[2], o_layer41[2],  # noqa: E501
                        o_layer42[2], o_layer43[2], o_layer44[2], o_layer45[2], o_layer46[2], o_layer47[2],  # noqa: E501
                        o_layer48[2], o_layer49[2], o_layer50[2], o_layer51[2], o_layer52[2], o_layer53[2],  # noqa: E501
                        o_layer54[2], o_layer55[2], o_layer56[2], o_layer57[2], o_layer58[2], o_layer59[2],  # noqa: E501
                        o_layer60[2], o_layer61[2], o_layer62[2], o_layer63[2]
                    )

                    gv1 = astype, paged_kv_cache, profiler
                    R.output(gv1)
                return gv1


            @R.function(pure=False)
            def batch_decode(
                input_embeds: R.Tensor(("batch_size", 1, mk.HIDDEN_SIZE), dtype="float16"),
                paged_kv_cache: R.Object,
                # rope
                cos_sin_cache: R.Tensor(("max_seq_len", mk.HEAD_DIM), dtype="float32"),
                packed_params: R.Tuple(
                    R.Tensor((mk.VOCAB_SIZE, mk.HIDDEN_SIZE), dtype="float16"),
                    #
                    *([R.Tensor(((mk.NUM_ATTENTION_HEADS + mk.NUM_KEY_VALUE_HEADS * 2) * mk.HEAD_DIM, mk.HIDDEN_SIZE), dtype="float16"),  # noqa: E501
                    R.Tensor((mk.HIDDEN_SIZE, mk.NUM_ATTENTION_HEADS * mk.HEAD_DIM), dtype="float16"),  # noqa: E501
                    R.Tensor((mk.HEAD_DIM,), dtype="float16"),
                    R.Tensor((mk.HEAD_DIM,), dtype="float16"),
                    R.Tensor((2 * mk.INTERMEDIATE_SIZE, mk.HIDDEN_SIZE), dtype="float16"),
                    R.Tensor((mk.HIDDEN_SIZE, mk.INTERMEDIATE_SIZE), dtype="float16"),
                    R.Tensor((mk.HIDDEN_SIZE,), dtype="float16"),
                    R.Tensor((mk.HIDDEN_SIZE,), dtype="float16"),] * NUM_HIDDEN_LAYERS),
                    R.Tensor((mk.HIDDEN_SIZE,), dtype="float16"),
                    R.Tensor((mk.VOCAB_SIZE, mk.HIDDEN_SIZE), dtype="float16"),
                ),
            ):
                cls = Module
                model_output = cls.batch_decode_inner(
                    input_embeds,
                    paged_kv_cache,
                    cos_sin_cache,
                    packed_params,
                )
                if PROFILER_ON:
                    rank = R.call_packed("runtime.disco.worker_id", sinfo_args=[R.Shape()]) if TP_SIZE > 1 else R.prim_value(-1)  # noqa: E501
                    _ = R.call_packed(
                        "megakernel.export_trace",
                        model_output[2],
                        rank,
                        sinfo_args=[],
                    )
                else:
                    # there must be some dummy thing in else statement to avoid error
                    _ = R.prim_value(-1)

                res = model_output[0], model_output[1]
                return res

        # fmt: on
        return Module

    if scheduler == "static":
        return static_mod()
    elif scheduler == "dynamic":
        return dynamic_mod()
    else:
        raise NotImplementedError
