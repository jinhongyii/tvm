/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/tensor.h>

#include <../../../../3rdparty/flashinfer/include/flashinfer/attention/decode.cuh>
#include <../../../../3rdparty/flashinfer/include/flashinfer/attention/variants.cuh>
#include <../../../../3rdparty/flashinfer/include/flashinfer/page.cuh>
#include <../../../../3rdparty/flashinfer/include/flashinfer/sampling.cuh>
#include <optional>

#include "../../../../3rdparty/flashinfer/include/flashinfer/attention/mask.cuh"
#include "../../../../3rdparty/flashinfer/include/flashinfer/attention/scheduler.cuh"
#include "../../../../3rdparty/flashinfer/include/flashinfer/pos_enc.cuh"

using IdType = int32_t;
using tvm::ffi::Any;
using tvm::ffi::Array;
using tvm::ffi::Shape;
using tvm::runtime::DataType;
using tvm::runtime::Tensor;

namespace tvm {
namespace runtime {

using namespace flashinfer;

using DTypeQ = half;
using DTypeKV = half;
using DTypeO = half;

struct BatchDecodeParams {
  using DTypeQ = DTypeQ;
  using DTypeKV = half;
  using DTypeO = half;
  using IdType = IdType;

  DTypeQ* q;
  paged_kv_t<DTypeKV, IdType> paged_kv;
  DTypeO* o;
  float* lse;

  double sm_scale;
  double rope_rcp_scale;
  double rope_rcp_theta;

  IdType* decode_maybe_q_rope_offset;

  uint32_t padded_batch_size;
  uint32_t num_qo_heads;
  IdType q_stride_n;
  IdType q_stride_h;
  int32_t window_left;

  IdType* request_indices;
  IdType* kv_tile_indices;
  IdType* o_indptr;
  IdType* kv_chunk_size_ptr;
  bool* block_valid_mask;
  bool partition_kv;

  __host__ __device__ __forceinline__ int32_t get_qo_len(int32_t batch_idx) const { return 1; }

  __host__ __device__ __forceinline__ int32_t get_kv_len(int32_t batch_idx) const {
    return paged_kv.get_length(batch_idx);
  }
};

ffi::Shape BatchPrefillWithKVCachePlan(
    DLTensor* float_workspace_buffer, DLTensor* int_workspace_buffer,
    DLTensor* page_locked_int_workspace_buffer, DLTensor* qo_indptr, DLTensor* kv_indptr,
    ffi::Shape kv_len_arr, int64_t total_num_rows, int64_t batch_size, int64_t num_qo_heads,
    int64_t num_kv_heads, int64_t page_size, bool enable_cuda_graph, int64_t head_dim_qk,
    int64_t head_dim_vo, bool causal, TVMStreamHandle cuda_stream) {
  size_t float_workspace_size_in_bytes =
      float_workspace_buffer->shape[0] * DataType(float_workspace_buffer->dtype).bytes();
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer->shape[0] * DataType(int_workspace_buffer->dtype).bytes();

  PrefillPlanInfo plan_info;

  cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);
  cudaError_t status = PrefillPlan<IdType>(
      static_cast<char*>(float_workspace_buffer->data) + float_workspace_buffer->byte_offset,
      float_workspace_size_in_bytes,
      static_cast<char*>(int_workspace_buffer->data) + int_workspace_buffer->byte_offset,
      static_cast<char*>(page_locked_int_workspace_buffer->data) +
          page_locked_int_workspace_buffer->byte_offset,
      int_workspace_size_in_bytes, plan_info,
      static_cast<IdType*>(qo_indptr->data) + qo_indptr->byte_offset / sizeof(IdType),
      static_cast<IdType*>(kv_indptr->data) + kv_indptr->byte_offset / sizeof(IdType),
      total_num_rows, batch_size, num_qo_heads, num_kv_heads, head_dim_qk, head_dim_vo, page_size,
      enable_cuda_graph,
      /*sizeof_dtype_o=*/2, stream);

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "Failed to plan prefill with error: " << cudaGetErrorString(status);

  std::vector<int64_t> plan_info_vec = plan_info.ToVector();
  return ffi::Shape{plan_info_vec.begin(), plan_info_vec.end()};
}

Array<Any> BatchDecodeWithPagedKVCachePlan(
    Tensor float_workspace_buffer, Tensor int_workspace_buffer,
    Tensor page_locked_int_workspace_buffer, Tensor indptr, int64_t batch_size,
    int64_t num_qo_heads, int64_t num_kv_heads, int64_t page_size, bool enable_cuda_graph,
    int64_t pos_encoding_mode_code, int64_t window_left, int64_t head_dim_qk, int64_t head_dim_vo,
    DataType q_scalar_type, DataType kv_scalar_type) {
  size_t float_workspace_size_in_bytes =
      float_workspace_buffer->shape[0] * DataType(float_workspace_buffer->dtype).bytes();
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer->shape[0] * DataType(int_workspace_buffer->dtype).bytes();

  DecodePlanInfo plan_info;

  TVM_FFI_ICHECK_EQ(head_dim_qk, head_dim_vo)
      << "CUDA cores template only supports equal head dim for QK and VO, please use tensor "
         "cores template for different head dim";

  const PosEncodingMode pos_encoding_mode = static_cast<PosEncodingMode>(pos_encoding_mode_code);

  Device device = float_workspace_buffer->device;
  const cudaStream_t stream =
      static_cast<cudaStream_t>(DeviceAPI::Get(device)->GetCurrentStream(device));
  constexpr int GROUP_SIZE = 8;
  constexpr int HEAD_DIM_QK = 128;
  constexpr PosEncodingMode POS_ENCODING_MODE = PosEncodingMode::kNone;
  using AttentionVariant = DefaultAttention<false, false, false, false>;
  using Params = BatchDecodeParams;

  auto work_estimation_func = BatchDecodeWithPagedKVCacheWorkEstimationDispatched<
      GROUP_SIZE, HEAD_DIM_QK, POS_ENCODING_MODE, AttentionVariant, Params>;
  cudaError_t status = DecodePlan<HEAD_DIM_QK, POS_ENCODING_MODE, AttentionVariant, Params>(
      static_cast<char*>(float_workspace_buffer->data) + float_workspace_buffer->byte_offset,
      float_workspace_size_in_bytes,
      static_cast<char*>(int_workspace_buffer->data) + int_workspace_buffer->byte_offset,
      static_cast<char*>(page_locked_int_workspace_buffer->data) +
          page_locked_int_workspace_buffer->byte_offset,
      int_workspace_size_in_bytes, plan_info,
      static_cast<IdType*>(indptr->data) + indptr->byte_offset / sizeof(IdType), batch_size,
      num_qo_heads, page_size, enable_cuda_graph,
      /*stream=*/stream, work_estimation_func);

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "BatchDecodeWithPagedKVCache failed with error " << cudaGetErrorString(status);

  std::vector<int64_t> plan_info_vec = plan_info.ToVector();
  int64_t padded_batch_size = plan_info_vec[0];
  int64_t split_kv = plan_info_vec[9];
  int64_t request_indices_offset = plan_info_vec[3];
  int64_t request_indices_size = padded_batch_size;
  int64_t kv_tile_indices_offset = plan_info_vec[4];
  int64_t kv_tile_indices_size = padded_batch_size;
  int64_t kv_chunk_size_ptr_offset = plan_info_vec[7];
  int64_t kv_chunk_size_ptr_size = 1;
  int64_t o_inptr_offset = plan_info_vec[5];
  int64_t o_indptr_size = batch_size + 1;

  DataType dtype = DataType::Int(32);
  Tensor request_indices =
      int_workspace_buffer.CreateView({request_indices_size}, dtype,
                                      /*relative_byte_offset=*/request_indices_offset);
  Tensor kv_tile_indices =
      int_workspace_buffer.CreateView({kv_tile_indices_size}, dtype,
                                      /*relative_byte_offset=*/kv_tile_indices_offset);
  Tensor kv_chunk_size_ptr =
      int_workspace_buffer.CreateView({kv_chunk_size_ptr_size}, dtype,
                                      /*relative_byte_offset=*/kv_chunk_size_ptr_offset);
  Tensor o_indptr_device = int_workspace_buffer.CreateView({o_indptr_size}, dtype,
                                                           /*relative_byte_offset=*/o_inptr_offset);
  Tensor o_indptr_host =
      page_locked_int_workspace_buffer.CreateView({o_indptr_size}, dtype,
                                                  /*relative_byte_offset=*/o_inptr_offset);
  return {request_indices, kv_tile_indices, kv_chunk_size_ptr,
          o_indptr_device, o_indptr_host,   split_kv};
}

Array<Any> BatchPagedAttentionPlan(Tensor float_workspace_buffer, Tensor int_workspace_buffer,
                                   Tensor page_locked_int_workspace_buffer, Tensor qo_indptr,
                                   Tensor kv_indptr, Tensor kv_len, int64_t batch_size,
                                   int64_t num_qo_heads, int64_t num_kv_heads, int64_t head_dim_o,
                                   bool causal) {
  size_t float_workspace_size_in_bytes =
      float_workspace_buffer->shape[0] * DataType(float_workspace_buffer->dtype).bytes();
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer->shape[0] * DataType(int_workspace_buffer->dtype).bytes();

  HolisticPlanInfo<2> plan_info;
  Device device = float_workspace_buffer->device;
  const cudaStream_t stream =
      static_cast<cudaStream_t>(DeviceAPI::Get(device)->GetCurrentStream(device));

  constexpr uint32_t NUM_TASKS = 2;
  const uint32_t CTA_TILE_Q_SIZES[NUM_TASKS] = {128, 16};
  cudaError_t status = TwoStageHolisticPlan<IdType>(
      static_cast<char*>(float_workspace_buffer->data) + float_workspace_buffer->byte_offset,
      float_workspace_size_in_bytes,
      static_cast<char*>(int_workspace_buffer->data) + int_workspace_buffer->byte_offset,
      static_cast<char*>(page_locked_int_workspace_buffer->data) +
          page_locked_int_workspace_buffer->byte_offset,
      int_workspace_size_in_bytes, plan_info,
      static_cast<IdType*>(qo_indptr->data) + qo_indptr->byte_offset / sizeof(IdType),
      static_cast<IdType*>(kv_indptr->data) + kv_indptr->byte_offset / sizeof(IdType),
      static_cast<IdType*>(kv_len->data) + kv_len->byte_offset / sizeof(IdType), batch_size,
      num_qo_heads, num_kv_heads, head_dim_o, causal, stream);

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "BatchPagedAttentionPlan failed with error " << cudaGetErrorString(status);

  constexpr uint32_t NUM_TASK_ARGS = 10;
  constexpr uint32_t NUM_SHARED_ARGS = 8;
  const int max_total_num_works = 1025;
  int num_clusters = 148;
  const int max_num_kv_splits = 4 * num_clusters * 2 * (CTA_TILE_Q_SIZES[1]);
  DataType int_dtype = DataType::Int(32);
  DataType fp16_dtype = DataType::Float(16);
  DataType fp32_dtype = DataType::Float(32);

  std::vector<int64_t> plan_info_vec = plan_info.ToVector();
  TVM_FFI_ICHECK_EQ(plan_info_vec.size(), NUM_SHARED_ARGS + NUM_TASKS * NUM_TASK_ARGS);

  std::vector<Any> ret;
  int64_t num_blks_x = plan_info_vec[0];
  int64_t num_blks_y = plan_info_vec[1];
  ret.push_back(num_blks_x);
  ret.push_back(num_blks_y);

  for (uint32_t i = 0; i < NUM_TASKS; ++i) {
    std::vector<Tensor> task_arrays;
    task_arrays.reserve(NUM_TASK_ARGS);
    int64_t q_indptr_offset = plan_info_vec[2 + i * NUM_TASK_ARGS + 0];
    Tensor q_indptr = int_workspace_buffer.CreateView({max_total_num_works}, int_dtype,
                                                      /*relative_byte_offset=*/q_indptr_offset);
    task_arrays.push_back(q_indptr);
    int64_t kv_indptr_offset = plan_info_vec[2 + i * NUM_TASK_ARGS + 1];
    Tensor kv_indptr = int_workspace_buffer.CreateView({max_total_num_works}, int_dtype,
                                                       /*relative_byte_offset=*/kv_indptr_offset);
    task_arrays.push_back(kv_indptr);
    int64_t partial_indptr_offset = plan_info_vec[2 + i * NUM_TASK_ARGS + 2];
    Tensor partial_indptr =
        int_workspace_buffer.CreateView({max_total_num_works}, int_dtype,
                                        /*relative_byte_offset=*/partial_indptr_offset);
    task_arrays.push_back(partial_indptr);
    int64_t q_len_offset = plan_info_vec[2 + i * NUM_TASK_ARGS + 3];
    Tensor q_len = int_workspace_buffer.CreateView({max_total_num_works}, int_dtype,
                                                   /*relative_byte_offset=*/q_len_offset);
    task_arrays.push_back(q_len);
    int64_t kv_len_offset = plan_info_vec[2 + i * NUM_TASK_ARGS + 4];
    Tensor kv_len = int_workspace_buffer.CreateView({max_total_num_works}, int_dtype,
                                                    /*relative_byte_offset=*/kv_len_offset);
    task_arrays.push_back(kv_len);
    int64_t q_start_offset = plan_info_vec[2 + i * NUM_TASK_ARGS + 5];
    Tensor q_start = int_workspace_buffer.CreateView({max_total_num_works}, int_dtype,
                                                     /*relative_byte_offset=*/q_start_offset);
    task_arrays.push_back(q_start);
    int64_t kv_start_offset = plan_info_vec[2 + i * NUM_TASK_ARGS + 6];
    Tensor kv_start = int_workspace_buffer.CreateView({max_total_num_works}, int_dtype,
                                                      /*relative_byte_offset=*/kv_start_offset);
    task_arrays.push_back(kv_start);
    int64_t kv_end_offset = plan_info_vec[2 + i * NUM_TASK_ARGS + 7];
    Tensor kv_end = int_workspace_buffer.CreateView({max_total_num_works}, int_dtype,
                                                    /*relative_byte_offset=*/kv_end_offset);
    task_arrays.push_back(kv_end);
    int64_t kv_head_idx_offset = plan_info_vec[2 + i * NUM_TASK_ARGS + 8];
    Tensor kv_head_idx =
        int_workspace_buffer.CreateView({max_total_num_works}, int_dtype,
                                        /*relative_byte_offset=*/kv_head_idx_offset);
    task_arrays.push_back(kv_head_idx);
    int64_t work_indptr_offset = plan_info_vec[2 + i * NUM_TASK_ARGS + 9];
    Tensor work_indptr =
        int_workspace_buffer.CreateView({max_total_num_works}, int_dtype,
                                        /*relative_byte_offset=*/work_indptr_offset);
    task_arrays.push_back(work_indptr);
    // on the host meomory, needed by event generation of megakernel
    Tensor q_indptr_host =
        page_locked_int_workspace_buffer.CreateView({max_total_num_works}, int_dtype,
                                                    /*relative_byte_offset=*/q_indptr_offset);
    task_arrays.push_back(q_indptr_host);
    Tensor kv_head_idx_host =
        page_locked_int_workspace_buffer.CreateView({max_total_num_works}, int_dtype,
                                                    /*relative_byte_offset=*/kv_head_idx_offset);
    task_arrays.push_back(kv_head_idx_host);
    Tensor work_indptr_host =
        page_locked_int_workspace_buffer.CreateView({max_total_num_works}, int_dtype,
                                                    /*relative_byte_offset=*/work_indptr_offset);
    task_arrays.push_back(work_indptr_host);
    ret.push_back(Array<Tensor>(task_arrays));
  }

  int64_t len_kv_chunk_offset = plan_info_vec[2 + NUM_TASKS * NUM_TASK_ARGS];
  Tensor len_kv_chunk =
      int_workspace_buffer.CreateView({NUM_TASKS}, int_dtype,
                                      /*relative_byte_offset=*/len_kv_chunk_offset);
  ret.push_back(len_kv_chunk);
  int64_t merge_indptr_offset = plan_info_vec[5 + NUM_TASKS * NUM_TASK_ARGS];
  Tensor merge_indptr =
      int_workspace_buffer.CreateView({max_num_kv_splits}, int_dtype,
                                      /*relative_byte_offset=*/merge_indptr_offset);
  ret.push_back(merge_indptr);
  int64_t merge_o_indices_offset = plan_info_vec[6 + NUM_TASKS * NUM_TASK_ARGS];
  Tensor merge_o_indices =
      int_workspace_buffer.CreateView({max_num_kv_splits}, int_dtype,
                                      /*relative_byte_offset=*/merge_o_indices_offset);
  ret.push_back(merge_o_indices);
  int64_t num_qo_len_offset = plan_info_vec[7 + NUM_TASKS * NUM_TASK_ARGS];
  Tensor num_qo_len = int_workspace_buffer.CreateView({1}, int_dtype,
                                                      /*relative_byte_offset=*/num_qo_len_offset);
  ret.push_back(num_qo_len);
  return ret;
}

void TopPSamplingFromProb(DLTensor* probs, DLTensor* output, DLTensor* top_p_arr,
                          uint64_t philox_seed, uint64_t philox_offset, uint64_t cuda_stream) {
  bool deterministic = true;
  TVM_FFI_ICHECK_EQ(probs->ndim, 2) << "Probs should have 2 dimensions";
  TVM_FFI_ICHECK_EQ(output->ndim, 1) << "Output should have 1 dimension";
  TVM_FFI_ICHECK_EQ(top_p_arr->ndim, 1) << "TopPArr should have 1 dimension";
  int64_t batch_size = output->shape[0];
  int64_t vocab_size = probs->shape[1];

  cudaError_t status = sampling::TopPSamplingFromProb<float, int>(
      static_cast<float*>(probs->data) + probs->byte_offset / sizeof(float),
      static_cast<int*>(output->data) + output->byte_offset / sizeof(int),
      /*indices=*/nullptr,
      static_cast<float*>(top_p_arr->data) + top_p_arr->byte_offset / sizeof(float), batch_size,
      /*top_p_val=*/0, vocab_size, deterministic, philox_seed, philox_offset,
      reinterpret_cast<cudaStream_t>(cuda_stream));
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "SamplingFromProbs failed with error " << cudaGetErrorString(status);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("flashinfer.batch_prefill_with_kv_cache_plan", BatchPrefillWithKVCachePlan);
  refl::GlobalDef().def("flashinfer.batch_decode_with_paged_kv_cache_plan",
                        BatchDecodeWithPagedKVCachePlan);
  refl::GlobalDef().def("flashinfer.batch_paged_attention_plan", BatchPagedAttentionPlan);
  refl::GlobalDef().def("flashinfer.top_p_sampling_from_prob", TopPSamplingFromProb);
}

}  // namespace runtime
}  // namespace tvm
