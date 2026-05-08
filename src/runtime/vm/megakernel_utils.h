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
/*!
 * \file src/runtime/vm/attn_utils.h
 * \brief Data structure and utilities for KV cache.
 */

#ifndef TVM_RUNTIME_VM_MEGAKERNEL_UTILS_H_
#define TVM_RUNTIME_VM_MEGAKERNEL_UTILS_H_

#include <tvm/runtime/tensor.h>

#include <string>
#include <unordered_map>

namespace tvm {
namespace runtime {
namespace vm {

class HostMemoryVector;

inline int ceildiv(int a, int b) { return (a + b - 1) / b; }

namespace megakernel {
// Kernel configurations
constexpr const int kNumSM = 148;
constexpr const int kNumWarpgroupPerBlock = 2;
constexpr const int kNumWarpPerWarpgroup = 4;
// Batch attention tasks config
constexpr const int kSplitKReduceTileNUnit = 128;
constexpr const int kSplitKReduceTileNRepeat = 1;
constexpr const int kGemmTileBlkN = 128;
constexpr const int kGemmTileBlkK = 64;
constexpr const int kSiluMultiplyTileSize = 128;
constexpr const int kAllReduceTileMTile = 16;
constexpr const int kAllReduceTileNTile = 128;
constexpr const int kMaxTotalNumWorks = 1025;
// MOE tasks config
constexpr const int kMoeBlkM = 128;
constexpr const int kGatingBlkM = 128;
constexpr const int kGroupGemmBlkN = 128;
constexpr const int kSiluMultiplyMoeTileSize = 768;
// Scheduler and semaphore config
constexpr const int kStaticTileSchedulerMaxTasks = 128;
constexpr const int kDyanmicTileSchedulerMaxTasks = 32768;
constexpr const int kSemaphoreBase = (1 << 16);
constexpr const int kSemaphoreFactor = (1 << 16) + 1;
constexpr const int kMaxSemaphore = 2147483647;
constexpr const int kEtensorWorkspaceSize = 1024 * 1024;
constexpr const int kMaxNumEtensors = 20;
// supported model names
const std::unordered_map<int, std::string> kModelNames = {
    {0, "qwen3_32b"}, {1, "qwen3_30b_a3b"}, {2, "qwen3_30b_a3b_unfused"}, {3, "llama3_1b"}};

enum class JobType : int32_t {
  kVReduceAppend = 0,
  kKReduceNormRopeAppend = 1,
  kQReduceNormRope = 2,
  kBatchAttention = 3,
  kBatchMerge = 4,
  kGateUpProjReduce = 5,
  kDownProjAllReduce = 6,
  kOAllReduce = 7,
  kAttnAddRMSNorm = 8,
  kGemmOReduce = 9,
  kGemmOProj = 10,
  kGemmQKVProj = 11,
  kMLPAddRMSNorm = 12,
  kDownProjReduce = 13,
  kGemmDownProj = 14,
  kSplitSiluMultiply = 15,
  kGemmGateUpProj = 16,
  kGateUpSilu = 17,
  kMoeGating = 18,
  kMoeTopkSoftmax = 19,
  kMoeAlign = 20,
  kMoeCountAndSort = 21,
  kMoeGroupGemmGateUp = 22,
  kMoeSiluMultiply = 23,
  kMoeGroupGemmDown = 24,
  kMoeTopkReduce = 25,
  kMoeGroupGemmGateUpSilu = 26,
  kInitEtensor = 27,
  kWaitEtensorInit = 28,
  // end
  kEnd = 31,
};

// every task info in exec queue will be squashed into 32bit:
// task_type: [0:5], m_idx: [5:18], n_idx: [18:28], k_idx: [28:32]
constexpr const int kMaxTaskType = 1 << 5;
constexpr const int kMaxMIdx = 1 << 13;
constexpr const int kMaxNIdx = 1 << 10;
constexpr const int kMaxKIdx = 1 << 4;

inline int32_t PackInto32bit(int32_t task_type, int32_t m_idx, int32_t n_idx, int32_t k_idx) {
  TVM_FFI_ICHECK_LT(task_type, kMaxTaskType);
  TVM_FFI_ICHECK_LT(m_idx, kMaxMIdx);
  TVM_FFI_ICHECK_LT(n_idx, kMaxNIdx);
  TVM_FFI_ICHECK_LT(k_idx, kMaxKIdx);
  return (task_type | (m_idx << 5) | (n_idx << 18) | (k_idx << 28)) & 0xFFFFFFFF;
}

ffi::Array<Tensor> GetEventTensorsOnLayer(ffi::Array<Tensor> etensors, int layer_id);

Tensor GenerateExecQueueStatic(int batch_size, int attn_task_num, int tp_size,
                               std::string model_name, Device device, Device preferred_host_device);

ffi::Array<ffi::Array<Tensor>> GenerateExecQueueDynamic(Tensor exec_queue_device_buf,
                                                        Tensor exec_queue_host_buf, int tp_size,
                                                        std::string model_name, int num_layers,
                                                        TVMStreamHandle copy_stream);

Tensor GetExecQueueStatic(tvm::ffi::AnyView vm_arg, ffi::ObjectRef gen_exec_func,
                          ffi::Shape cache_args);

}  // namespace megakernel
}  // namespace vm
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_VM_MEGAKERNEL_UTILS_H_
