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

#ifndef TVM_RUNTIME_DISCO_NCCL_NCCL_H_
#define TVM_RUNTIME_DISCO_NCCL_NCCL_H_

#include <dlpack/dlpack.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/disco/builtin.h>
#include <tvm/runtime/disco/session.h>
#include <tvm/runtime/registry.h>

#include "../../../support/process_id.h"
#include "../utils.h"

/* `TVM_NCCL_RCCL_SWITCH` is set to 0 for NCCL, 1 for RCCL */
#ifndef TVM_NCCL_RCCL_SWITCH
#define TVM_NCCL_RCCL_SWITCH 0
#endif
#if TVM_NCCL_RCCL_SWITCH == 0
#include <nccl.h>

#include "../../cuda/cuda_common.h"
#else
#include <rccl/rccl.h>

#include "../../rocm/rocm_common.h"
#endif


namespace tvm {
namespace runtime {
namespace nccl {

#define NCCL_CALL(cmd)                                                      \
  do {                                                                      \
    auto r = (cmd);                                                         \
    if (r != ncclSuccess) {                                                 \
      LOG(FATAL) << TVM_DISCO_CCL_NAME "Errror: " << ncclGetErrorString(r); \
    }                                                                       \
  } while (0)

#if TVM_NCCL_RCCL_SWITCH == 0

#define TVM_DISCO_DEVICE_NAME "cuda"
#define TVM_DISCO_CCL_NAME "nccl"

using deviceStream_t = cudaStream_t;
const constexpr DLDeviceType TVM_DISCO_DEVICE_TYPE = DLDeviceType::kDLCUDA;
inline void SetDevice(int device_id) { CUDA_CALL(cudaSetDevice(device_id)); }
inline void StreamSynchronize(deviceStream_t stream) { CUDA_CALL(cudaStreamSynchronize(stream)); }
inline void StreamCreate(deviceStream_t* stream) { CUDA_CALL(cudaStreamCreate(stream)); }
inline void StreamDestroy(deviceStream_t stream) { CUDA_CALL(cudaStreamDestroy(stream)); }

#else

#define TVM_DISCO_DEVICE_NAME "rocm"
#define TVM_DISCO_CCL_NAME "rccl"

using deviceStream_t = hipStream_t;
const constexpr DLDeviceType TVM_DISCO_DEVICE_TYPE = DLDeviceType::kDLROCM;
inline void SetDevice(int device_id) { ROCM_CALL(hipSetDevice(device_id)); }
inline void StreamSynchronize(deviceStream_t stream) { ROCM_CALL(hipStreamSynchronize(stream)); }
inline void StreamCreate(deviceStream_t* stream) { ROCM_CALL(hipStreamCreate(stream)); }
inline void StreamDestroy(deviceStream_t stream) { ROCM_CALL(hipStreamDestroy(stream)); }

#endif

struct CCLThreadLocalContext {
  DiscoWorker* worker;
  int device_id;
  deviceStream_t default_stream = nullptr;
  ncclComm_t comm;

  void Clear() {
    NCCL_CALL(ncclCommDestroy(comm));
    if (default_stream != nullptr) {
      StreamDestroy(default_stream);
    }
  }

  deviceStream_t GetDefaultStream() {
    const auto* func = tvm::runtime::Registry::Get("runtime.get_" TVM_DISCO_DEVICE_NAME "_stream");
    ICHECK(func != nullptr);
    deviceStream_t stream = static_cast<deviceStream_t>((*func)().operator void*());
    return stream == nullptr ? default_stream : stream;
  }

  static CCLThreadLocalContext* Get();
};

}  // namespace nccl
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_DISCO_NCCL_NCCL_H_