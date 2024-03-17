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

#ifndef TVM_RUNTIME_DISCO_CUDA_IPC_IPC_ALLOCATOR_H_
#define TVM_RUNTIME_DISCO_CUDA_IPC_IPC_ALLOCATOR_H_

#include <tvm/runtime/device_api.h>
#include <tvm/runtime/memory/memory_manager.h>

#include "../../memory/pooled_allocator.h"
#include "../nccl/nccl_context.h"
#include "custom_allreduce.h"
#include "ipc_memory.h"
namespace tvm {
namespace runtime {
namespace cuda_ipc {



using tvm::runtime::memory::Buffer;

class CUDAIPCMemoryAllocator final : public memory::PooledAllocator {
 public:
  explicit CUDAIPCMemoryAllocator(Device dev, size_t page_size = kDefaultPageSize)
      : PooledAllocator(dev, page_size) {
    ICHECK(dev.device_type == kDLCUDA);
    CUDA_CALL(cudaStreamCreateWithFlags(&cpu_comm_stream_, cudaStreamNonBlocking));
  }

  Buffer Alloc(ShapeTuple shape, DLDataType type_hint, const std::string& mem_scope) override {
    if (mem_scope.empty() || mem_scope == "cuda_ipc") {
      NDArray::Container container(nullptr, shape, type_hint, device_);
      size_t size = DeviceAPI::Get(device_)->GetDataSize(container.dl_tensor);
      size_t alignment = GetDataAlignment(container.dl_tensor);
      return memory::PooledAllocator::Alloc(size, alignment, type_hint);
    }
    LOG(FATAL) << "This alloc should be implemented";
    return {};
  }

  IPCMemoryContext GetIPCMemoryContext(void* ptr) {
    ICHECK(ipc_remote_mem.count(ptr));
    return ipc_remote_mem.at(ptr);
  }

 private:
  inline size_t GetDataAlignment(const DLTensor& arr) {
    size_t align = (arr.dtype.bits / 8) * arr.dtype.lanes;
    if (align < kAllocAlignment) return kAllocAlignment;
    return align;
  }
  void* DeviceAllocDataSpace(Device dev, size_t size, size_t alignment,
                             DLDataType type_hint) override {
    auto [data_ptr, dataCommPtrs] = AllocIPCMemory(dev, size, alignment, type_hint);
    auto [barrier_in_ptr, barrier_in_commPtrs] =
        AllocIPCMemory(dev, sizeof(uint32_t) * MAX_RANKS_PER_NODE, alignment, DataType::UInt(32));
    auto [barrier_out_ptr, barrier_out_commPtrs] =
        AllocIPCMemory(dev, sizeof(uint32_t) * MAX_RANKS_PER_NODE, alignment, DataType::UInt(32));
    ipc_remote_mem[data_ptr] =
        IPCMemoryContext{dataCommPtrs, barrier_in_commPtrs, barrier_out_commPtrs, 0};
    return data_ptr;
  }

  std::pair<void*, std::vector<void*>> AllocIPCMemory(Device dev, size_t size, size_t alignment,
                                                      DLDataType type_hint) {
    // alloc local buffer
    ICHECK(dev.device_type == kDLCUDA);
    void* ptr;
    CUDA_CALL(cudaSetDevice(dev.device_id));
    CUDA_CALL(cudaMalloc(&ptr, size));
    // create ipc handle
    cudaIpcMemHandle_t localHandle;
    CUDA_CALL(cudaIpcGetMemHandle(&localHandle, ptr));
    // all gather ipc handle
    nccl::CCLThreadLocalContext* ctx = nccl::CCLThreadLocalContext::Get();
    void *d_src, *d_dst;
    CUDA_CALL(cudaMalloc(&d_src, CUDA_IPC_HANDLE_SIZE));
    CUDA_CALL(cudaMalloc(&d_dst, CUDA_IPC_HANDLE_SIZE * ctx->worker->num_workers));
    CUDA_CALL(cudaMemcpyAsync(d_src, &localHandle, CUDA_IPC_HANDLE_SIZE, cudaMemcpyDefault,
                              cpu_comm_stream_));
    NCCL_CALL(
        ncclAllGather(d_src, d_dst, CUDA_IPC_HANDLE_SIZE, ncclChar, ctx->comm, cpu_comm_stream_));
    CUDA_CALL(cudaStreamSynchronize(cpu_comm_stream_));
    std::vector<char> serialHandles(CUDA_IPC_HANDLE_SIZE * ctx->worker->num_workers, 0);
    CUDA_CALL(cudaMemcpy(serialHandles.data(), d_dst,
                         CUDA_IPC_HANDLE_SIZE * ctx->worker->num_workers, cudaMemcpyDefault));
    std::vector<cudaIpcMemHandle_t> handles(ctx->worker->num_workers);
    for (int i = 0; i < ctx->worker->num_workers; i++) {
      memcpy(handles[i].reserved, &serialHandles[i * CUDA_IPC_HANDLE_SIZE], CUDA_IPC_HANDLE_SIZE);
    }
    std::vector<void*> mCommPtrs(ctx->worker->num_workers);
    for (size_t nodeId = 0; nodeId < handles.size(); nodeId++) {
      if ((int)nodeId == ctx->worker->worker_id) {
        mCommPtrs[nodeId] = ptr;
      } else {
        uint8_t* foreignBuffer;
        CUDA_CALL(cudaIpcOpenMemHandle(reinterpret_cast<void**>(&foreignBuffer), handles[nodeId],
                                       cudaIpcMemLazyEnablePeerAccess));
        mCommPtrs[nodeId] = foreignBuffer;
      }
    }
    CUDA_CALL(cudaFree(d_src));
    CUDA_CALL(cudaFree(d_dst));
    return std::make_pair(ptr, mCommPtrs);
  }

  void DeviceFreeDataSpace(Device dev, void* ptr) override {
    ICHECK(dev.device_type == kDLCUDA);
    CUDA_CALL(cudaSetDevice(dev.device_id));
    nccl::CCLThreadLocalContext* ctx = nccl::CCLThreadLocalContext::Get();
    // free local buffer
    CUDA_CALL(cudaFree(ptr));
    // free ipc handle
    for (int i = 0; i < ctx->worker->num_workers; i++) {
      if(i != ctx->worker->worker_id){
        CUDA_CALL(cudaIpcCloseMemHandle(ipc_remote_mem[ptr].remote_data[i]));
        CUDA_CALL(cudaIpcCloseMemHandle(ipc_remote_mem[ptr].barrier_in[i]));
        CUDA_CALL(cudaIpcCloseMemHandle(ipc_remote_mem[ptr].barrier_out[i]));
      }
    }
    ipc_remote_mem.erase(ptr);
  }

  private:
   cudaStream_t cpu_comm_stream_;
   std::unordered_map<void*, IPCMemoryContext> ipc_remote_mem;
};

extern CUDAIPCMemoryAllocator* ipc_alloc;

}  // namespace cuda_ipc
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_DISCO_CUDA_IPC_IPC_ALLOCATOR_H_