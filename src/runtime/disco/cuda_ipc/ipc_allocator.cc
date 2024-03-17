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

#include "ipc_allocator.h"

#include <tvm/runtime/container/shape_tuple.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/memory/memory_manager.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/relax_vm/builtin.h>
#include <tvm/runtime/relax_vm/bytecode.h>
#include <tvm/runtime/relax_vm/vm.h>

namespace tvm {
namespace runtime {
namespace cuda_ipc {

using namespace tvm::runtime::relax_vm;

CUDAIPCMemoryAllocator* ipc_alloc;

Storage IPCAllocStorage(ShapeTuple buffer_shape, DLDataType dtype_hint) {
  auto storage_obj = runtime::SimpleObjAllocator().make_object<StorageObj>();
  storage_obj->buffer = ipc_alloc->Alloc(buffer_shape, dtype_hint, "");
  Storage storage(storage_obj);
  return storage;
}

void InitIPCAllocator() {
  nccl::CCLThreadLocalContext* ctx = nccl::CCLThreadLocalContext::Get();
  ipc_alloc = new CUDAIPCMemoryAllocator(DLDevice{kDLCUDA, ctx->device_id});
}


TVM_REGISTER_GLOBAL("cuda_ipc.alloc_storage").set_body_typed(IPCAllocStorage);
TVM_REGISTER_GLOBAL("cuda_ipc.init_ipc_allocator").set_body_typed(InitIPCAllocator);

}  // namespace cuda_ipc
}  // namespace runtime
}  // namespace tvm