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

#include "megakernel_utils.h"

#include <tvm/ffi/cast.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/cast.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/nvtx.h>
#include <tvm/runtime/tensor.h>
#include <tvm/runtime/vm/vm.h>

#include <utility>

#include "attn_utils.h"

namespace tvm {
namespace runtime {
namespace vm {
namespace megakernel {

using ffi::Array;
using ffi::Shape;
using runtime::Tensor;

Array<Tensor> GetEventTensorsOnLayer(Array<Tensor> etensors, int layer_id) {
  TVM_FFI_ICHECK_GE(layer_id, 0) << "Layer id must be non-negative, but got " << layer_id;
  std::vector<Tensor> etensors_on_layer;
  etensors_on_layer.reserve(etensors.size());
  for (int i = 0; i < static_cast<int>(etensors.size()); i++) {
    Shape shape = etensors[i].shape();
    TVM_FFI_ICHECK_GE(shape.size(), 2) << "Event tensor must have at least 2 dimensions";
    TVM_FFI_ICHECK_LT(layer_id, shape[0])
        << "Layer id must be less than the first dimension of the event tensor, but got "
        << layer_id << " and " << shape[0];
    std::vector<int64_t> new_shape_vec{shape.begin() + 1, shape.end()};
    Shape new_shape(new_shape_vec);
    int64_t offset = layer_id * new_shape.Product() * DataType(etensors[i].dtype()).bytes();
    etensors_on_layer.push_back(etensors[i].CreateView(new_shape, etensors[i].dtype(), offset));
  }
  return Array<Tensor>(etensors_on_layer);
}

Tensor GenerateExecQueueStatic(int batch_size, int attn_task_num, int tp_size,
                               std::string model_name, Device device,
                               Device preferred_host_device) {
  NVTXScopedRange range("Generate execution queue");
  const auto f_get_config =
      tvm::ffi::Function::GetGlobalRequired("tirx.megakernel.get_model_config");
  auto config = f_get_config(model_name).cast<ffi::Map<ffi::String, ffi::Any>>();
  int num_qo_heads = config["NUM_ATTENTION_HEADS"].cast<int>() / tp_size;
  int num_kv_heads = config["NUM_KEY_VALUE_HEADS"].cast<int>() / tp_size;
  int head_dim = config["HEAD_DIM"].cast<int>();
  int hidden_size = config["HIDDEN_SIZE"].cast<int>();
  int intermediate_size = config["INTERMEDIATE_SIZE"].cast<int>() / tp_size;
  int split_qkv_project = config["SPLIT_QKV_PROJECT_DICT"].cast<ffi::Map<int, int>>()[tp_size];
  int split_o_project = config["SPLIT_O_PROJECT_DICT"].cast<ffi::Map<int, int>>()[tp_size];
  TVM_FFI_ICHECK_NE(split_qkv_project, -1);
  TVM_FFI_ICHECK_NE(split_o_project, -1);
  bool is_moe = config.count("NUM_EXPERTS");
  bool split_kv = attn_task_num > num_kv_heads * batch_size;

  Tensor exec_queue_host = Tensor::Empty({kNumSM, kStaticTileSchedulerMaxTasks}, DataType::Int(32),
                                         preferred_host_device);
  Tensor exec_queue_device =
      Tensor::Empty({kNumSM, kStaticTileSchedulerMaxTasks}, DataType::Int(32), device);
  int32_t* exec_queue_host_data = static_cast<int32_t*>(exec_queue_host->data);

  // Generate round-robin static execution queue
  std::vector<int32_t> task_counts(kNumSM, 0);
  int cur_sm = 0;

  auto f_push_task = [&exec_queue_host_data, &task_counts, &cur_sm](int m_idx, int n_idx, int k_idx,
                                                                    JobType job_type) {
    int task_id = task_counts[cur_sm]++;
    exec_queue_host_data[cur_sm * kStaticTileSchedulerMaxTasks + task_id] =
        PackInto32bit(static_cast<int32_t>(job_type), m_idx, n_idx, k_idx);
    cur_sm = (cur_sm + 1) % kNumSM;
  };

  for (int i = 0; i < kMaxNumEtensors; ++i) {
    f_push_task(i, 0, 0, JobType::kInitEtensor);
  }

  for (int n_idx = 0; n_idx < ceildiv((num_qo_heads + 2 * num_kv_heads) * head_dim, kGemmTileBlkN);
       ++n_idx) {
    for (int k_idx = 0; k_idx < split_qkv_project; ++k_idx) {
      f_push_task(0, n_idx, k_idx, JobType::kGemmQKVProj);
    }
  }

  for (int i = 0; i < kNumSM; ++i) {
    f_push_task(i, 0, 0, JobType::kWaitEtensorInit);
  }

  int32_t m_split = ceildiv(kNumSM, num_qo_heads + 2 * num_kv_heads);
  int32_t m_tile = ceildiv(batch_size, m_split);
  m_split = ceildiv(batch_size, m_tile);

  for (int m_idx = 0; m_idx < m_split; ++m_idx) {
    for (int n_idx = 0; n_idx < num_qo_heads; ++n_idx) {
      f_push_task(m_idx, n_idx, -1, JobType::kQReduceNormRope);
    }
  }

  for (int m_idx = 0; m_idx < m_split; ++m_idx) {
    for (int n_idx = 0; n_idx < num_kv_heads; ++n_idx) {
      f_push_task(m_idx, n_idx, -1, JobType::kKReduceNormRopeAppend);
    }
  }

  for (int m_idx = 0; m_idx < m_split; ++m_idx) {
    for (int n_idx = 0; n_idx < num_kv_heads; ++n_idx) {
      f_push_task(m_idx, n_idx, -1, JobType::kVReduceAppend);
    }
  }

  for (int m_idx = 0; m_idx < std::min(ceildiv(attn_task_num, kNumWarpgroupPerBlock), kNumSM);
       ++m_idx) {
    f_push_task(m_idx, -1, -1, JobType::kBatchAttention);
  }

  if (split_kv) {
    for (int m_idx = 0;
         m_idx < ceildiv(batch_size * num_qo_heads, kNumWarpgroupPerBlock * kNumWarpPerWarpgroup);
         ++m_idx) {
      f_push_task(m_idx, -1, -1, JobType::kBatchMerge);
    }
  }

  for (int n_idx = 0; n_idx < ceildiv(hidden_size, kGemmTileBlkN); ++n_idx) {
    for (int k_idx = 0; k_idx < split_o_project; ++k_idx) {
      f_push_task(0, n_idx, k_idx, JobType::kGemmOProj);
    }
  }

  if (tp_size > 1) {
    int32_t m_split_o_proj_reduce =
        std::min(batch_size, kNumSM / (hidden_size / kSplitKReduceTileNUnit));
    int32_t n_tile_o_proj_reduce =
        (ceildiv(kSplitKReduceTileNRepeat, ceildiv(batch_size, m_split_o_proj_reduce)) *
         kSplitKReduceTileNUnit);
    int32_t m_tile_o_reduce = ceildiv(batch_size, m_split_o_proj_reduce);
    m_split_o_proj_reduce = ceildiv(batch_size, m_tile_o_reduce);
    for (int n_idx = 0; n_idx < ceildiv(hidden_size, n_tile_o_proj_reduce); ++n_idx) {
      for (int m_idx = 0; m_idx < m_split_o_proj_reduce; ++m_idx) {
        f_push_task(m_idx, n_idx, 0, JobType::kGemmOReduce);
      }
    }

    for (int m_idx = 0; m_idx < ceildiv(batch_size, kAllReduceTileMTile); ++m_idx) {
      for (int n_idx = 0; n_idx < ceildiv(hidden_size / tp_size, kAllReduceTileNTile); ++n_idx) {
        f_push_task(m_idx, n_idx, 0, JobType::kOAllReduce);
      }
    }
  }

  for (int m_idx = 0; m_idx < batch_size; ++m_idx) {
    f_push_task(m_idx, -1, -1, JobType::kAttnAddRMSNorm);
  }

  if (is_moe) {
    int gating_split_k_factor = config["GATING_SPLIT_K_FACTOR"].cast<int>();
    int num_experts_per_tok = config["NUM_EXPERTS_PER_TOK"].cast<int>();
    int num_experts = config["NUM_EXPERTS"].cast<int>();
    for (int m_idx = 0; m_idx < ceildiv(batch_size, kGatingBlkM); ++m_idx) {
      for (int k_idx = 0; k_idx < gating_split_k_factor; ++k_idx) {
        f_push_task(m_idx, 0, k_idx, JobType::kMoeGating);
      }
    }
    for (int m_idx = 0; m_idx < kNumSM; ++m_idx) {
      f_push_task(m_idx, 0, 0, JobType::kMoeTopkSoftmax);
    }
    f_push_task(0, 0, 0, JobType::kMoeAlign);
    for (int m_idx = 0; m_idx < kNumSM; ++m_idx) {
      f_push_task(m_idx, 0, 0, JobType::kMoeCountAndSort);
    }
    const auto f_get_max_num_tokens_padded =
        ffi::Function::GetGlobalRequired("tirx.megakernel.get_max_num_tokens_padded");
    int max_num_tokens_padded =
        f_get_max_num_tokens_padded(batch_size, num_experts_per_tok, num_experts, kMoeBlkM)
            .cast<int>();
    for (int m_idx = 0; m_idx < max_num_tokens_padded / kMoeBlkM; ++m_idx) {
      for (int n_idx = 0; n_idx < (intermediate_size * 2) / kGroupGemmBlkN; ++n_idx) {
        f_push_task(m_idx, n_idx, 0, JobType::kMoeGroupGemmGateUpSilu);
      }
    }
    for (int m_idx = 0; m_idx < max_num_tokens_padded / kMoeBlkM; ++m_idx) {
      for (int n_idx = 0; n_idx < hidden_size / kGroupGemmBlkN; ++n_idx) {
        f_push_task(m_idx, n_idx, 0, JobType::kMoeGroupGemmDown);
      }
    }
  } else {
    int gate_up_proj_split_k_factor =
        config["GATE_UP_PROJ_SPLIT_K_FACTOR_DICT"].cast<ffi::Map<int, int>>()[tp_size];
    int down_proj_split_k_factor =
        config["DOWN_PROJ_SPLIT_K_FACTOR_DICT"].cast<ffi::Map<int, int>>()[tp_size];
    TVM_FFI_ICHECK_NE(down_proj_split_k_factor, -1);
    TVM_FFI_ICHECK_NE(gate_up_proj_split_k_factor, -1);
    if (gate_up_proj_split_k_factor == 1) {
      TVM_FFI_ICHECK_EQ(intermediate_size % kGemmTileBlkN, 0);
      for (int n_idx = 0; n_idx < 2 * intermediate_size / kGemmTileBlkN; ++n_idx) {
        f_push_task(0, n_idx, 0, JobType::kGateUpSilu);
      }
    } else {
      TVM_FFI_ICHECK_EQ(intermediate_size % kGemmTileBlkN, 0);
      for (int n_idx = 0; n_idx < intermediate_size / kGemmTileBlkN; ++n_idx) {
        for (int k_idx = 0; k_idx < gate_up_proj_split_k_factor; ++k_idx) {
          f_push_task(0, n_idx, k_idx, JobType::kGemmGateUpProj);
          f_push_task(0, n_idx + intermediate_size / kGemmTileBlkN, k_idx,
                      JobType::kGemmGateUpProj);
        }
      }
      int32_t m_split_gate_up_proj_reduce =
          std::min(batch_size, kNumSM / (intermediate_size * 2 / kSplitKReduceTileNUnit));
      int32_t m_tile_gate_up_proj_reduce = ceildiv(batch_size, m_split_gate_up_proj_reduce);
      m_split_gate_up_proj_reduce = ceildiv(batch_size, m_tile_gate_up_proj_reduce);
      for (int m_idx = 0; m_idx < m_split_gate_up_proj_reduce; ++m_idx) {
        for (int n_idx = 0; n_idx < ceildiv(intermediate_size * 2, kGemmTileBlkN); ++n_idx) {
          f_push_task(m_idx, n_idx, 0, JobType::kGateUpProjReduce);
        }
      }
      for (int n_idx = 0; n_idx < ceildiv(intermediate_size, kSiluMultiplyTileSize); ++n_idx) {
        f_push_task(n_idx, 0, 0, JobType::kSplitSiluMultiply);
      }
    }

    for (int n_idx = 0; n_idx < ceildiv(hidden_size, kGemmTileBlkN); ++n_idx) {
      for (int k_idx = 0; k_idx < down_proj_split_k_factor; ++k_idx) {
        f_push_task(0, n_idx, k_idx, JobType::kGemmDownProj);
      }
    }

    if (tp_size > 1) {
      int32_t m_split_down_proj_reduce =
          std::min(kNumSM / (hidden_size / kSplitKReduceTileNUnit), batch_size);
      int32_t n_tile_down_proj_reduce =
          (ceildiv(kSplitKReduceTileNRepeat, ceildiv(batch_size, m_split_down_proj_reduce)) *
           kSplitKReduceTileNUnit);
      int32_t m_tile_down_proj_reduce = ceildiv(batch_size, m_split_down_proj_reduce);
      m_split_down_proj_reduce = ceildiv(batch_size, m_tile_down_proj_reduce);
      for (int m_idx = 0; m_idx < m_split_down_proj_reduce; ++m_idx) {
        for (int n_idx = 0; n_idx < ceildiv(hidden_size, n_tile_down_proj_reduce); ++n_idx) {
          f_push_task(m_idx, n_idx, 0, JobType::kDownProjReduce);
        }
      }

      for (int m_idx = 0; m_idx < ceildiv(batch_size, kAllReduceTileMTile); ++m_idx) {
        for (int n_idx = 0; n_idx < ceildiv(hidden_size / tp_size, kAllReduceTileNTile); ++n_idx) {
          f_push_task(m_idx, n_idx, 0, JobType::kDownProjAllReduce);
        }
      }
    }
  }
  for (int m_idx = 0; m_idx < batch_size; ++m_idx) {
    f_push_task(m_idx, 0, 0, JobType::kMLPAddRMSNorm);
  }

  // Set the uninitialized queue to kEnd
  for (int sm_id = 0; sm_id < kNumSM; ++sm_id) {
    for (int task_id = task_counts[sm_id]; task_id < kStaticTileSchedulerMaxTasks; ++task_id) {
      exec_queue_host_data[sm_id * kStaticTileSchedulerMaxTasks + task_id] =
          PackInto32bit(static_cast<int32_t>(JobType::kEnd), -1, -1, -1);
    }
  }

  // Transfer to device
  DLTensor exec_queue_device_dl = *exec_queue_device.operator->();
  Tensor::CopyFromTo(exec_queue_host.operator->(), &exec_queue_device_dl);
  return exec_queue_device;
}

Array<Array<Tensor>> GenerateExecQueueDynamic(Tensor exec_queue_device_buf,
                                              Tensor exec_queue_host_buf, int tp_size,
                                              std::string model_name, int num_layers,
                                              TVMStreamHandle copy_stream) {
  NVTXScopedRange range("Generate execution queue");
  int elem_per_layer = kDyanmicTileSchedulerMaxTasks + 4;
  TVM_FFI_ICHECK(exec_queue_device_buf.dtype() == DataType::Int(32));
  TVM_FFI_ICHECK(exec_queue_host_buf.dtype() == DataType::Int(32));
  TVM_FFI_ICHECK(exec_queue_device_buf.Shape().Product() >= num_layers * elem_per_layer);
  TVM_FFI_ICHECK(exec_queue_host_buf.Shape().Product() >= num_layers * elem_per_layer);
  int32_t* exec_queue_host_data = static_cast<int32_t*>(exec_queue_host_buf->data);
  int num_tasks = 0;
  const auto f_get_config =
      tvm::ffi::Function::GetGlobalRequired("tirx.megakernel.get_model_config");
  auto config = f_get_config(model_name).cast<ffi::Map<ffi::String, ffi::Any>>();
  int num_qo_heads = config["NUM_ATTENTION_HEADS"].cast<int>() / tp_size;
  int num_kv_heads = config["NUM_KEY_VALUE_HEADS"].cast<int>() / tp_size;
  int head_dim = config["HEAD_DIM"].cast<int>();
  int split_qkv_project = config["SPLIT_QKV_PROJECT_DICT"].cast<ffi::Map<int, int>>()[tp_size];

  // fill the exec queue with -1
  for (int i = 0; i < num_layers * elem_per_layer; ++i) {
    exec_queue_host_data[i] = -1;
  }

  auto f_push_task = [&exec_queue_host_data, &num_tasks, &elem_per_layer](
                         int m_idx, int n_idx, int k_idx, JobType job_type, int layer_id) {
    exec_queue_host_data[layer_id * elem_per_layer + num_tasks] =
        PackInto32bit(static_cast<int32_t>(job_type), m_idx, n_idx, k_idx);
    num_tasks++;
  };

  for (int layer_id = 0; layer_id < num_layers; ++layer_id) {
    num_tasks = 0;
    // Push initial tasks.
    for (int i = 0; i < kMaxNumEtensors; ++i) {
      f_push_task(i, 0, 0, JobType::kInitEtensor, layer_id);
    }
    for (int n_idx = ceildiv((num_qo_heads + 2 * num_kv_heads) * head_dim, kGemmTileBlkN) - 1;
         n_idx >= 0; --n_idx) {
      for (int k_idx = 0; k_idx < split_qkv_project; ++k_idx) {
        f_push_task(0, n_idx, k_idx, JobType::kGemmQKVProj, layer_id);
      }
    }
    // Set head & tail.
    exec_queue_host_data[layer_id * elem_per_layer + kDyanmicTileSchedulerMaxTasks] = 0;
    exec_queue_host_data[layer_id * elem_per_layer + kDyanmicTileSchedulerMaxTasks + 1] = num_tasks;
  }

  // Transfer to device
  DLTensor exec_queue_device_dl = *exec_queue_device_buf.operator->();
  Tensor::CopyFromTo(exec_queue_host_buf.operator->(), &exec_queue_device_dl, copy_stream);

  // Slice the execution queue to get the dynamic execution queue.
  std::vector<Array<Tensor>> queue_by_layer;
  queue_by_layer.reserve(num_layers);
  for (int layer_id = 0; layer_id < num_layers; ++layer_id) {
    Tensor exec_queue = exec_queue_device_buf.CreateView(
        {elem_per_layer}, DataType::Int(32),
        /*relative_byte_offset=*/layer_id * elem_per_layer * DataType::Int(32).bytes());
    Tensor queue_tasks =
        exec_queue.CreateView({megakernel::kDyanmicTileSchedulerMaxTasks}, DataType::Int(32));
    Tensor queue_head =
        exec_queue.CreateView({1}, DataType::Int(32),
                              /*relative_byte_offset=*/megakernel::kDyanmicTileSchedulerMaxTasks *
                                  DataType::Int(32).bytes());
    Tensor queue_tail = exec_queue.CreateView(
        {1}, DataType::Int(32),
        /*relative_byte_offset=*/
        (megakernel::kDyanmicTileSchedulerMaxTasks + 1) * DataType::Int(32).bytes());

    queue_by_layer.push_back(Array<Tensor>({queue_tasks, queue_head, queue_tail}));
  }
  return queue_by_layer;
}

Tensor GetExecQueueStatic(tvm::ffi::AnyView vm_arg, ffi::ObjectRef gen_exec_func,
                          ffi::Shape cache_args) {
  static std::unordered_map<size_t, Tensor> cache;
  // calculate hash key
  size_t hash = cache_args.size();
  for (int32_t i : cache_args) {
    hash ^= std::hash<int32_t>()(i) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  }
  if (cache.count(hash)) {
    // hit cache
    return cache[hash];
  } else {
    // miss cache
    using namespace vm;
    VirtualMachine* vm = VirtualMachine::GetContextPtr(vm_arg);
    VMClosure func = tvm::Downcast<VMClosure>(gen_exec_func);
    std::vector<tvm::ffi::AnyView> packed_args(1);
    packed_args[0] = cache_args;
    tvm::ffi::Any rv;
    vm->InvokeClosurePacked(func, tvm::ffi::PackedArgs(packed_args.data(), packed_args.size()),
                            &rv);
    Tensor exec_queue = rv.cast<Tensor>();
    cache[hash] = exec_queue;
    return exec_queue;
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("megakernel.get_event_tensors_on_layer", GetEventTensorsOnLayer)
      .def("megakernel.generate_exec_queue_static", GenerateExecQueueStatic)
      .def("megakernel.generate_exec_queue_dynamic", GenerateExecQueueDynamic)
      .def("megakernel.horizontal_fusion.get_exec_queue_static", GetExecQueueStatic);
}

}  // namespace megakernel
}  // namespace vm
}  // namespace runtime
}  // namespace tvm
