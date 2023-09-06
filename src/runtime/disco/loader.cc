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
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <functional>
#include <numeric>
#include <string>
#include <vector>

#include "../file_utils.h"
#include "../relax_vm/ndarray_cache_support.h"
#include "./builtin.h"
#include "./utils.h"

namespace tvm {
namespace runtime {

using relax_vm::NDArrayCacheMetadata;
using FileRecord = NDArrayCacheMetadata::FileRecord;
using ParamRecord = NDArrayCacheMetadata::FileRecord::ParamRecord;
using relax_vm::LoadShardInfoFromStr;

/*! \brief An object that helps to load parameters in shards. */
class ShardLoaderObj : public Object {
 public:
  /*! \brief Create a shard loader. */
  static ObjectRef Create(const std::string& path_to_metadata, const std::string& metadata,
                          const std::string& shard_info,
                          PackedFunc f_shard);
  /*! \brief Load the i-th parameter */
  NDArray Load(std::string weight_name) const;
  /*! \brief Slice the given tensor at a specific dimension */
  NDArray Shard(NDArray source, int dim, int num_slices) const;

  static constexpr const char* _type_key = "runtime.disco.ShardLoader";
  TVM_DECLARE_FINAL_OBJECT_INFO(ShardLoaderObj, Object);

 public:
  /*! \brief Information of how each weight is stored and sharded */
  struct ShardInfo {
    const FileRecord* file;
    const ParamRecord* param;
    int shard_dim;
  };
  /*! \brief The metadata loaded from `ndarray-cache.json` */
  NDArrayCacheMetadata metadata_;
  /*! \brief Sharding information for each weight */
  std::unordered_map<std::string, ShardInfo> shard_info_;
  /*! \brief A method to slice a 3-D tensor */
  PackedFunc f_shard_;
  /*! \brief The current file opened to load weights in it */
  mutable const FileRecord* current_file_;
  /*! \brief The context of the current file to be loaded from */
  mutable std::string current_file_stream_;
};

TVM_REGISTER_OBJECT_TYPE(ShardLoaderObj);

/*!
 * \brief Get the shape of a result tensor if it is scattered along a given axis.
 * \param shape The shape of the input tensor.
 * \param dim The axis along which the tensor is scattered.
 * \param num_shards The number of shards.
 * \return The shape of the result tensor.
 */
inline std::vector<ShapeTuple::index_type> ShardShape(const ShapeTuple& shape, int dim,
                                                      int num_shards) {
  CHECK(0 <= dim && dim < static_cast<int>(shape.size()))
      << "ValueError: Cannot scatter at dim " << dim << ", because "
      << "shape is " << shape << ".";
  CHECK_EQ(shape[dim] % num_shards, 0)
      << "ValueError: The shape " << shape << " cannot be scattered at dim " << dim << " into "
      << num_shards << " shards.";
  std::vector<ShapeTupleObj::index_type> result{shape.begin(), shape.end()};
  result[dim] /= num_shards;
  return result;
}


ObjectRef ShardLoaderObj::Create(const std::string& path_to_metadata, const std::string& metadata,
                                 const std::string& shard_info,
                                 PackedFunc f_shard) {
  ObjectPtr<ShardLoaderObj> n = make_object<ShardLoaderObj>();
  n->f_shard_ = f_shard;
  n->metadata_ = NDArrayCacheMetadata::LoadFromStr(metadata, path_to_metadata);
  n->current_file_ = nullptr;
  n->shard_info_.clear();
  std::unordered_map<std::string, int> shards = LoadShardInfoFromStr(shard_info);
  for (const FileRecord& file_record : n->metadata_.records) {
    for (const ParamRecord& param_record : file_record.records) {
      const std::string& name = param_record.name;
      int shard_id = shards.count(name) ? shards[name] : -1;
      n->shard_info_[name]=ShardInfo{&file_record, &param_record, shard_id};
    }
  }
  return ObjectRef(std::move(n));
}

std::string GetSiblingPath(const std::string& path, const std::string& filename) {
  size_t found = path.find_last_of("/\\");
  if (found != std::string::npos) {
    return path.substr(0, found + 1) + filename;
  }
  LOG(FATAL) << "ValueError: Cannot find the parent directory: " << path;
}

NDArray ShardLoaderObj::Load(std::string weight_name) const {
  DiscoWorker* worker = DiscoWorker::ThreadLocal();
  int shard_idx = worker->worker_id;
  Device device = worker->default_device;
  const auto& shard_info = shard_info_.at(weight_name);
  const ParamRecord* param = shard_info.param;
  const FileRecord* file = shard_info.file;
  int shard_dim = shard_info.shard_dim;
  int num_shards = worker->num_workers;
  Optional<NDArray> send = NullOpt;
  if (shard_idx == 0) {
    if (file != current_file_) {
      current_file_ = file;
      std::string file_name = GetSiblingPath(this->metadata_.path, file->data_path);
      LoadBinaryFromFile(file_name, &this->current_file_stream_);
    }
    auto f_load = [](NDArray param, const void* data, size_t nbytes) {
      param.CopyFromBytes(data, nbytes);
    };
    if(shard_dim != -1){
      send = this->Shard(param->Load(device, &this->current_file_stream_, f_load), shard_dim,
                        num_shards);
    } else {
      send = param->Load(device, &this->current_file_stream_, f_load);
    }
  }
  if(shard_dim != -1){
    NDArray recv =
        NDArray::Empty(ShardShape(param->shape, shard_dim, num_shards), param->dtype, device);
    ScatterFromWorker0(send, recv);
    return recv;
  } else {
    NDArray recv;
    if(send.defined()){
      recv = NDArray(send.value());
    } else {
      recv = NDArray::Empty(param->shape, param->dtype, device);
    }
    return BroadcastFromWorker0(recv);
  }
}

NDArray ShardLoaderObj::Shard(NDArray source, int dim, int num_slices) const {
  ICHECK(dim >= 0 && dim < source->ndim);
  // Assemble a flattened 3d src tensor
  int64_t src_flat[3] = {1, 1, 1};
  {
    const int64_t* s = source.Shape().data();
    int ndim = source->ndim;
    src_flat[0] = std::accumulate(&s[0], &s[dim], 1, std::multiplies<int64_t>());
    src_flat[1] = s[dim];
    src_flat[2] = std::accumulate(&s[dim + 1], &s[ndim], 1, std::multiplies<int64_t>());
  }
  DLTensor src_tensor = *source.operator->();
  src_tensor.ndim = 3;
  src_tensor.shape = src_flat;
  // Assmeble a flattened 4d dst tensor
  int64_t dst_flat[4] = {num_slices, src_flat[0], src_flat[1] / num_slices, src_flat[2]};
  NDArray destination{nullptr};
  {
    std::vector<ShapeTuple::index_type> dst_shape = ShardShape(source.Shape(), dim, num_slices);
    dst_shape.insert(dst_shape.begin(), static_cast<ShapeTuple::index_type>(num_slices));
    destination = NDArray::Empty(dst_shape, source->dtype, source->device);
  }
  DLTensor dst_tensor = *destination.operator->();
  dst_tensor.ndim = 4;
  dst_tensor.shape = dst_flat;
  // Copy slices using the API
  if (this->f_shard_.defined()){
    this->f_shard_(&src_tensor, num_slices, &dst_tensor);
  } else {
    DLTensor src_sub_tensor = *source.operator->();
    src_sub_tensor.ndim = 2;
    src_sub_tensor.shape = dst_flat + 2;

    int64_t old_src_byte_offset = src_sub_tensor.byte_offset;
    int64_t dst_offset = 0;
    for (int i = 0; i < num_slices; ++i) {
      dst_tensor.data = destination->data;
      src_sub_tensor.byte_offset = old_src_byte_offset + src_sub_tensor.dtype.bits * dst_flat[2] * dst_flat[3] * i / 8;
      for (int j = 0; j < dst_flat[1]; j++) {
        ArrayCopyToBytes(&src_sub_tensor, (char*) dst_tensor.data + dst_offset,
                        src_sub_tensor.dtype.bits * dst_flat[2] * dst_flat[3] / 8);
        src_sub_tensor.byte_offset += src_sub_tensor.dtype.bits * src_flat[1] * src_flat[2]/ 8;
        dst_offset += dst_tensor.dtype.bits * dst_flat[2] * dst_flat[3] / 8;
      }
    }
  }
  return destination;
}

TVM_REGISTER_GLOBAL("runtime.disco.ShardLoader").set_body_typed(ShardLoaderObj::Create);
TVM_REGISTER_GLOBAL("runtime.disco.ShardLoaderLoad")
    .set_body_typed([](ObjectRef loader_obj, String weight_name) {
      const auto* loader = loader_obj.as<ShardLoaderObj>();
      CHECK(loader != nullptr) << "TypeError: Expected ShardLoaderObj, but gets: "
                               << loader_obj->GetTypeKey();
      return loader->Load(weight_name);
    });

TVM_REGISTER_GLOBAL("runtime.disco.ShardLoaderLoadAll").set_body_typed([](ObjectRef loader_obj) {
  const auto* loader = loader_obj.as<ShardLoaderObj>();
  CHECK(loader != nullptr) << "TypeError: Expected ShardLoaderObj, but gets: "
                           << loader_obj->GetTypeKey();
  Array<NDArray> shards;
  for (int i = 0; i < static_cast<int>(loader->shard_info_.size()); i++) {
    shards.push_back(loader->Load("param_" + std::to_string(i)));
  }
  return shards;
});

}  // namespace runtime
}  // namespace tvm
