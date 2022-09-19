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
#include "../utils.h"

namespace tvm {
namespace tir {

/*!
 * \brief Collect the block and index where the buffer is read.
 * \note The buffers are expected to be read by only one BufferLoad
 */
class BufferPosCollector : public StmtExprVisitor {
 public:
  BufferPosCollector(const Array<Buffer>& input_buffers, const Buffer& output_buffer) {
    for (const Buffer& buf : input_buffers) {
      read_buffers_.insert(buf.get());
    }
    write_buffer_ = output_buffer.get();
  }

  const std::unordered_map<const BufferNode*, std::pair<Block, int>>& GetReadBufferLocations() const {
    return read_buffer_locs_;
  }

  const std::unordered_map<const BufferNode*, std::pair<Block, int>>& GetWriteBufferLocations() const {
    return write_buffer_locs_;
  }

  const std::unordered_map<const BufferNode*, Optional<IndexMap>>& GetBufferIndexMap() const {
    return buffer_index_maps_;
  }

 private:
  void VisitStmt_(const ForNode* op) final {
    loop_stack_.push_back(GetRef<For>(op));
    StmtVisitor::VisitStmt_(op);
    loop_stack_.pop_back();
  }

  void VisitStmt_(const BlockRealizeNode* op) final {
    BlockRealize outer_block_realize = GetRef<BlockRealize>(op);
    std::swap(outer_block_realize, cur_realize_);
    StmtVisitor::VisitStmt_(op);
    std::swap(cur_realize_, outer_block_realize);
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    const Buffer& buffer = op->buffer;
    if (read_buffers_.count(buffer.get())) {
      Map<Var, PrimExpr> subst_map;
      for (size_t i = 0; i < cur_realize_->iter_values.size(); i++) {
        const Var& var = cur_realize_->block->iter_vars[i]->var;
        const PrimExpr& value = cur_realize_->iter_values[i];
        subst_map.Set(var, value);
      }
      Array<PrimExpr> subst_indices;
      for (const PrimExpr& e : op->indices) {
        subst_indices.push_back(Substitute(e, subst_map));
      }
      buffer_index_maps_[buffer.get()] = SuggestIndexMap(/*buffer=*/buffer,                      //
                                                         /*indices=*/subst_indices,              //
                                                         /*loops=*/loop_stack_,                  //
                                                         /*predicate=*/cur_realize_->predicate,  //
                                                         /*analyzer=*/&analyzer_);
      int buffer_index = GetBufferIndex(cur_realize_->block, buffer, false);
      ICHECK(buffer_index != -1);
      read_buffer_locs_[buffer.get()] = std::make_pair(cur_realize_->block, buffer_index);
    }
  }
  void VisitStmt_(const BufferStoreNode* op) final {
    const Buffer& buffer = op->buffer;
    if(write_buffer_ == buffer.get()) {
      Map<Var, PrimExpr> subst_map;
      for (size_t i = 0; i < cur_realize_->iter_values.size(); i++) {
        const Var& var = cur_realize_->block->iter_vars[i]->var;
        const PrimExpr& value = cur_realize_->iter_values[i];
        subst_map.Set(var, value);
      }
      Array<PrimExpr> subst_indices;
      for (const PrimExpr& e : op->indices) {
        subst_indices.push_back(Substitute(e, subst_map));
      }
      buffer_index_maps_[buffer.get()] = SuggestIndexMap(/*buffer=*/buffer,                      //
                                                         /*indices=*/subst_indices,              //
                                                         /*loops=*/loop_stack_,                  //
                                                         /*predicate=*/cur_realize_->predicate,  //
                                                         /*analyzer=*/&analyzer_);
      int buffer_index = GetBufferIndex(cur_realize_->block, buffer, true);
      ICHECK(buffer_index != -1);
      write_buffer_locs_[buffer.get()] = std::make_pair(cur_realize_->block, buffer_index);
    }
    StmtVisitor::VisitStmt_(op);
  }

  static int GetBufferIndex(const Block& block, const Buffer& buffer, bool is_write) {
    if(is_write){
      for(size_t i = 0; i < block->writes.size(); i++) {
        if(block->writes[i]->buffer.same_as(buffer)) {
          return i;
        } 
      }
    }else{
      for (size_t i = 0; i < block->reads.size(); i++) {
        if (block->reads[i]->buffer.same_as(buffer)) {
          return i;
        }
      }
    }
    return -1;
  }

 private:
  /*! \brief All interested buffer. */
  std::unordered_set<const BufferNode*> read_buffers_;
  const BufferNode* write_buffer_;
  /*! \brief The result mapping from buffer to its inner-most block and read index. */
  std::unordered_map<const BufferNode*, std::pair<Block, int>> read_buffer_locs_;
  std::unordered_map<const BufferNode*, std::pair<Block, int>> write_buffer_locs_;
  /*! \brief The result mapping from buffer to its IndexMap. */
  std::unordered_map<const BufferNode*, Optional<IndexMap>> buffer_index_maps_;

  /*! \brief Loop stack for calculating IndexMap. */
  Array<For> loop_stack_;
  /*! \brief Arithmetic analyzer. */
  arith::Analyzer analyzer_;
  /*! \brief Current BlockRealize scope, used in recursive visit */
  BlockRealize cur_realize_;
};

bool RewriteLayout(const Schedule& sch) {
  std::vector<std::pair<StmtSRef, String>> results;
  for (const auto& kv : sch->mod()->functions) {
    const GlobalVar& g_var = kv.first;
    const String& func_name = g_var->name_hint;
    const auto* prim_func = kv.second.as<PrimFuncNode>();
    // Only consider PrimFunc
    if (prim_func == nullptr) {
      continue;
    }
    Array<Buffer> input_buffers;
    for (int i = 0; i < static_cast<int>(prim_func->params.size())-1; i++) {
      input_buffers.push_back(prim_func->buffer_map.at(prim_func->params[i]));
    }
    Buffer output_buffer = prim_func->buffer_map.at(prim_func->params.back());
    // Collect Buffer read positions
    BufferPosCollector collector(input_buffers, output_buffer);
    collector(prim_func->body);
    const auto& read_locations = collector.GetReadBufferLocations();
    const auto& index_maps = collector.GetBufferIndexMap();
    // Check all buffers are collected
    if (read_locations.size() != input_buffers.size()) {
      return false;
    }

    for (const auto& kv : read_locations) {
      const Buffer& buffer = GetRef<Buffer>(kv.first);
      const Block& block = kv.second.first;
      int buffer_index = kv.second.second;

      // Get IndexMap
      const Optional<IndexMap> index_map = index_maps.at(buffer.get());
      if (!index_map.defined()) {
        continue;
      }

      // Apply schedule
      BlockRV block_rv = sch->GetBlock(block->name_hint, func_name);
      BlockRV cached_block_rv = sch->CacheRead(block_rv, buffer_index, "global");
      sch->TransformLayout(block_rv, buffer_index, BufferIndexType::kRead, index_map.value());
      sch->Annotate(cached_block_rv, attr::meta_schedule_layout_rewrite_preproc, const_true());
    }
    for (const auto& kv : collector.GetWriteBufferLocations()) {
      const Buffer& buffer = GetRef<Buffer>(kv.first);
      const Block& block = kv.second.first;
      int buffer_index = kv.second.second;

      // Get IndexMap
      const Optional<IndexMap> index_map = index_maps.at(buffer.get());
      if (!index_map.defined()) {
        continue;
      }

      // Apply schedule
      BlockRV block_rv = sch->GetBlock(block->name_hint, func_name);
      BlockRV cached_block_rv = sch->CacheWrite(block_rv, buffer_index, "global");
      sch->TransformLayout(block_rv, buffer_index, BufferIndexType::kWrite, index_map.value());
      sch->Annotate(cached_block_rv, attr::meta_schedule_layout_rewrite_postproc, const_true());
    }
  }
  return true;
}

}  // namespace tir

namespace meta_schedule {
/*! \brief Layout Rewrite. */
class RewriteLayoutNode : public PostprocNode {
 public:
  // Inherited from PostprocNode
  void InitializeWithTuneContext(const TuneContext& context) final {}

  // Inherited from PostprocNode
  bool Apply(const tir::Schedule& sch) final { return tir::RewriteLayout(sch); }

  static constexpr const char* _type_key = "meta_schedule.RewriteLayout";
  TVM_DECLARE_FINAL_OBJECT_INFO(RewriteLayoutNode, PostprocNode);
};

Postproc Postproc::RewriteLayout() {
  auto n = make_object<RewriteLayoutNode>();
  return Postproc(n);
}

TVM_REGISTER_NODE_TYPE(RewriteLayoutNode);
TVM_REGISTER_GLOBAL("meta_schedule.PostprocRewriteLayout").set_body_typed(Postproc::RewriteLayout);

}  // namespace meta_schedule
}  // namespace tvm
