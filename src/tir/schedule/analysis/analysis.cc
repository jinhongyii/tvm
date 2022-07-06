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
#include <tvm/runtime/container/optional.h>
#include <tvm/tir/expr.h>

#include <unordered_map>

#include "../ir_comparator.h"
#include "../utils.h"

namespace tvm {
namespace tir {

/******** IR Module ********/

const PrimFuncNode* GetRootPrimFunc(const IRModule& mod, const StmtNode* root_block,
                                    GlobalVar* result_g_var) {
  for (const auto& kv : mod->functions) {
    const GlobalVar& g_var = kv.first;
    const BaseFunc& base_func = kv.second;
    if (const auto* func = base_func.as<PrimFuncNode>()) {
      if (const auto* realize = func->body.as<BlockRealizeNode>()) {
        if (realize->block.get() == root_block) {
          if (result_g_var != nullptr) {
            *result_g_var = g_var;
          }
          return func;
        }
      }
    }
  }
  LOG(FATAL) << "IndexError: Could not get the corresponding function in the schedule state of the "
                "statement:\n"
             << GetRef<Stmt>(root_block);
  throw;
}

/******** Scope ********/

StmtSRef GetScopeRoot(const ScheduleState& self, const StmtSRef& sref,
                      bool require_stage_pipeline) {
  class RootBlockError : public ScheduleError {
   public:
    explicit RootBlockError(IRModule mod) : mod_(mod) {}
    IRModule mod() const final { return mod_; }
    String FastErrorString() const final {
      return "ScheduleError: The primitive does not operate on the root block";
    }
    String DetailRenderTemplate() const final {
      return "The primitive does not operate on the root block";
    }
    Array<ObjectRef> LocationsOfInterest() const final { return {}; }
    IRModule mod_;
  };

  class NotStagePipelineError : public ScheduleError {
   public:
    explicit NotStagePipelineError(IRModule mod, Block block) : mod_(mod), block_(block) {}
    IRModule mod() const final { return mod_; }
    String FastErrorString() const final {
      return "ScheduleError: The scope root is not a stage pipeline";
    }
    String DetailRenderTemplate() const final {
      return R"(The scope {0} is not a stage pipeline.
Definition of a scope that is a stage pipeline:
- The region cover property holds for every of its child blocks
- No write-after-read dependency or opaque dependency,
- only read-after-write and write-after-write are allowed
- All the statements in the scope are schedulable statements, i.e. Block and For
)";
    }
    Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }
    IRModule mod_;
    Block block_;
  };

  StmtSRef scope_root_sref{nullptr};
  StmtSRef scope_root_subtree{nullptr};
  // Step 1. Find the scope root and the subtree that the given sref is in
  {
    const StmtSRefNode* p = sref->parent;
    const StmtSRefNode* subtree = sref.get();
    for (; p != nullptr; subtree = p, p = p->parent) {
      if (p->stmt->IsInstance<BlockNode>()) {
        scope_root_sref = GetRef<StmtSRef>(p);
        scope_root_subtree = GetRef<StmtSRef>(subtree);
        break;
      }
    }
    if (p == nullptr) {
      throw RootBlockError(self->mod);
    }
  }
  // Step 2. Handle `require_stage_pipeline`
  if (require_stage_pipeline) {
    bool stage_pipeline = self->GetBlockInfo(scope_root_sref).scope->stage_pipeline;
    if (stage_pipeline == false) {
      const BlockNode* block = TVM_SREF_TO_BLOCK(block, scope_root_sref);
      throw NotStagePipelineError(self->mod, GetRef<Block>(block));
    }
  }
  return scope_root_sref;
}

ScopeBlockLoopInfo GetScopeBlockLoopInfo(const Block& scope_block) {
  struct Collector : public StmtVisitor {
    void VisitStmt_(const BlockRealizeNode* realize) final {
      result.realizes.push_back(GetRef<BlockRealize>(realize));
      const Array<IterVar>& iter_vars = realize->block->iter_vars;
      const Array<PrimExpr>& iter_values = realize->iter_values;
      ICHECK_EQ(iter_vars.size(), iter_values.size());
      int n = realize->iter_values.size();
      for (int i = 0; i < n; ++i) {
        const IterVar& iter_var = iter_vars[i];
        const PrimExpr& iter_value = iter_values[i];
        std::unordered_set<const VarNode*>* vars = nullptr;
        if (iter_var->iter_type == IterVarType::kDataPar) {
          vars = &result.spatial_vars;
        } else {
          vars = &result.non_spatial_vars;
        }
        PostOrderVisit(iter_value, [vars](const ObjectRef& obj) {
          if (const VarNode* var = obj.as<VarNode>()) {
            vars->insert(var);
          }
        });
      }
    }

    ScopeBlockLoopInfo result;
  } visitor;
  visitor(scope_block->body);
  return std::move(visitor.result);
}

/*!
 * \brief Check whether the given sref_a is higher than or equal to sref_b.
 */
void CheckSRefHigherOrEqual(const StmtSRef& sref_a, const StmtSRef& sref_b) {
  const StmtSRefNode* p = sref_b.get();
  for (; p != nullptr; p = p->parent) {
    if (p == sref_a.get()) {
      return;
    }
  }
  CHECK(false) << "Expect StmtSRef " << sref_a << "to be higher than or equal to " << sref_b;
}

/*!
 * \brief Check the dominant property of a block:
 * the block is the only writer of its output, dominating the reader of its output buffers under the
 * given root scope.
 * \param self The schedule state.
 * \param scope_root_sref The StmtSRef corresponding to the root scope.
 * \param block_sref The block whose dominant property is to be checked.
 * \return A boolean indicating if the block is a dominant block.
 */
bool IsDominantBlock(const ScheduleState& self, const StmtSRef& scope_root_sref,
                     const StmtSRef& block_sref) {
  std::unordered_map<Buffer, Array<StmtSRef>, ObjectPtrHash, ObjectPtrEqual> buffer_writers;
  CheckSRefHigherOrEqual(scope_root_sref, block_sref);
  const BlockNode* maybe_root_block = scope_root_sref->StmtAs<BlockNode>();
  if (maybe_root_block) {
    BlockScope scope = self->GetBlockScope(scope_root_sref);
    buffer_writers = scope->buffer_writers;
  } else {
    // Collect all child blocks of root sub-tree, and merge their buffer writers.
    Array<StmtSRef> child_block_srefs = GetChildBlockSRefOnSRefTree(self, scope_root_sref);
    for (const StmtSRef& child_block_sref : child_block_srefs) {
      BlockScope child_scope = self->GetBlockScope(child_block_sref);
      for (const auto& it : child_scope->buffer_writers) {
        buffer_writers.insert(it);
      }
    }
  }
  // Check whether the input block is the only writer of its outputs
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  for (const BufferRegion& write_region : block->writes) {
    if (buffer_writers.count(write_region->buffer)) {
      if (buffer_writers.at(write_region->buffer).size() != 1) {
        return false;
      }
    }
  }
  return true;
}

/*!
 * \brief A helper function that checks whether a given block is a complete block under the scope,
 * or return the condition it violates if it is not a complete block
 * \param self The schedule state
 * \param block_sref The block to be checked
 * \param scope_root_sref The sref to the root block of the scope that `block_sref` is in
 * \return 0 if the block is a complete block, or a positive integer indicating which condition is
 * first violated
 */
int CheckCompleteBlockErrorCode(const ScheduleState& self, const StmtSRef& block_sref,
                                const StmtSRef& scope_root_sref) {
  // Cond 1. All block vars are data parallel
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  for (const IterVar& iter_var : block->iter_vars) {
    if (iter_var->iter_type != kDataPar) {
      return 1;
    }
  }
  // Cond 2. Dominant: the block is the only writer of its output,
  // dominating the reader of its output buffers
  if (!IsDominantBlock(self, scope_root_sref, block_sref)) {
    return 2;
  }
  // Cond 3. No overlap between the buffers the block reads and writes
  std::unordered_set<const BufferNode*> written_buffers;
  written_buffers.reserve(block->writes.size());
  for (const BufferRegion& write : block->writes) {
    written_buffers.insert(write->buffer.get());
  }
  for (const BufferRegion& read : block->reads) {
    if (written_buffers.count(read->buffer.get())) {
      return 3;
    }
  }
  return 0;
}

static const char* kCompleteBlockDefinition = R"(Definition of a complete block:
1) All block vars are data parallel
2) Dominant: the block is the only writer of its output, dominating the reader of its output buffers
3) No overlap between the buffers the block reads and writes)";

static const char* kReductionBlockDefinition = R"(Definition of a reduction block:
1) The block has the `init` statement
2) All the block bindings are quasi-affine expressions
3) All block vars are either data parallel block vars or reduction block vars
4) Dominant: the block is the only writer of its output, dominating the reader of its output buffers
5) The reduction block vars are not used to index the output buffers)";

static const char* kLocalCompleteBlockDefinition = R"(Definition of a local complete block:
1) All block vars are data parallel
2) Local Dominant: the block is the only writer of its output, dominating the reader of its output buffers under a given subtree
3) No overlap between the buffers the block reads and writes)";

static const char* kLocalReductionBlockDefinition = R"(Definition of a reduction block:
1) The block has the `init` statement
2) All the block bindings are quasi-affine expressions
3) All block vars are either data parallel block vars or reduction block vars
4) Local Dominant: the block is the only writer of its output, dominating the reader of its output buffers under a given subtree
5) The reduction block vars are not used to index the output buffers)";

bool IsCompleteBlock(const ScheduleState& self, const StmtSRef& block_sref,
                     const StmtSRef& scope_root_sref) {
  return CheckCompleteBlockErrorCode(self, block_sref, scope_root_sref) == 0;
}

void CheckCompleteBlock(const ScheduleState& self, const StmtSRef& block_sref,
                        const StmtSRef& scope_root_sref) {
  class IncompleteBlockError : public ScheduleError {
   public:
    explicit IncompleteBlockError(IRModule mod, Block block, int violated_cond)
        : mod_(std::move(mod)), block_(std::move(block)), violated_cond_(violated_cond) {}
    String FastErrorString() const final { return "ScheduleError: Incomplete block"; }
    String DetailRenderTemplate() const final {
      std::ostringstream os;
      os << "The block {0} is not a complete block - it violates condition #" << violated_cond_;
      os << ".\n" << kCompleteBlockDefinition;
      return os.str();
    }
    IRModule mod() const final { return mod_; }
    Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }
    IRModule mod_;
    Block block_;
    int violated_cond_;
  };

  int error_code = CheckCompleteBlockErrorCode(self, block_sref, scope_root_sref);
  if (error_code != 0) {
    const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
    throw IncompleteBlockError(self->mod, GetRef<Block>(block), error_code);
  }
}

/*!
 * \brief A helper function that checks whether a given block is a reduction block under the scope,
 * or return the condition it violates if it is not a reduction block
 * \param self The schedule state
 * \param block_sref The block to be checked
 * \param scope_root_sref The sref to the root block of the scope that `block_sref` is in
 * \return 0 if the block is a reduction block, or a positive integer indicating which condition is
 * first violated
 */
int CheckReductionBlockErrorCode(const ScheduleState& self, const StmtSRef& block_sref,
                                 const StmtSRef& scope_root_sref) {
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  // Cond 1. The block has the `init` statement.
  if (!block->init.defined()) {
    return 1;
  }
  // Cond 2. All the block bindings are quasi-affine expressions.
  if (!self->IsAffineBlockBinding(block_sref)) {
    return 2;
  }
  // Cond 3. All block vars are either data parallel block vars or reduction block vars. Meanwhile,
  // we collect all the reduction block vars.
  if (!ContainsOnlyDataParAndReductionBlockIter(block->iter_vars)) {
    return 3;
  }
  // Cond 4. Dominant: the block is the only writer of its output, dominating the reader of its
  // output buffers.
  if (!IsDominantBlock(self, scope_root_sref, block_sref)) {
    return 4;
  }
  // Cond 5. The reduction block vars are not used to index the output buffers.
  return ReductionIterNotIndexOutputBuffer(GetRef<Block>(block)) ? 0 : 5;
}

bool IsReductionBlock(const ScheduleState& self, const StmtSRef& block_sref,
                      const StmtSRef& scope_root_sref) {
  return CheckReductionBlockErrorCode(self, block_sref, scope_root_sref) == 0;
}

void CheckReductionBlock(const ScheduleState& self, const StmtSRef& block_sref,
                         const StmtSRef& scope_root_sref) {
  class NotReductionBlockError : public ScheduleError {
   public:
    explicit NotReductionBlockError(IRModule mod, Block block, int violated_cond)
        : mod_(std::move(mod)), block_(std::move(block)), violated_cond_(violated_cond) {}
    String FastErrorString() const final { return "ScheduleError: Not a reduction block"; }
    String DetailRenderTemplate() const final {
      std::ostringstream os;
      os << "The block {0} is not a reduction block - it violates condition #" << violated_cond_;
      os << ".\n" << kReductionBlockDefinition;
      return os.str();
    }
    IRModule mod() const final { return mod_; }
    Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }
    IRModule mod_;
    Block block_;
    int violated_cond_;
  };

  int error_code = CheckReductionBlockErrorCode(self, block_sref, scope_root_sref);
  if (error_code != 0) {
    const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
    throw NotReductionBlockError(self->mod, GetRef<Block>(block), error_code);
  }
}

void CheckCompleteOrReductionBlock(const ScheduleState& self, const StmtSRef& block_sref,
                                   const StmtSRef& scope_root_sref) {
  class NotCompleteOrReductionBlockError : public ScheduleError {
   public:
    explicit NotCompleteOrReductionBlockError(IRModule mod, Block block,
                                              int complete_block_error_code,
                                              int reduction_block_error_code)
        : mod_(mod),
          block_(block),
          complete_block_error_code_(complete_block_error_code),
          reduction_block_error_code_(reduction_block_error_code) {}

    String FastErrorString() const final {
      return "ScheduleError: Not a complete or reduction block";
    }
    String DetailRenderTemplate() const final {
      std::ostringstream os;
      os << "The block {0} is not a complete block - it violates condition #"
         << complete_block_error_code_;
      os << ".\n" << kCompleteBlockDefinition;
      os << "\nThe block is not a reduction block either - it violates condition #"
         << reduction_block_error_code_;
      os << ".\n" << kReductionBlockDefinition;
      return os.str();
    }
    IRModule mod() const final { return mod_; }
    Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }

    IRModule mod_;
    Block block_;
    int complete_block_error_code_;
    int reduction_block_error_code_;
  };

  int complete_block_error_code = CheckCompleteBlockErrorCode(self, block_sref, scope_root_sref);
  if (complete_block_error_code == 0) {
    return;
  }
  int reduction_block_error_code = CheckReductionBlockErrorCode(self, block_sref, scope_root_sref);
  if (reduction_block_error_code == 0) {
    return;
  }
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  throw NotCompleteOrReductionBlockError(self->mod, GetRef<Block>(block), complete_block_error_code,
                                         reduction_block_error_code);
}

void CheckSubtreeCompactDataflow(const ScheduleState& self, const StmtSRef& subtree_root) {
  class NotCompactDataFlowError : public ScheduleError {
   public:
    explicit NotCompactDataFlowError(IRModule mod, Stmt subtree_root, Block violate_block,
                                     int local_complete_block_code, int local_reduction_block_code)
        : mod_(std::move(mod)),
          subtree_root_(std::move(subtree_root)),
          violate_block_(std::move(violate_block)),
          local_complete_block_code_(local_complete_block_code),
          local_reduction_block_code_(local_reduction_block_code) {
      ICHECK(subtree_root_->IsInstance<BlockNode>() || subtree_root_->IsInstance<ForNode>());
    }
    String FastErrorString() const final {
      return "ScheduleError: The queried subtree root in SRef tree does not have compact dataflow, "
             "because some of its child block on SRef tree is neither a local complete block nor a "
             "local reduction block.";
    }
    String DetailRenderTemplate() const final {
      std::ostringstream os;
      os << "The queried subtree root {0} in SRef tree does not have compact dataflow, because "
            "its child block {1} on SRef tree is neither a local complete block nor a local "
            "reduction block.\n";
      os << "It violates condition #" << local_complete_block_code_
         << " as a local complete block.\n";
      os << kLocalCompleteBlockDefinition << "\n";
      os << "It violates condition #" << local_reduction_block_code_
         << " as a local reduction block.\n";
      os << kLocalReductionBlockDefinition << "\n";
      return os.str();
    }
    IRModule mod() const final { return mod_; }
    Array<ObjectRef> LocationsOfInterest() const final { return {subtree_root_, violate_block_}; }

    IRModule mod_;
    Stmt subtree_root_;
    Block violate_block_;
    int local_complete_block_code_;
    int local_reduction_block_code_;
  };

  Array<StmtSRef> child_block_srefs = GetChildBlockSRefOnSRefTree(self, subtree_root);
  for (const StmtSRef& block_sref : child_block_srefs) {
    int local_complete_block_code = CheckCompleteBlockErrorCode(self, block_sref, subtree_root),
        local_reduction_block_code = CheckReductionBlockErrorCode(self, block_sref, subtree_root);
    if (local_complete_block_code != 0 && local_reduction_block_code != 0) {
      const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
      throw NotCompactDataFlowError(self->mod, GetRef<Stmt>(subtree_root->stmt),
                                    GetRef<Block>(block), local_complete_block_code,
                                    local_reduction_block_code);
    }
  }
}

bool IsOutputBlock(const ScheduleState& self, const StmtSRef& block_sref,
                   const StmtSRef& scope_root_sref) {
  const BlockNode* scope_root = TVM_SREF_TO_BLOCK(scope_root, scope_root_sref);
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  std::unordered_set<const BufferNode*> scope_allocated;
  scope_allocated.reserve(scope_root->alloc_buffers.size());
  for (const Buffer& buffer : scope_root->alloc_buffers) {
    scope_allocated.insert(buffer.get());
  }
  for (const BufferRegion& buffer_region : block->writes) {
    if (!scope_allocated.count(buffer_region->buffer.get())) {
      return true;
    }
  }
  return false;
}

void CheckNotOutputBlock(const ScheduleState& self, const StmtSRef& block_sref,
                         const StmtSRef& scope_root_sref) {
  class OutputBlockError : public ScheduleError {
   public:
    explicit OutputBlockError(IRModule mod, Block block) : mod_(mod), block_(block) {}
    String FastErrorString() const final {
      return "ScheduleError: Cannot operate on an output block";
    }
    String DetailRenderTemplate() const final { return "The block {0} is an output block"; }
    IRModule mod() const final { return mod_; }
    Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }

    IRModule mod_;
    Block block_;
  };
  if (IsOutputBlock(self, block_sref, scope_root_sref)) {
    const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
    throw OutputBlockError(self->mod, GetRef<Block>(block));
  }
}

std::vector<IterVarType> GetBlockVarTypes(const BlockNode* block) {
  std::vector<IterVarType> results;
  results.reserve(block->iter_vars.size());
  for (const IterVar& iter_var : block->iter_vars) {
    results.push_back(iter_var->iter_type);
  }
  return results;
}

std::vector<IterVarType> GetBlockVarTypes(const StmtSRef& block_sref) {
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  return GetBlockVarTypes(block);
}

bool IsWriteCache(const StmtSRef& block_sref) {
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  if (block->writes.size() != 1) {
    return false;
  }
  const BufferRegion& write_region = block->writes[0];
  for (const BufferRegion& read_region : block->reads) {
    bool exists, surjective, injective, ordered, no_const_read, no_shift_read;
    std::tie(exists, surjective, injective, ordered, no_const_read, no_shift_read) =
        AnalyzeReadWritePattern(read_region, write_region);
    if (!(injective && ordered)) {
      return false;
    }
  }
  return true;
}

/******** Binding ********/

bool IsAffineBinding(const BlockRealize& realize, const Map<Var, Range>& loop_var_ranges,
                     arith::Analyzer* analyzer) {
  if (loop_var_ranges.empty()) {
    return true;
  }
  auto res = arith::DetectIterMap(
      /*indices=*/realize->iter_values,
      /*input_iters=*/loop_var_ranges,
      /*predicate=*/realize->predicate,
      /*check_level=*/arith::IterMapLevel::Surjective,
      /*analyzer=*/analyzer);
  if (res->indices.empty()) {
    return false;
  }
  for (const arith::IterSumExpr& sum_expr : res->indices) {
    const Array<arith::IterSplitExpr>& args = sum_expr->args;
    if (!args.empty() && !is_one(args[0]->scale)) {
      return false;
    }
  }
  return true;
}

void CheckPartialAffineBinding(const ScheduleState& self, Block block,
                               const Optional<StmtSRef>& high_exclusive) {
  class NotAffineBindingError : public ScheduleError {
   public:
    explicit NotAffineBindingError(IRModule mod, Block block, Optional<StmtSRef> high_exclusive)
        : mod_(std::move(mod)), block_(std::move(block)) {
      if (high_exclusive.defined()) {
        high_exclusive_loop_ = high_exclusive.value()->StmtAs<ForNode>();
      }
    }
    String FastErrorString() const final {
      std::ostringstream ss;
      if (high_exclusive_loop_) {
        ss << "ScheduleError: The block is required to have an partial affine binding under "
           << high_exclusive_loop_->loop_var;
      } else {
        ss << "ScheduleError: The block is required to have an affine binding";
      }
      return ss.str();
    }
    String DetailRenderTemplate() const final {
      std::ostringstream ss;
      if (high_exclusive_loop_) {
        ss << "The block {0} is required to have an partial affine binding under "
           << high_exclusive_loop_->loop_var;
      } else {
        ss << "The block {0} is required to have an affine binding";
      }
      return ss.str();
    }
    IRModule mod() const final { return mod_; }
    Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }
    IRModule mod_;
    Block block_;
    const ForNode* high_exclusive_loop_{nullptr};
  };

  StmtSRef block_sref = self->stmt2ref.at(block.get());
  if (self->IsAffineBlockBinding(block_sref)) {
    // check block cached state for global affineness
    return;
  }
  if (block_sref->parent && high_exclusive.defined()) {
    // if it is not of global affine binding, check affineness under high_exclusive,
    arith::Analyzer analyzer;
    Map<Var, Range> dom_map =
        LoopDomainOfSRefTreePath(GetRef<StmtSRef>(block_sref->parent), high_exclusive);
    if (IsAffineBinding(GetBlockRealize(self, block_sref), dom_map, &analyzer)) {
      return;
    }
  }
  throw NotAffineBindingError(self->mod, std::move(block), high_exclusive);
}

void CheckAffineBinding(const ScheduleState& self, Block block) {
  CheckPartialAffineBinding(self, std::move(block), NullOpt);
}

Map<Var, Range> LoopDomainOfSRefTreePath(const StmtSRef& low_inclusive,
                                         const Optional<StmtSRef>& high_exclusive,
                                         const runtime::StorageScope& extra_relax_scope) {
  Map<Var, Range> result;
  const StmtSRefNode* p = low_inclusive.get();
  const StmtSRefNode* limit = static_cast<const StmtSRefNode*>(high_exclusive.get());
  for (; p != limit; p = p->parent) {
    const ForNode* loop = p->StmtAs<ForNode>();
    if (loop == nullptr) {
      break;
    }
    result.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
  }
  if (extra_relax_scope.rank != runtime::StorageRank::kGlobal) {
    for (; p; p = p->parent) {
      if (const ForNode* loop = p->StmtAs<ForNode>()) {
        if (loop->kind == ForKind::kThreadBinding) {
          const String& thread_tag = loop->thread_binding.value()->thread_tag;
          if (CanRelaxStorageUnderThread(extra_relax_scope,
                                         runtime::ThreadScope::Create(thread_tag))) {
            result.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
          }
        }
      }
    }
  }
  return result;
}

Map<Var, PrimExpr> GetBindings(const BlockRealize& realize) {
  const BlockNode* block = realize->block.get();
  const Array<IterVar>& all_lhs = block->iter_vars;
  const Array<PrimExpr>& all_rhs = realize->iter_values;
  ICHECK_EQ(all_lhs.size(), all_rhs.size());
  Map<Var, PrimExpr> result;
  for (int i = 0, n = all_lhs.size(); i < n; ++i) {
    const IterVar& lhs = all_lhs[i];
    const PrimExpr& rhs = all_rhs[i];
    result.Set(lhs->var, rhs);
  }
  return result;
}

bool GetVarsTouchedByBlockIters(const BlockRealize& block_realize,
                                std::unordered_set<const VarNode*>* data_par_vars,
                                std::unordered_set<const VarNode*>* reduce_vars) {
  Block block = block_realize->block;
  ICHECK(block_realize->block.same_as(block))
      << "ValueError: The input `block_realize` is required to be the exact BlockRealize of the "
         "input block";

  bool has_block_vars_of_other_types = false;
  ICHECK_EQ(block->iter_vars.size(), block_realize->iter_values.size());
  int n = static_cast<int>(block->iter_vars.size());
  for (int i = 0; i < n; ++i) {
    const IterVar& iter_var = block->iter_vars[i];
    const PrimExpr& iter_value = block_realize->iter_values[i];
    std::unordered_set<const VarNode*>* set = nullptr;
    if (iter_var->iter_type == IterVarType::kDataPar) {
      set = data_par_vars;
    } else if (iter_var->iter_type == IterVarType::kCommReduce) {
      set = reduce_vars;
    } else {
      has_block_vars_of_other_types = true;
    }
    if (set == nullptr) {
      continue;
    }
    Array<Var> vars_in_binding = UndefinedVars(iter_value);
    for (const Var& var : vars_in_binding) {
      set->insert(var.get());
    }
  }

  return has_block_vars_of_other_types;
}

/******** Loop properties ********/

void CheckLoopStartsWithZero(const ScheduleState& self, const StmtSRef& loop_sref,
                             arith::Analyzer* analyzer) {
  class LoopNotStartWithZeroError : public ScheduleError {
   public:
    explicit LoopNotStartWithZeroError(IRModule mod, For loop)
        : mod_(mod), loop_(std::move(loop)) {}

    String FastErrorString() const final {
      return "ScheduleError: The primitive only supports loop starting with 0";
    }

    String DetailRenderTemplate() const final {
      return "The loop {0} does not start with 0, which is not supported";
    }

    IRModule mod() const final { return mod_; }
    Array<ObjectRef> LocationsOfInterest() const final { return {loop_}; }

    IRModule mod_;
    For loop_;
  };
  const ForNode* loop = TVM_SREF_TO_FOR(loop, loop_sref);
  if (!analyzer->CanProve(loop->min == 0)) {
    throw LoopNotStartWithZeroError(self->mod, GetRef<For>(loop));
  }
}

/******** Block-loop relation ********/

Array<StmtSRef> GetChildBlockSRefOnSRefTree(const ScheduleState& self,
                                            const StmtSRef& parent_sref) {
  Array<BlockRealize> child_block_realize = GetChildBlockRealizeOnSRefTree(parent_sref);
  Array<StmtSRef> child_block_srefs;
  child_block_srefs.reserve(child_block_realize.size());

  for (BlockRealize realize : child_block_realize) {
    child_block_srefs.push_back(self->stmt2ref.at(realize->block.get()));
  }
  return child_block_srefs;
}

Array<BlockRealize> GetChildBlockRealizeOnSRefTree(const StmtSRef& parent_sref) {
  struct Collector : public StmtVisitor {
    static Array<BlockRealize> Collect(const Stmt& stmt) {
      Collector collector;
      collector(stmt);
      return std::move(collector.result_);
    }

    void VisitStmt_(const BlockRealizeNode* block_realize) final {
      result_.push_back(GetRef<BlockRealize>(block_realize));
    }

    Array<BlockRealize> result_;
  };

  if (parent_sref->stmt->IsInstance<ForNode>()) {
    const auto* loop = static_cast<const ForNode*>(parent_sref->stmt);
    return Collector::Collect(loop->body);
  } else if (parent_sref->stmt->IsInstance<BlockNode>()) {
    const auto* block = static_cast<const BlockNode*>(parent_sref->stmt);
    return Collector::Collect(block->body);
  }
  ICHECK(false) << "Unreachable";
  throw;
}

BlockRealize CheckGetSingleChildBlockRealizeOnSRefTree(const ScheduleState& self,
                                                       const StmtSRef& parent_sref) {
  class NonSingleChildBlockError : public ScheduleError {
   public:
    explicit NonSingleChildBlockError(IRModule mod, const StmtSRef& sref)
        : mod_(std::move(mod)), stmt_(GetRef<Stmt>(sref->stmt)) {
      sref_type_ = stmt_.as<BlockNode>() != nullptr ? "block" : "loop";
    }

    String FastErrorString() const final {
      std::ostringstream os;
      os << "ScheduleError: The " << sref_type_ << " is required to have only one child block";
      return os.str();
    }

    String DetailRenderTemplate() const final {
      std::ostringstream os;
      os << "The " << sref_type_ << " {0} is required to have only one child block";
      return os.str();
    }

    IRModule mod() const final { return mod_; }
    Array<ObjectRef> LocationsOfInterest() const final { return {stmt_}; }

    IRModule mod_;
    Stmt stmt_;
    String sref_type_;
  };

  Array<BlockRealize> child_block_realize = GetChildBlockRealizeOnSRefTree(parent_sref);
  if (child_block_realize.size() != 1) {
    throw NonSingleChildBlockError(self->mod, parent_sref);
  }
  return child_block_realize[0];
}

BlockRealize GetBlockRealize(const ScheduleState& self, const StmtSRef& block_sref) {
  struct BlockRealizeFinder : public StmtVisitor {
    explicit BlockRealizeFinder(const BlockNode* target_block)
        : target_block(target_block), result(nullptr) {}

    void VisitStmt(const Stmt& stmt) final {
      if (result != nullptr) {
        return;
      }
      StmtVisitor::VisitStmt(stmt);
    }

    void VisitStmt_(const BlockRealizeNode* block_realize) final {
      if (block_realize->block.get() == target_block) {
        result = block_realize;
      }
      // No need to visit recursively, since the deeper BlockRealizes must not be the result.
    }

    const BlockNode* target_block;
    const BlockRealizeNode* result;
  };

  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  if (block_sref->parent == nullptr) {
    const PrimFuncNode* func = GetRootPrimFunc(self->mod, block, nullptr);
    return Downcast<BlockRealize>(func->body);
  } else {
    BlockRealizeFinder finder(block);
    finder(GetRef<Stmt>(block_sref->parent->stmt));
    ICHECK(finder.result != nullptr)
        << "InternalError: Cannot find the BlockRealize of block " << GetRef<Block>(block);
    return GetRef<BlockRealize>(finder.result);
  }
}

IterVarType GetLoopIterType(const StmtSRef& loop_sref) {
  const ForNode* loop = TVM_SREF_TO_FOR(loop, loop_sref);
  const Var& loop_var = loop->loop_var;
  int n_spatial = 0;
  int n_reduce = 0;
  int n_other = 0;
  auto f_visit = [&loop_var, &n_spatial, &n_reduce, &n_other](const ObjectRef& obj) -> bool {
    if (const auto* realize = obj.as<BlockRealizeNode>()) {
      const BlockNode* block = realize->block.get();
      // Number of block vars and their bindings
      ICHECK_EQ(realize->iter_values.size(), block->iter_vars.size());
      size_t n = realize->iter_values.size();
      for (size_t i = 0; i < n; ++i) {
        const IterVar& iter_var = block->iter_vars[i];
        const PrimExpr& binding = realize->iter_values[i];
        // Categorize the current block var
        int* ref = nullptr;
        if (iter_var->iter_type == IterVarType::kDataPar) {
          ref = &n_spatial;
        } else if (iter_var->iter_type == IterVarType::kCommReduce) {
          ref = &n_reduce;
        } else {
          ref = &n_other;
        }
        // Visit the binding to see if `loop_var` appears
        PostOrderVisit(binding, [&ref, &loop_var](const ObjectRef& obj) -> void {
          if (obj.same_as(loop_var)) {
            (*ref) += 1;
          }
        });
      }
      return false;
    }
    return true;
  };
  PreOrderVisit(loop->body, f_visit);
  if (n_other) {
    return IterVarType::kOpaque;
  } else if (n_spatial && n_reduce) {
    return IterVarType::kOpaque;
  } else if (n_reduce) {
    return IterVarType::kCommReduce;
  } else {
    return IterVarType::kDataPar;
  }
}

StmtSRef GetSRefLowestCommonAncestor(const Array<StmtSRef>& srefs) {
  CHECK(!srefs.empty()) << "ValueError: The input array is required to have at least one sref";

  std::unordered_map<const StmtSRefNode*, size_t> sref_visited_cnt;
  for (const StmtSRef& sref : srefs) {
    const StmtSRefNode* p = sref.get();
    while (p != nullptr) {
      ++sref_visited_cnt[p];
      p = p->parent;
    }
  }
  size_t n_sref = srefs.size();
  const StmtSRefNode* p = srefs[0].get();
  while (p != nullptr && sref_visited_cnt[p] != n_sref) {
    p = p->parent;
  }
  ICHECK(p != nullptr);
  return GetRef<StmtSRef>(p);
}

bool HasBeenMultiLevelTiled(const StmtSRef& block_sref) {
  return tir::GetAnn<String>(block_sref, tir::attr::meta_schedule_tiling_structure).defined();
}

std::pair<Array<StmtSRef>, std::vector<int>> CollectComputeLocation(const ScheduleState& self,
                                                                    const StmtSRef& block_sref) {
  Array<StmtSRef> location_srefs;
  std::vector<int> location_indices;

  // Step 1. Add the "compute-root" candidate. Add the "compute-inline" candidate if the block can
  // be inlined.
  if (CanComputeInline(self, block_sref)) {
    location_srefs.push_back(StmtSRef::InlineMark());
    location_indices.push_back(-2);
  }
  location_srefs.push_back(StmtSRef::RootMark());
  location_indices.push_back(-1);

  // Step 2. If the block has no consumer, there is no more candidate.
  Array<StmtSRef> consumers = GetConsumers(self, block_sref);
  if (consumers.empty()) {
    return std::make_pair(location_srefs, location_indices);
  }

  // Step 3. Get the deepest loop that the input block can be computed at (namely "boundary"). If
  // such a loop cannot be found, there is no more candidate and we just return.
  StmtSRef loop_boundary = consumers.size() > 1 ? GetSRefLowestCommonAncestor(consumers)
                                                : GetRef<StmtSRef>(consumers[0]->parent);
  if (loop_boundary->StmtAs<ForNode>() == nullptr) {
    return std::make_pair(location_srefs, location_indices);
  }

  // Step 4. Collect the loops outside the first consumer and locate the boundary loop. The position
  // of the boundary loop reveals the number of possible additional candidates.
  Array<StmtSRef> loop_srefs = GetLoops(consumers[0]);
  size_t lca_pos =
      std::find(loop_srefs.begin(), loop_srefs.end(), loop_boundary) - loop_srefs.begin();
  ICHECK_LT(lca_pos, loop_srefs.size());
  size_t n_candidate = lca_pos + 1;

  // Step 5. Find the position of the deepest data-parallel loop among the candidate loops. This
  // position is used for removing the unwanted candidates from the perspective of performance.
  std::vector<IterVarType> loop_iter_types;
  loop_iter_types.reserve(n_candidate);
  int i_last_datapar = -1;
  for (size_t i = 0; i < n_candidate; ++i) {
    // TODO(siyuan): improve the performance
    IterVarType iter_type = GetLoopIterType(loop_srefs[i]);
    loop_iter_types.push_back(iter_type);
    if (iter_type == IterVarType::kDataPar) {
      i_last_datapar = i;
    }
  }
  // Step 6. Check and add the candidates in turn according to the following rules:
  //  - skip the unit loops (loops with extent 1);
  //  - do not consider the data-parallel loops after a not-data-parallel loop;
  //  - do not consider the trailing not-data-parallel loops.
  location_srefs.reserve(n_candidate + 2);
  location_indices.reserve(n_candidate + 2);
  bool visited_reduce = false;
  for (size_t i = 0; i < n_candidate; ++i) {
    const int64_t* loop_extent = GetLoopIntExtent(loop_srefs[i]);
    if (loop_extent != nullptr && *loop_extent == 1) {
      continue;
    }

    if (loop_iter_types[i] == IterVarType::kDataPar) {
      if (visited_reduce) {
        break;
      }
    } else {
      visited_reduce = true;
      if (static_cast<int>(i) > i_last_datapar) {
        break;
      }
    }
    if (CanComputeAt(self, block_sref, loop_srefs[i], true)) {
      location_srefs.push_back(loop_srefs[i]);
      location_indices.push_back(i);
    }
  }

  return std::make_pair(location_srefs, location_indices);
}

/******** Auto Tensorization ********/

class AutoTensorizeExtractor : public TensorizeComparator {
 public:
  explicit AutoTensorizeExtractor() : TensorizeComparator(IRModule(), false) {}

 private:
  bool VisitExprDefault_(const Object* op, const PrimExpr& other) override;
  bool VisitStmtDefault_(const Object* op, const Stmt& other) override;

  bool VisitStmt_(const BlockNode* op, const Stmt& other) override;
  bool VisitStmt_(const BufferStoreNode* op, const Stmt& other) override;

  bool VisitExpr_(const BufferLoadNode* op, const PrimExpr& other) override;

  bool CompareBuffer(const Buffer& lhs, const Buffer& rhs) override;
  template <typename T>
  bool CompareBufferAccess(const T* lhs, const T* rhs);
  template <typename T, typename F>
  bool CompareArray(const Array<T>& lhs, const Array<T>& rhs, F cmp);

 public:
  std::vector<IterVar> lhs_iters_, rhs_iters_;
  std::unordered_map<Buffer, Array<PrimExpr>, ObjectPtrHash, ObjectPtrEqual>
      lhs_buffer_indices_map_;
  std::unordered_map<Buffer, Array<PrimExpr>, ObjectPtrHash, ObjectPtrEqual>
      rhs_buffer_indices_map_;
};

bool AutoTensorizeExtractor::VisitExprDefault_(const Object* op, const PrimExpr& other) {
  return false;
}

bool AutoTensorizeExtractor::VisitStmtDefault_(const Object* op, const Stmt& other) {
  return false;
}

bool AutoTensorizeExtractor::VisitStmt_(const BlockNode* op, const Stmt& other) {
  const auto* rhs = other.as<BlockNode>();
  // Check block equality.
  // All iter vars and buffer regions including the order should match.
  // When checking iter vars, DefEqual is used to remap variables.
  if (!is_scope_block) {
    if (!CompareArray(op->iter_vars, rhs->iter_vars, &AutoTensorizeExtractor::CompareIterVar)) {
      return false;
    }
    if (!CompareAnnotationMap(op->annotations, rhs->annotations)) {
      return false;
    }
    if (!CompareArray(op->alloc_buffers, rhs->alloc_buffers,
                      &AutoTensorizeExtractor::CompareBuffer)) {
      return false;
    }
    for (const IterVar& block_iter : op->iter_vars) {
      inner_iter_dom_map_.Set(block_iter->var, arith::IntSet::FromRange(block_iter->dom));
    }
  } else {
    auto iter_collect = [&](const BlockNode* op, std::vector<IterVar>& iters) -> bool {
      for (const auto& iter : op->iter_vars) {
        analyzer_.Bind(iter->var, iter->dom);
        if (iter->iter_type == IterVarType::kDataPar ||
            iter->iter_type == IterVarType::kCommReduce) {
          iters.push_back(iter);
        } else {
          return false;
        }
      }
      return true;
    };
    if (!iter_collect(op, lhs_iters_)) {
      return false;
    }
    if (!iter_collect(rhs, rhs_iters_)) {
      return false;
    }
  }
  is_scope_block = false;
  return VisitStmt(op->body, rhs->body);
}

bool AutoTensorizeExtractor::CompareBuffer(const Buffer& lhs, const Buffer& rhs) {
  if (lhs.same_as(rhs)) return true;
  auto it = rhs_buffer_map_.find(rhs);
  bool equal;
  if (it != rhs_buffer_map_.end()) {
    equal = (*it).second.same_as(lhs);
  } else {
    // Remap both buffer itself and buffer data, skip buffer shape
    equal = DefEqual(lhs->data, rhs->data) && lhs->dtype == rhs->dtype;
    if (equal) {
      rhs_buffer_map_[rhs] = lhs;
    }
  }
  return equal;
}

bool AutoTensorizeExtractor::VisitStmt_(const BufferStoreNode* op, const Stmt& other) {
  const auto* rhs = other.as<BufferStoreNode>();
  return AutoTensorizeExtractor::CompareBufferAccess(op, rhs) && VisitExpr(op->value, rhs->value);
}

bool AutoTensorizeExtractor::VisitExpr_(const BufferLoadNode* op, const PrimExpr& other) {
  const auto* rhs = other.as<BufferLoadNode>();
  return AutoTensorizeExtractor::CompareBufferAccess(op, rhs);
}

template <typename T>
bool AutoTensorizeExtractor::CompareBufferAccess(const T* lhs, const T* rhs) {
  if (!CompareBuffer(lhs->buffer, rhs->buffer)) return false;
  auto it_lhs = lhs_buffer_indices_map_.find(lhs->buffer);
  if (it_lhs == lhs_buffer_indices_map_.end()) {
    if (rhs_buffer_indices_map_.find(rhs->buffer) != rhs_buffer_indices_map_.end()) {
      return false;
    }
    std::vector<PrimExpr> lhs_indices;
    for (const auto& index : lhs->indices) {
      lhs_indices.push_back(analyzer_.Simplify(index));
    }
    for (const auto& index : rhs->indices) {
      if (!index.template as<VarNode>()) return false;
    }
    lhs_buffer_indices_map_[lhs->buffer] = lhs_indices;
    rhs_buffer_indices_map_[rhs->buffer] = rhs->indices;
  } else {
    auto it_rhs = rhs_buffer_indices_map_.find(rhs->buffer);
    if (it_rhs == rhs_buffer_indices_map_.end()) {
      return false;
    }
    auto indices_check = [&](const Array<PrimExpr>& indices,
                             const Array<PrimExpr>& old_indices) -> bool {
      if (indices.size() != old_indices.size()) {
        return false;
      }
      for (size_t i = 0; i < indices.size(); ++i) {
        if (!analyzer_.CanProveEqual(indices[i], old_indices[i])) {
          return false;
        }
      }
      return true;
    };
    if (!indices_check(lhs->indices, it_lhs->second)) return false;
    if (!indices_check(rhs->indices, it_rhs->second)) return false;
  }
  return true;
}

template <typename T, typename F>
bool AutoTensorizeExtractor::CompareArray(const Array<T>& lhs, const Array<T>& rhs, F cmp) {
  if (lhs.same_as(rhs)) return true;
  if (lhs.size() != rhs.size()) return false;
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (!(this->*cmp)(lhs[i], rhs[i])) return false;
  }
  return true;
}

class MappingProposer {
 public:
  explicit MappingProposer(AutoTensorizeExtractor* extractor) : extractor_(extractor) {}

  void Propose() {
    CollectFeasibleSet();
    ProposeAllFuseMapping();
  }

 private:
  using VarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;

  void SetIntersection(VarSet* var_set, const VarSet& other) {
    VarSet to_be_removed;
    for (const Var& var : (*var_set)) {
      if (other.find(var) == other.end()) to_be_removed.insert(var);
    }
    for (const Var& var : to_be_removed) {
      var_set->erase(var);
    }
  }

  void SetSubtraction(VarSet* var_set, const VarSet& other) {
    for (const Var& var : other) {
      var_set->erase(var);
    }
  }

  void CollectFeasibleSet() {
    std::unordered_map<Buffer, VarSet, ObjectPtrHash, ObjectPtrEqual> 
      lhs_buffer_var_map, rhs_buffer_var_map;
    for (const auto& it : extractor_->rhs_buffer_indices_map_) {
      VarSet lhs_vars, rhs_vars;
      auto lhs_buffer_it = extractor_->rhs_buffer_map_.find(it.first);
      ICHECK(lhs_buffer_it != extractor_->rhs_buffer_map_.end());
      for (const PrimExpr& index : extractor_->lhs_buffer_indices_map_[lhs_buffer_it->second]) {
        PreOrderVisit(index, [&](const ObjectRef& obj) -> bool {
          if (const VarNode* var = obj.as<VarNode>()) {
            lhs_vars.insert(GetRef<Var>(var));
          }
          return true;
        });
      }
      for (const PrimExpr& rhs_index : it.second) {
        if (const VarNode* var_ptr = rhs_index.as<VarNode>()) {
          Var var = GetRef<Var>(var_ptr);
          rhs_vars.insert(var);
          if (rhs_feasible_vars_.find(var) == rhs_feasible_vars_.end()) {
            rhs_feasible_vars_[var] = lhs_vars;
          } else {
            SetIntersection(&rhs_feasible_vars_[var], lhs_vars);
          }
        } else {
          LOG(FATAL);
        }
      }
      lhs_buffer_var_map[lhs_buffer_it->second] = std::move(lhs_vars);
      rhs_buffer_var_map[lhs_buffer_it->second] = std::move(rhs_vars);
    }
    for (const auto& it : extractor_->rhs_buffer_indices_map_) {
      auto lhs_buffer_it = extractor_->rhs_buffer_map_.find(it.first);
      ICHECK(lhs_buffer_it != extractor_->rhs_buffer_map_.end());
      const VarSet& lhs_vars = lhs_buffer_var_map[lhs_buffer_it->second];
      const VarSet& rhs_vars = rhs_buffer_var_map[lhs_buffer_it->second];
      for (const IterVar& iter : extractor_->rhs_iters_) {
        if (rhs_vars.find(iter->var) == rhs_vars.end()) {
          SetSubtraction(&rhs_feasible_vars_[iter->var], lhs_vars);
        }
      }
    }
    for (const auto& it : rhs_feasible_vars_) {
      for (const auto& lhs_var : it.second) {
        if (lhs_feasible_vars_.find(lhs_var) == lhs_feasible_vars_.end()) {
          lhs_feasible_vars_[lhs_var] = std::move(VarSet());
        }
        lhs_feasible_vars_[lhs_var].insert(it.first);
      }
    }
    for (const auto& iter : extractor_->lhs_iters_) {
      lhs_representers.push_back(iter->var.copy_with_suffix("_l"));
    }
    for (const auto& it : extractor_->rhs_buffer_map_) {
      lhs_buffer_map_[it.second] = it.first;
    }
  }

  void ProposeAllFuseMapping() {
    std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual> fused_iter, extent_map;
    for (const auto& iter : extractor_->rhs_iters_) {
      fused_iter[iter->var] = 0;
    }
    for (const auto& iter : extractor_->lhs_iters_) {
      extent_map[iter->var] = iter->dom->extent;
    }
    std::vector<PrimExpr> tgt_iters;
    for (size_t i = 0; i < extractor_->lhs_iters_.size(); ++i) {
      const VarSet& feasible = lhs_feasible_vars_[extractor_->lhs_iters_[i]->var];
      if (feasible.empty()) {
        tgt_iters.push_back(lhs_representers[i]);
      } else if (feasible.size() == 1) {
        Var rhs_var = *feasible.begin();
        fused_iter[rhs_var] =
            fused_iter[rhs_var] * extent_map[extractor_->lhs_iters_[i]->var] + lhs_representers[i];
      } else {
        // give up this proposal
        return;
      }
    }
    arith::Analyzer analyzer;
    for (const auto& iter : extractor_->rhs_iters_) {
      tgt_iters.push_back(analyzer.Simplify(fused_iter[iter->var]));
    }
    mappings_.emplace_back(lhs_representers, tgt_iters);
  }

 public:
  std::vector<Var> lhs_representers;
  std::unordered_map<Buffer, Buffer, ObjectHash, ObjectEqual> lhs_buffer_map_;
  std::unordered_map<Var, VarSet, ObjectPtrHash, ObjectPtrEqual> rhs_feasible_vars_;
  std::unordered_map<Var, VarSet, ObjectPtrHash, ObjectPtrEqual> lhs_feasible_vars_;
  std::vector<IndexMap> mappings_;
  AutoTensorizeExtractor* extractor_;
};

Optional<LayoutInfo> GetTensorizeLayoutInfo(const tir::ScheduleState& self,
                                            const tir::StmtSRef& block_sref,
                                            const tir::PrimFunc& desc_func) {
  arith::Analyzer analyzer;
  const tir::BlockRealize& block = tir::GetBlockRealize(self, block_sref);
  // Step 1. Analyze desc_func, extract its block, loops and loop vars
  const tir::BlockRealizeNode* desc_block = nullptr;
  std::vector<const tir::ForNode*> desc_loops;
  std::unordered_set<const tir::VarNode*> desc_loop_vars;
  const auto* desc_scope_realize = desc_func->body.as<tir::BlockRealizeNode>();
  ICHECK(desc_scope_realize);
  {
    auto f_visit = [&desc_block, &desc_loops, &desc_loop_vars,
                    &analyzer](const ObjectRef& obj) -> bool {
      // Extract the block
      if (const auto* block = obj.as<tir::BlockRealizeNode>()) {
        desc_block = block;
        return false;
      }
      // Extract the loops
      if (const auto* loop = obj.as<tir::ForNode>()) {
        desc_loops.push_back(loop);
        desc_loop_vars.insert(loop->loop_var.get());
        if (!analyzer.CanProve(loop->min == 0)) {
          return false;
        }
      }
      return true;
    };
    tir::PostOrderVisit(desc_scope_realize->block->body, f_visit);
    std::reverse(desc_loops.begin(), desc_loops.end());
    ICHECK(desc_block);
  }
  // Step 2. Check if `desc_block` matches `block`
  // Ignore the scope of buffers when comparing, since we can do cache_read/write
  const StmtSRef& scope_sref = GetScopeRoot(self, block_sref, false);
  const tir::BlockNode* scope_block = TVM_SREF_TO_BLOCK(scope_block, scope_sref);
  AutoTensorizeExtractor extractor;
  MappingProposer proposer(&extractor);
  if (!extractor.VisitStmt(block->block, desc_block->block)) {
    LOG(INFO)<<"fail extractor";
    return NullOpt;
  }
  proposer.Propose();
  if (proposer.mappings_.empty()) {
    LOG(INFO)<<"fail prposer";
    return NullOpt;
  }
  ObjectPtr<LayoutInfoNode> ret = make_object<LayoutInfoNode>();
  // Only using 1 layout now
  ret->mapping = std::move(proposer.mappings_[0]);
  ret->lhs_buffer_map = std::move(proposer.lhs_buffer_map_);
  ret->rhs_indices_map = std::move(extractor.rhs_buffer_indices_map_);
  ret->lhs_iters = std::move(extractor.lhs_iters_);
  ret->rhs_iters = std::move(extractor.rhs_iters_);
  return LayoutInfo(ret);
}

TVM_REGISTER_NODE_TYPE(LayoutInfoNode);

/******** Producer-consumer relation ********/

Array<StmtSRef> GetProducers(const StmtSRef& block_sref, const BlockScope& scope) {
  Array<Dependency> deps = scope->GetDepsByDst(block_sref);
  Array<StmtSRef> result;
  result.reserve(deps.size());
  for (const Dependency& dep : deps) {
    result.push_back(dep->src);
  }
  return result;
}

Array<StmtSRef> GetConsumers(const StmtSRef& block_sref, const BlockScope& scope) {
  Array<Dependency> deps = scope->GetDepsBySrc(block_sref);
  Array<StmtSRef> result;
  result.reserve(deps.size());
  for (const Dependency& dep : deps) {
    result.push_back(dep->dst);
  }
  return result;
}

ProducerConsumerSplit ProducerConsumerSplit::Find(
    const ScheduleState& self, const Array<Stmt>& subtrees,
    const Array<StmtSRef>& producer_block_srefs, const Array<StmtSRef>& consumer_block_srefs,
    std::unordered_map<const BlockNode*, const BlockRealizeNode*>* block2realize) {
  class InsertionPointNotFoundError : public ScheduleError {
   public:
    explicit InsertionPointNotFoundError(IRModule mod, int last_producer_position,
                                         int first_consumer_position)
        : mod_(mod),
          last_producer_position_(last_producer_position),
          first_consumer_position_(first_consumer_position) {}

    String FastErrorString() const final {
      return "ScheduleError: Cannot find the insertion point that satisfies the producer-consumer "
             "constraint";
    }

    String DetailRenderTemplate() const final {
      return "Cannot find the insertion point that satisfies the producer-consumer constraint. In "
             "0-based indexing, the last producer appears in subtree " +
             std::to_string(last_producer_position_) +
             ", and the first consumer appears in subtree " +
             std::to_string(first_consumer_position_);
    }

    IRModule mod() const final { return mod_; }

    Array<ObjectRef> LocationsOfInterest() const final { return {}; }

   private:
    IRModule mod_;
    int last_producer_position_;
    int first_consumer_position_;
  };

  class Finder : public StmtVisitor {
   public:
    void VisitStmt_(const BlockRealizeNode* realize) final {
      const BlockNode* block = realize->block.get();
      if (block2realize_) {
        block2realize_->emplace(block, realize);
      }
      if (producer_blocks_.count(block)) {
        ++this->n_producers_visited_;
      }
      if (consumer_blocks_.count(block)) {
        ++this->n_consumers_visited_;
      }
    }

    std::unordered_map<const BlockNode*, const BlockRealizeNode*>* block2realize_;
    std::unordered_set<const StmtNode*> producer_blocks_;
    std::unordered_set<const StmtNode*> consumer_blocks_;
    int n_producers_visited_ = 0;
    int n_consumers_visited_ = 0;
  };

  Finder finder;
  finder.block2realize_ = block2realize;
  // Set up the lookup table for producers
  finder.producer_blocks_.reserve(producer_block_srefs.size());
  for (const StmtSRef& block_sref : producer_block_srefs) {
    finder.producer_blocks_.insert(block_sref->stmt);
  }
  // Set up the lookup table for consumers
  finder.consumer_blocks_.reserve(consumer_block_srefs.size());
  for (const StmtSRef& block_sref : consumer_block_srefs) {
    finder.consumer_blocks_.insert(block_sref->stmt);
  }
  // Visit the subtrees
  int n = subtrees.size();
  int last_producer_position = -1;
  int first_consumer_position = n;
  for (int i = 0; i < n; ++i) {
    int n_producers_visited_before = finder.n_producers_visited_;
    int n_consumers_visited_before = finder.n_consumers_visited_;
    finder(subtrees[i]);
    // Check if the subtree contains at least a producer
    if (finder.n_producers_visited_ != n_producers_visited_before) {
      last_producer_position = i;
    }
    // Check if the subtree contains at least a consumer
    if (finder.n_consumers_visited_ != n_consumers_visited_before) {
      if (first_consumer_position == n) {
        first_consumer_position = i;
      }
    }
  }
  if (last_producer_position >= first_consumer_position) {
    throw InsertionPointNotFoundError(self->mod, last_producer_position, first_consumer_position);
  }
  return ProducerConsumerSplit{last_producer_position,       //
                               first_consumer_position,      //
                               finder.n_producers_visited_,  //
                               finder.n_consumers_visited_};
}

/******** Block-buffer relation ********/

Buffer GetNthAccessBuffer(const ScheduleState& self, const Block& block, int n, bool is_write) {
  class BufferIndexOutOfRangeError : public ScheduleError {
   public:
    explicit BufferIndexOutOfRangeError(IRModule mod, Block block, int buffer_index, bool is_write)
        : mod_(std::move(mod)),
          block_(std::move(block)),
          buffer_index_(buffer_index),
          is_write_(is_write) {}

    String FastErrorString() const final {
      if (is_write_) {
        return "ScheduleError: The input `buffer_index` is out of range. It is required to be in "
               "range "
               "[0, num_write_regions) where `num_write_regions` is the number of buffer regions "
               "written by the block.";
      } else {
        return "ScheduleError: The input `buffer_index` is out of range. It is required to be in "
               "range "
               "[0, num_read_regions) where `num_read_regions` is the number of buffer regions "
               "read by the block.";
      }
    }

    String DetailRenderTemplate() const final {
      std::ostringstream os;
      size_t num = is_write_ ? block_->writes.size() : block_->reads.size();
      std::string access_type = is_write_ ? "write" : "read";
      os << "The block {0} has " << num << " " << access_type
         << " regions, so `buffer_index` is required to be in [0, " << num
         << "). However, the input `buffer_index` is " << buffer_index_
         << ", which is out of the expected range.";
      return os.str();
    }

    IRModule mod() const final { return mod_; }
    Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }

   private:
    IRModule mod_;
    Block block_;
    int buffer_index_;
    bool is_write_;
  };

  const Array<BufferRegion>& access_region = is_write ? block->writes : block->reads;

  if (n < 0 || static_cast<int>(access_region.size()) <= n) {
    throw BufferIndexOutOfRangeError(self->mod, block, n, is_write);
  }
  return access_region[n]->buffer;
}

std::pair<Optional<StmtSRef>, bool> GetBufferDefiningSite(const StmtSRef& block_sref,
                                                          const Buffer& buffer) {
  // Climb up along the sref tree, and find the block where `buffer` is in alloc_buffers or
  // match_buffers.
  const StmtSRefNode* defining_site_sref = block_sref.get();
  while (defining_site_sref != nullptr) {
    const auto* block = defining_site_sref->StmtAs<BlockNode>();
    // If this sref is not a block sref, skip it.
    if (block == nullptr) {
      defining_site_sref = defining_site_sref->parent;
      continue;
    }
    // Try to find the buffer in `allloc_buffers`
    for (const Buffer& alloc_buffer : block->alloc_buffers) {
      if (buffer.same_as(alloc_buffer)) {
        return {GetRef<StmtSRef>(defining_site_sref), true};
      }
    }
    // We do not allow the buffer being defined in `match_buffer`.
    for (const MatchBufferRegion match_buffer : block->match_buffers) {
      if (buffer.same_as(match_buffer)) {
        return {GetRef<StmtSRef>(defining_site_sref), false};
      }
    }
    defining_site_sref = defining_site_sref->parent;
  }
  // If we cannot find the defining site block, it means that the buffer must be in the function's
  // buffer_map, which isn't an intermediate buffer.
  return {NullOpt, false};
}

/******** Pattern Matcher ********/

/*!
 * \brief PrimExpr pattern matcher.
 *
 * It is different from the pattern matcher in arith/pattern_match.h, which is dedicated
 * for compile-time constant patterns. This pattern matcher can work on dynamic user-specific
 * patterns.
 *
 * The code below shows how to use the pattern matcher.
 *
 * \code
 *
 * Var x("x"), y("y");
 * // use PrimExpr to declare patterns, x, y are holes that can be filled with
 * PatternMatcher pattern_matcher(x + y);
 * // expr = C[i, j] + A[i, k] * B[k, j], which is the expr we want to match
 * pattern_matcher.Match(expr);
 *
 * if (pattern_matcher.Success()) {
 *   pattern_matcher.Eval(x) // C[i, j]
 *   pattern_matcher.Eval(y) // A[i, k] * B[k, j]
 * }
 *
 * \endcode
 */
class PatternMatcher : public ExprVisitor {
 public:
  explicit PatternMatcher(PrimExpr pattern) : pattern_(std::move(pattern)) {}

  void VisitExpr_(const VarNode* op) final {
    auto it = filled_map_.find(op);
    if (it == filled_map_.end()) {
      filled_map_[op] = expr_to_match_;
    } else {
      ExprDeepEqual equal;
      if (it->second.same_as(expr_to_match_) || equal(it->second, expr_to_match_)) return;
      match_success_ = false;
    }
  }

  void VisitExpr_(const LoadNode* op) final {
    const auto* ptr = expr_to_match_.as<LoadNode>();
    if (ptr == nullptr) {
      match_success_ = false;
    } else {
      if (!op->buffer_var.same_as(ptr->buffer_var)) {
        match_success_ = false;
      } else {
        PrimExpr tmp = expr_to_match_;
        expr_to_match_ = ptr->predicate;
        VisitExpr(op->predicate);
        expr_to_match_ = ptr->index;
        VisitExpr(op->index);
        std::swap(expr_to_match_, tmp);
      }
    }
  }

  void VisitExpr_(const LetNode* op) final {
    const auto* ptr = expr_to_match_.as<LetNode>();
    if (ptr == nullptr) {
      match_success_ = false;
    } else {
      PrimExpr tmp = expr_to_match_;
      expr_to_match_ = ptr->var;
      VisitExpr(op->var);
      expr_to_match_ = ptr->value;
      VisitExpr(op->value);
      expr_to_match_ = ptr->body;
      VisitExpr(op->body);
      std::swap(expr_to_match_, tmp);
    }
  }

  void VisitExpr_(const CallNode* op) final {
    const auto* ptr = expr_to_match_.as<CallNode>();
    if (ptr == nullptr) {
      match_success_ = false;
    } else {
      if (!op->op.same_as(ptr->op)) {
        match_success_ = false;
      } else {
        PrimExpr tmp = expr_to_match_;
        for (size_t i = 0; i < op->args.size(); ++i) {
          expr_to_match_ = ptr->args[i];
          VisitExpr(op->args[i]);
        }
        std::swap(expr_to_match_, tmp);
      }
    }
  }

#define TVM_DECLARE_PATTERN_MATCHER_BIN_OP(OpName) \
  void VisitExpr_(const OpName* op) {              \
    const auto* ptr = expr_to_match_.as<OpName>(); \
    if (ptr == nullptr) {                          \
      match_success_ = false;                      \
    } else {                                       \
      PrimExpr current = expr_to_match_;           \
      expr_to_match_ = ptr->a;                     \
      VisitExpr(op->a);                            \
      expr_to_match_ = ptr->b;                     \
      VisitExpr(op->b);                            \
      std::swap(expr_to_match_, current);          \
    }                                              \
  }

  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(AddNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(SubNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(MulNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(DivNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(ModNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(FloorDivNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(FloorModNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(MinNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(MaxNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(EQNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(NENode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(LTNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(LENode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(GTNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(GENode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(AndNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(OrNode);

  void VisitExpr_(const CastNode* op) final {
    const auto* ptr = expr_to_match_.as<CastNode>();
    if (ptr == nullptr) {
      match_success_ = false;
    } else {
      if (!runtime::TypeEqual(op->dtype, ptr->dtype)) {
        match_success_ = false;
      } else {
        PrimExpr tmp = expr_to_match_;
        expr_to_match_ = ptr->value;
        VisitExpr(op->value);
        std::swap(expr_to_match_, tmp);
      }
    }
  }

  void VisitExpr_(const NotNode* op) final {
    const auto* ptr = expr_to_match_.as<NotNode>();
    if (ptr == nullptr) {
      match_success_ = false;
    } else {
      PrimExpr tmp = expr_to_match_;
      expr_to_match_ = ptr->a;
      VisitExpr(op->a);
      std::swap(expr_to_match_, tmp);
    }
  }

  void VisitExpr_(const SelectNode* op) final {
    const auto* ptr = expr_to_match_.as<SelectNode>();
    if (ptr == nullptr) {
      match_success_ = false;
    } else {
      PrimExpr tmp = expr_to_match_;
      expr_to_match_ = ptr->condition;
      VisitExpr(op->condition);
      expr_to_match_ = ptr->true_value;
      VisitExpr(op->true_value);
      expr_to_match_ = ptr->false_value;
      VisitExpr(op->false_value);
      std::swap(expr_to_match_, tmp);
    }
  }

  void VisitExpr_(const RampNode* op) final {
    const auto* ptr = expr_to_match_.as<RampNode>();
    if (ptr == nullptr) {
      match_success_ = false;
    } else {
      if (op->lanes != ptr->lanes) {
        match_success_ = false;
      } else {
        PrimExpr tmp = expr_to_match_;
        expr_to_match_ = ptr->base;
        VisitExpr(op->base);
        expr_to_match_ = ptr->stride;
        VisitExpr(op->stride);
        std::swap(expr_to_match_, tmp);
      }
    }
  }

  void VisitExpr_(const BroadcastNode* op) final {
    const auto* ptr = expr_to_match_.as<BroadcastNode>();
    if (ptr == nullptr) {
      match_success_ = false;
    } else {
      if (op->lanes != ptr->lanes) {
        match_success_ = false;
      } else {
        PrimExpr tmp = expr_to_match_;
        expr_to_match_ = ptr->value;
        VisitExpr(op->value);
        std::swap(expr_to_match_, tmp);
      }
    }
  }

  void VisitExpr_(const ShuffleNode* op) final {
    const auto* ptr = expr_to_match_.as<ShuffleNode>();
    if (ptr == nullptr) {
      match_success_ = false;
    } else {
      if (op->vectors.size() != ptr->vectors.size() || op->indices.size() != ptr->indices.size()) {
        match_success_ = false;
      } else {
        PrimExpr tmp = expr_to_match_;
        for (size_t i = 0; i < op->indices.size(); ++i) {
          expr_to_match_ = ptr->indices[i];
          VisitExpr(op->indices[i]);
        }
        for (size_t i = 0; i < op->vectors.size(); ++i) {
          expr_to_match_ = ptr->vectors[i];
          VisitExpr(op->vectors[i]);
        }
        std::swap(expr_to_match_, tmp);
      }
    }
  }

  void VisitExpr_(const IntImmNode* op) final {
    const auto* ptr = expr_to_match_.as<IntImmNode>();
    match_success_ = ptr != nullptr && op->value == ptr->value;
  }

  void VisitExpr_(const FloatImmNode* op) final {
    const auto* ptr = expr_to_match_.as<FloatImmNode>();
    match_success_ = ptr != nullptr && op->value == ptr->value;
  }

  void VisitExpr_(const StringImmNode* op) final {
    const auto* ptr = expr_to_match_.as<StringImmNode>();
    match_success_ = ptr != nullptr && op->value == ptr->value;
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    const auto* ptr = expr_to_match_.as<BufferLoadNode>();
    if (ptr == nullptr) {
      match_success_ = false;
    } else {
      if (!op->buffer.same_as(ptr->buffer) || op->indices.size() != ptr->indices.size()) {
        match_success_ = false;
      } else {
        PrimExpr tmp = expr_to_match_;
        for (size_t i = 0; i < op->indices.size(); ++i) {
          expr_to_match_ = ptr->indices[i];
          VisitExpr(op->indices[i]);
        }
        std::swap(expr_to_match_, tmp);
      }
    }
  }

  void Match(const PrimExpr& expr_to_match) {
    this->match_success_ = true;
    this->filled_map_.clear();
    this->expr_to_match_ = expr_to_match;
    this->operator()(pattern_);
  }

  PrimExpr Eval(const Var& var) {
    auto it = filled_map_.find(var.operator->());
    ICHECK(it != filled_map_.end()) << "Unknown pattern variable";
    ICHECK(match_success_) << "Match failed";
    return it->second;
  }

  bool Success() const { return match_success_; }

 private:
  bool match_success_{true};
  PrimExpr pattern_, expr_to_match_;
  std::unordered_map<const VarNode*, PrimExpr> filled_map_;
};

/******** Reduction Block Related ********/

class InitBodyNotBufferStoreError : public ScheduleError {
 public:
  explicit InitBodyNotBufferStoreError(IRModule mod, Block block, bool init_is_bufferstore,
                                       bool body_is_bufferstore)
      : mod_(std::move(mod)),
        block_(std::move(block)),
        init_is_bufferstore_(init_is_bufferstore),
        body_is_bufferstore_(body_is_bufferstore) {}

  String FastErrorString() const final {
    return "ScheduleError: The `init` and `body` of reduction block are required to be both "
           "BufferStore so that rfactor or cross-thread reduction can be applied";
  }

  String DetailRenderTemplate() const final {
    if (!init_is_bufferstore_ && !body_is_bufferstore_) {
      return "The `init` and `body` of block {0} are required to be BufferStore so that rfactor or "
             "cross-thread reduction can be applied";
    } else if (!init_is_bufferstore_) {
      return "The `init` of block {0} is required to be BufferStore so that rfactor or cross-thread"
             " reduction can be applied";
    } else {
      ICHECK(!body_is_bufferstore_);
      return "The `body` of block {0} is required to be BufferStore so that rfactor or cross-thread"
             " reduction can be applied";
    }
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }

  IRModule mod_;
  Block block_;
  bool init_is_bufferstore_;
  bool body_is_bufferstore_;
};

class InitBodyNotSameBufferAccessError : public ScheduleError {
 public:
  explicit InitBodyNotSameBufferAccessError(IRModule mod, Block block)
      : mod_(std::move(mod)), block_(std::move(block)) {}

  String FastErrorString() const final {
    return "ScheduleError: The `init` and `body` of the reduction block are required to have the "
           "same buffer access pattern";
  }

  String DetailRenderTemplate() const final {
    std::ostringstream os;
    const auto* init = block_->init.as<BufferStoreNode>();
    const auto* update = block_->body.as<BufferStoreNode>();
    os << "The `init` and `body` of the block {0} is required to have the same buffer access "
          "pattern. However, in block {0} the `init` writes to "
       << init->buffer->name << init->indices << ", and the `body` writes to "
       << update->buffer->name << update->indices;
    return os.str();
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }

  IRModule mod_;
  Block block_;
};

std::pair<BufferStore, BufferStore> GetBufferStoresFromReductionBlock(
    const Optional<ScheduleState>& self, const Block& block) {
  static constexpr const char* error_str1 =
      "ValueError: The `init` and `body` of the reduction block are required to be both "
      "BufferStore so that rfactor or cross-thread reduction can be applied. However, a reduction "
      "block that doesn't meet this requirement is ";
  static constexpr const char* error_str2 =
      "ValueError: The `init` and `body` of the reduction block are required to have the same "
      "buffer access pattern so that rfactor or cross-thread reduction can be applied. However, a "
      "reduction block that doesn't meet this requirement is ";

  const auto* init = block->init.as<BufferStoreNode>();
  const auto* body = block->body.as<BufferStoreNode>();
  if (!(init && body)) {
    if (self.defined()) {
      throw InitBodyNotBufferStoreError(self.value()->mod, block, init != nullptr, body != nullptr);
    } else {
      LOG(FATAL) << error_str1 << block;
    }
  }
  if (!init->buffer.same_as(body->buffer)) {
    if (self.defined()) {
      throw InitBodyNotSameBufferAccessError(self.value()->mod, block);
    } else {
      LOG(FATAL) << error_str2 << block;
    }
  }
  int ndim = static_cast<int>(init->buffer->shape.size());
  for (int i = 0; i < ndim; ++i) {
    if (!ExprDeepEqual()(init->indices[i], body->indices[i])) {
      if (self.defined()) {
        throw InitBodyNotSameBufferAccessError(self.value()->mod, block);
      } else {
        LOG(FATAL) << error_str2 << block;
      }
    }
  }
  return std::make_pair(GetRef<BufferStore>(init), GetRef<BufferStore>(body));
}

bool ContainsOnlyDataParAndReductionBlockIter(const Array<IterVar>& iters) {
  for (const IterVar& iter_var : iters) {
    if (iter_var->iter_type != kDataPar && iter_var->iter_type != kCommReduce) {
      return false;
    }
  }
  return true;
}

bool ReductionIterNotIndexOutputBuffer(const Block& block) {
  // Step 1. Collect the reduction block iters.
  std::unordered_set<const VarNode*> reduction_block_iters;
  reduction_block_iters.reserve(block->iter_vars.size());
  for (const IterVar& iter_var : block->iter_vars) {
    if (iter_var->iter_type == kCommReduce) {
      reduction_block_iters.insert(iter_var->var.get());
    }
  }
  // Step 2. Check if the reduction block iters are used to index the output buffer.
  std::unordered_set<const BufferNode*> buffer_written;
  buffer_written.reserve(block->writes.size());
  for (const BufferRegion& write_region : block->writes) {
    buffer_written.insert(write_region->buffer.get());
  }
  auto f_uses_reduction_block_var = [&](const PrimExpr& expr) -> bool {
    return UsesVar(expr, [&](const VarNode* var) {  //
      return reduction_block_iters.count(var);
    });
  };
  bool affected = false;
  PreOrderVisit(block->body, [&](const ObjectRef& obj) {
    if (affected) {
      return false;
    }
    const auto* store = obj.as<BufferStoreNode>();
    if (!store) {
      return true;
    }
    ICHECK(buffer_written.count(store->buffer.get()))
        << "ValueError: The buffer \"" << store->buffer
        << "\" is written in the block but is not in the block's signature";
    for (const PrimExpr& index : store->indices) {
      if (f_uses_reduction_block_var(index)) {
        affected = true;
        return false;
      }
    }
    return false;
  });
  return !affected;
}

class NoMatchedReducerError : public ScheduleError {
 public:
  explicit NoMatchedReducerError(IRModule mod, PrimExpr identity, BufferStore combiner)
      : mod_(std::move(mod)), identity_(std::move(identity)), combiner_(std::move(combiner)) {}

  String FastErrorString() const final {
    return "ScheduleError: No matched reducer for the identity and the combiner of this reduction "
           "block. So rfactor and cross-thread reduction cannot be applied.";
  }

  String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "No matched reducer for identity " << identity_ << " and combiner " << combiner_
       << "In this case rfactor cannot be applied. You can check tvm::tir::ReducerRegistry for "
          "default reducers or registering new reducers.";
    return os.str();
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {}; }

  IRModule mod_;
  PrimExpr identity_;
  BufferStore combiner_;
};

std::tuple<CommReducer, PrimExpr, PrimExpr> GetReducerAndCombinerLhsRhs(
    const Optional<ScheduleState>& self, const PrimExpr& identity, const BufferStore& combiner) {
  CommReducer reducer{nullptr};
  PrimExpr combiner_lhs{nullptr}, combiner_rhs{nullptr};
  bool matched = FromIdentityCombiner(identity, combiner, &reducer, &combiner_lhs, &combiner_rhs);
  if (!matched) {
    if (self.defined()) {
      throw NoMatchedReducerError(self.value()->mod, identity, combiner);
    } else {
      LOG(FATAL) << "ValueError: No matched reducer for the identity and the combiner of the "
                    "reduction block. So rfactor and cross-thread reduction cannot be applied.";
    }
  }
  return std::make_tuple(std::move(reducer), std::move(combiner_lhs), std::move(combiner_rhs));
}

/******** Commutative Reducer ********/

bool MatchReducer(const CommReducer& reducer, const PrimExpr& identity, const PrimExpr& combiner,
                  const BufferLoad& load, PrimExpr* lhs, PrimExpr* rhs) {
  if (!ExprDeepEqual()(reducer->identity_element[0], identity)) {
    return false;
  }
  PatternMatcher pattern_matcher(reducer->result[0]);
  pattern_matcher.Match(combiner);
  if (pattern_matcher.Success()) {
    PrimExpr lhs_tmp = pattern_matcher.Eval(reducer->lhs[0]);
    PrimExpr rhs_tmp = pattern_matcher.Eval(reducer->rhs[0]);
    if (ExprDeepEqual()(load, lhs_tmp)) {
      *lhs = std::move(lhs_tmp);
      *rhs = std::move(rhs_tmp);
    }
    return true;
  }
  return false;
}

bool FromIdentityCombiner(const PrimExpr& identity, const BufferStore& combiner,
                          CommReducer* result_reducer, PrimExpr* lhs, PrimExpr* rhs) {
  BufferLoad load(combiner->buffer, combiner->indices);
  // Check reduction patterns.
  for (const TypedPackedFunc<CommReducer(DataType)>& reducer_getter : GetReducerGetters()) {
    CommReducer reducer = reducer_getter(identity.dtype());
    if (MatchReducer(reducer, identity, combiner->value, load, lhs, rhs)) {
      *result_reducer = std::move(reducer);
      return true;
    }
  }
  return false;
}

/******** SRef Tree Related ********/

StmtSRef GetSRefTreeRoot(const StmtSRef& sref) {
  const StmtSRefNode* p = sref.get();
  for (; p->parent != nullptr; p = p->parent) {
  }
  return GetRef<StmtSRef>(p);
}

/******** Misc ********/

bool HasOp(const Stmt& stmt, const Array<Op>& ops) {
  std::unordered_set<const Object*> op_set;
  op_set.reserve(ops.size());
  for (const Op& op : ops) {
    op_set.insert(op.operator->());
  }
  bool found = false;
  PreOrderVisit(stmt, [&found, &op_set](const ObjectRef& obj) -> bool {
    if (found) {
      return false;
    }
    if (const auto* call = obj.as<CallNode>()) {
      if (op_set.count(call->op.operator->())) {
        found = true;
      }
    }
    return !found;
  });
  return found;
}

bool HasIfThenElse(const Stmt& stmt) {
  bool has_branch = false;
  auto f_visit = [&has_branch](const ObjectRef& obj) -> bool {
    if (has_branch) {
      // stop visiting
      return false;
    }
    if (const auto* realize = obj.as<BlockRealizeNode>()) {
      // Case 1: BlockRealize
      if (!is_one(realize->predicate)) {
        has_branch = true;
      }
    } else if (obj->IsInstance<IfThenElseNode>() || obj->IsInstance<SelectNode>()) {
      // Case 2: IfThenElse / Select
      has_branch = true;
    } else if (const auto* call = obj.as<CallNode>()) {
      // Case 3: Call the `if_then_else` operator
      static const Op& op_if_then_else = Op::Get("tir.if_then_else");
      if (call->op.same_as(op_if_then_else)) {
        has_branch = true;
      }
    }
    return !has_branch;
  };
  PreOrderVisit(stmt, f_visit);
  return has_branch;
}

std::tuple</*exists=*/bool,
           /*surjective=*/bool,
           /*injective=*/bool,
           /*ordered=*/bool,
           /*no_const_read=*/bool,
           /*no_shift_read=*/bool>
AnalyzeReadWritePattern(const BufferRegion& read_region, const BufferRegion& write_region) {
  static constexpr const std::tuple<bool, bool, bool, bool, bool, bool> kNotExist =
      std::make_tuple(false, false, false, false, false, false);
  // Step 1. Extract the write indices
  int w_dim = write_region->buffer->shape.size();
  std::unordered_map<const VarNode*, int> var2idx;
  var2idx.reserve(w_dim);
  for (int i = 0; i < w_dim; ++i) {
    const Range& dom = write_region->region[i];
    if (as_const_int(dom->extent) == nullptr) {
      return kNotExist;
    }
    if (const auto* v = dom->min.as<VarNode>()) {
      var2idx.emplace(v, i);
    } else {
      return kNotExist;
    }
  }
  // Step 2. Map each read index to a write index
  bool no_const_read = true;
  bool no_shift_read = true;
  int r_dim = read_region->buffer->shape.size();
  std::vector<int> mapped(r_dim, -1);
  for (int i = 0; i < r_dim; ++i) {
    const Range& dom = read_region->region[i];
    if (as_const_int(dom->extent) == nullptr) {
      return kNotExist;
    }
    // Case 1. Read index is a constant
    if (as_const_int(dom->min) != nullptr) {
      no_const_read = false;
      continue;
    }
    // Case 2. Read index cannot be recognized as `var +/- const`
    // where `var` is a write index and `const` is an optional constant shift
    Optional<IntImm> opt_const = NullOpt;
    const VarNode* var =
        static_cast<const VarNode*>(AnalyzeVarWithShift(dom->min, &opt_const).get());
    if (var == nullptr || !var2idx.count(var)) {
      return kNotExist;
    }
    // Case 3. Read index is `var +/- const`
    mapped[i] = var2idx.at(var);
    if (opt_const.defined()) {
      no_shift_read = false;
    }
  }
  // Step 3. Check if the mapping is ordered, and count how many times each var is mapped
  std::vector<int> mapped_counter(w_dim, 0);
  bool ordered = true;
  int last_mapped = -1;
  for (int i : mapped) {
    if (i != -1) {
      ++mapped_counter[i];
      if (last_mapped != -1 && last_mapped > i) {
        ordered = false;
      }
      last_mapped = i;
    }
  }
  // Step 4. Check if the mapping is surjective or injective
  // Surjective: each write index is mapped at least once
  // Injective: each write index is mapped at most once
  bool surjective = true;
  bool injective = true;
  for (int cnt : mapped_counter) {
    if (cnt == 0) {
      surjective = false;
    } else if (cnt >= 2) {
      injective = false;
    }
  }
  return std::make_tuple(/*exist=*/true, surjective, injective, ordered, no_const_read,
                         no_shift_read);
}

/******** Storage Scope ********/

void CheckStorageScope(const ScheduleState& self, String storage_scope) {
  class InvalidStorageScopeError : public ScheduleError {
   public:
    explicit InvalidStorageScopeError(IRModule mod, String storage_scope)
        : mod_(std::move(mod)), storage_scope_(std::move(storage_scope)) {}

    String FastErrorString() const final {
      return "ScheduleError: The input storage scope is invalid";
    }

    String DetailRenderTemplate() const final {
      return "The input storage scope \"" + storage_scope_ + "\" is invalid.";
    }

    Array<ObjectRef> LocationsOfInterest() const final { return {}; }
    IRModule mod() const final { return mod_; }

   private:
    IRModule mod_;
    String storage_scope_;
  };

  try {
    runtime::StorageScope::Create(std::string(storage_scope));
  } catch (...) {
    throw InvalidStorageScopeError(self->mod, std::move(storage_scope));
  }
}

bool IsSpatial(const StmtSRef& block_sref) {
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  for (const IterVar& iter_var : block->iter_vars) {
    if (iter_var->iter_type != IterVarType::kDataPar) {
      return false;
    }
  }
  return true;
}

bool IsTrivialBinding(const ScheduleState& self, const StmtSRef& block_sref) {
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  Array<StmtSRef> loops = GetLoops(block_sref);
  Array<PrimExpr> binds = GetBlockRealize(self, block_sref)->iter_values;
  if (loops.size() != binds.size()) {
    return false;
  }
  for (int i = 0, n = loops.size(); i < n; ++i) {
    const ForNode* loop = TVM_SREF_TO_FOR(loop, loops[i]);
    if (binds[i].get() != loop->loop_var.get()) {
      return false;
    }
  }
  return true;
}

bool NeedsMultiLevelTiling(const ScheduleState& self, const StmtSRef& block_sref) {
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  if (block->writes.size() != 1 || block->reads.empty() || IsSpatial(block_sref) ||
      !IsTrivialBinding(self, block_sref)) {
    return false;
  }
  const BufferNode* write_buffer = block->writes[0]->buffer.get();
  // Step 1. Sort out spatial block variables. Skip the block iters of domain [0, 1), since such
  // block iters distracts the following check of the unused block iters.
  std::vector<const VarNode*> spatial_block_vars;
  spatial_block_vars.reserve(block->iter_vars.size());
  for (const IterVar& block_var : block->iter_vars) {
    const int64_t* dom_min = as_const_int(block_var->dom->min);
    const int64_t* dom_extent = as_const_int(block_var->dom->extent);
    bool has_trivial_dom =
        dom_min != nullptr && dom_extent != nullptr && *dom_min == 0 && *dom_extent == 1;
    if (block_var->iter_type == IterVarType::kDataPar && !has_trivial_dom) {
      spatial_block_vars.push_back(block_var->var.get());
    }
  }
  // Step 2. Enumerate each read region, check the number of block vars that are not used
  // to index the read region
  int total_unused_block_vars = 0;
  std::unordered_set<const BufferNode*> read_buffers;
  read_buffers.reserve(block->reads.size());
  for (const BufferRegion& buffer_region : block->reads) {
    const BufferNode* buffer = buffer_region->buffer.get();
    const Array<Range>& regions = buffer_region->region;
    // Step 2.1. Duplication of read buffers are not allowed
    if (read_buffers.insert(buffer).second == false) {
      return false;
    }
    // Step 2.2. Skip the reduction buffer
    if (buffer == write_buffer) {
      continue;
    }
    // Step 2.3. Collect the block vars that are used to index the read region
    std::unordered_set<const VarNode*> vars;
    for (const Range& range : regions) {
      if (as_const_int(range->extent) == nullptr) {
        return false;
      }
      for (const Var& var : UndefinedVars(range->min)) {
        vars.insert(var.get());
      }
    }
    // Step 2.4. Check if the block vars are not used to index the read region
    int n_unused_block_vars = 0;
    for (const VarNode* block_var : spatial_block_vars) {
      if (vars.count(block_var) == 0) {
        ++n_unused_block_vars;
      }
    }
    total_unused_block_vars += n_unused_block_vars;
  }
  return total_unused_block_vars >= 1;
}

bool IsSpatialPrimFunc(const PrimFunc& func) {
  bool result = true;
  PreOrderVisit(func->body, [&result](const ObjectRef& obj) {
    if (result == false) {
      return false;
    }
    if (const auto* block = obj.as<BlockNode>()) {
      for (const IterVar& iter_var : block->iter_vars) {
        if (iter_var->iter_type != IterVarType::kDataPar) {
          result = false;
          return false;
        }
      }
    }
    return true;
  });
  return result;
}

std::pair<int64_t, int64_t> GetCumulativeSpaceAndReductionLength(const tir::ScheduleState& self,
                                                                 const tir::StmtSRef& block_sref) {
  Array<tir::StmtSRef> loops = tir::GetLoops(block_sref);
  int64_t cum_space_len = 1, cum_reduce_len = 1;
  /*
   * Return (-1, -1) if
   *   1. there is some loop with type other than kDataPar and kCommReduce;
   *   2. there is some loop which is dynamic.
   */
  for (const tir::StmtSRef& loop_sref : loops) {
    tir::IterVarType type = GetLoopIterType(loop_sref);
    if (type == tir::kDataPar) {
      const int64_t* extent = GetLoopIntExtent(loop_sref);
      if (*extent != -1) {
        cum_space_len *= *extent;
      } else {
        return std::make_pair(-1, -1);
      }
    } else if (type == tir::kCommReduce) {
      const int64_t* extent = GetLoopIntExtent(loop_sref);
      if (*extent != -1) {
        cum_reduce_len *= *extent;
      } else {
        return std::make_pair(-1, -1);
      }
    } else {
      return std::make_pair(-1, -1);
    }
  }
  return std::make_pair(cum_space_len, cum_reduce_len);
}

bool NeedsRFactorOrCrossThreadReduction(const tir::ScheduleState& self,   //
                                        const tir::StmtSRef& block_sref,  //
                                        int64_t max_parallel_extent,      //
                                        int64_t max_parallel_basic) {
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  Array<tir::StmtSRef> loops = tir::GetLoops(block_sref);

  // Cond 1. The block has only one write buffer
  if (block->writes.size() != 1) {
    return false;
  }

  // Cond 2. The block is a reduction block and has trivial binding.
  const StmtSRef& scope_sref = GetScopeRoot(self, block_sref,
                                            /*require_stage_pipeline=*/false);
  if (!IsReductionBlock(self, block_sref, scope_sref)  //
      || !IsTrivialBinding(self, block_sref)           //
      || HasBeenMultiLevelTiled(block_sref)) {
    return false;
  }

  // Cond 3. Every the loop axis must be either spatial axis or reduction axis.
  for (const tir::StmtSRef& loop_sref : loops) {
    const tir::IterVarType& type = GetLoopIterType(loop_sref);
    if (type != tir::kDataPar && type != tir::kCommReduce) {
      return false;
    }
  }

  // Cond 4. Whether there is at least one reduction loop.
  // Cond 5. The loops are continuous, and the body of the innermost loop is exactly the block.
  bool has_reduction_loop = false;
  for (size_t i = 0; i < loops.size(); ++i) {
    // Cond 4.
    if (GetLoopIterType(loops[i]) == tir::kCommReduce) {
      has_reduction_loop = true;
    }

    // Cond 5.
    const ForNode* loop_i = TVM_SREF_TO_FOR(loop_i, loops[i]);
    if (i < loops.size() - 1) {
      const ForNode* loop_i1 = TVM_SREF_TO_FOR(loop_i1, loops[i + 1]);
      if (loop_i->body.get() != loop_i1) {
        return false;
      }
    } else {
      const auto* block_realize = loop_i->body.as<tir::BlockRealizeNode>();
      if (!block_realize || block_realize->block.get() != block) {
        return false;
      }
    }
  }
  if (!has_reduction_loop) {
    return false;
  }

  // Cond 6. Can successfully calculating the cumulative loop length.
  int64_t cum_space_len, cum_reduce_len;
  std::tie(cum_space_len, cum_reduce_len) = GetCumulativeSpaceAndReductionLength(self, block_sref);
  if (cum_space_len == -1 || cum_reduce_len == -1) {
    return false;
  }

  // Cond 7.
  if (NeedsMultiLevelTiling(self, block_sref)) {
    // Do not use rfactor/cross-thread-reduction if we have enough parallelism on spatial loops.
    return !(cum_space_len >= cum_reduce_len || cum_space_len > max_parallel_extent);
  } else if (cum_reduce_len > 1) {
    // Always try rfactor/cross-thread-reduction for other reduction blocks.
    return cum_reduce_len > max_parallel_basic;
  } else {
    return false;
  }
}

TVM_REGISTER_NODE_TYPE(TensorizeInfoNode);

Optional<TensorizeInfo> GetTensorizeLoopMapping(const tir::ScheduleState& self,
                                                const tir::StmtSRef& block_sref,
                                                const tir::PrimFunc& desc_func) {
  arith::Analyzer analyzer;
  const tir::BlockRealize& block = tir::GetBlockRealize(self, block_sref);
  // Step 1. Analyze desc_func, extract its block, loops and loop vars
  const tir::BlockRealizeNode* desc_block = nullptr;
  std::vector<const tir::ForNode*> desc_loops;
  std::unordered_set<const tir::VarNode*> desc_loop_vars;
  const auto* desc_scope_realize = desc_func->body.as<tir::BlockRealizeNode>();
  ICHECK(desc_scope_realize);
  {
    auto f_visit = [&desc_block, &desc_loops, &desc_loop_vars,
                    &analyzer](const ObjectRef& obj) -> bool {
      // Extract the block
      if (const auto* block = obj.as<tir::BlockRealizeNode>()) {
        desc_block = block;
        return false;
      }
      // Extract loops
      if (const auto* loop = obj.as<tir::ForNode>()) {
        desc_loops.push_back(loop);
        desc_loop_vars.insert(loop->loop_var.get());
        if (!analyzer.CanProve(loop->min == 0)) {
          return false;
        }
      }
      return true;
    };
    tir::PostOrderVisit(desc_scope_realize->block->body, f_visit);
    std::reverse(desc_loops.begin(), desc_loops.end());
    ICHECK(desc_block);
  }
  // Step 2. Collect loops from block_sref
  const tir::StmtSRef& scope_sref = GetScopeRoot(self, block_sref, false);
  const tir::BlockNode* scope_block = TVM_SREF_TO_BLOCK(scope_block, scope_sref);
  std::vector<const tir::ForNode*> block_loops;
  std::unordered_set<const tir::VarNode*> block_loop_vars;
  {
    for (const tir::StmtSRefNode* loop_sref = block_sref->parent;; loop_sref = loop_sref->parent) {
      const auto* loop = loop_sref->StmtAs<tir::ForNode>();
      if (loop == nullptr || loop->body->IsInstance<tir::SeqStmtNode>()) {
        break;
      }
      block_loops.push_back(loop);
      block_loop_vars.insert(loop->loop_var.get());
      if (!analyzer.CanProve(loop->min == 0)) {
        return NullOpt;
      }
    }
    std::reverse(block_loops.begin(), block_loops.end());
  }
  // Step 3. Map from block loops to desc block loops
  ObjectPtr<TensorizeInfoNode> ret = make_object<TensorizeInfoNode>();
  const int n_block_vars = block->iter_values.size();
  const int n_desc_vars = desc_block->iter_values.size();
  const int offset = n_block_vars - n_desc_vars;

  if (offset < 0) {
    return NullOpt;
  }

  const std::vector<IterVarType> iter_types_block = GetBlockVarTypes(block_sref);
  const std::vector<IterVarType> iter_types_desc = GetBlockVarTypes(desc_block->block.get());

  ICHECK(desc_loops.size() == static_cast<size_t>(n_desc_vars));
  ICHECK(block_loops.size() == iter_types_block.size());

  // We assume that the orders of iter_vars in the target and the desc block are consistent.
  // Based on that assumption, the following logic supports arbitrary permutations of a loop order,
  // such as

  // for k:
  //   for i:
  //     for j:
  //       C[i, j] += A[i, k] * B[k, j]

  // or

  // for i:
  //   for j:
  //     for k:
  //       C[i, j] += A[i, k] * B[k, j]

  bool paddable = true;
  std::unordered_map<int, int> padding;

  int next_block_ind = block_loops.size() - 1;
  for (int i_desc = n_desc_vars - 1; i_desc >= 0; --i_desc) {
    // Step 3.1. Find the corresponding loop of the i_desc-th block var of desc
    const PrimExpr& desc_bind = desc_block->iter_values[i_desc];
    const tir::ForNode* desc_loop = nullptr;
    IterVarType iter_type_desc = iter_types_desc[i_desc];
    for (int i = 0, n = desc_loops.size(); i < n; ++i) {
      // Check if desc_bind = loops[i]->loop_var + stuff-irrelevant-of-loop-vars
      PrimExpr residual = analyzer.Simplify(desc_bind - desc_loops[i]->loop_var);
      if (!UsesVar(residual,
                   [&desc_loop_vars](const VarNode* var) { return desc_loop_vars.count(var); })) {
        desc_loop = desc_loops[i];
        iter_type_desc = iter_types_desc[i];
        break;
      }
    }
    if (desc_loop == nullptr || desc_loop->extent.as<IntImmNode>() == nullptr) {
      return NullOpt;
    }

    const IntImmNode* int_desc_extent = desc_loop->extent.as<IntImmNode>();

    // Step 3.2. Find the corresponding iter_value of the target block with a matching iterator type
    PrimExpr block_bind;
    for (int i = next_block_ind; i >= 0; --i) {
      if (iter_types_block[i] == iter_type_desc) {
        next_block_ind = i - 1;
        block_bind = block->iter_values[i];
        break;
      } else {
        return NullOpt;
      }
    }
    if (!block_bind.defined()) return NullOpt;

    // Step 3.3. Find the corresponding loop of the target block
    for (int i = 0, n = block_loops.size(); i < n; ++i) {
      // Check if block_bind = block_loops[i]->loop_var + stuff-irrelevant-of-loop-vars
      const tir::ForNode* block_loop = block_loops[i];
      const tir::StmtSRef& block_loop_sref = self->stmt2ref[block_loop];
      // Skip i-th loop if it has already been mapped
      if (ret->loop_map.find(block_loop_sref) != ret->loop_map.end()) continue;

      PrimExpr residual = analyzer.Simplify(block_bind - block_loops[i]->loop_var);
      if (UsesVar(residual,
                  [&block_loop_vars](const VarNode* var) { return block_loop_vars.count(var); }))
        continue;
      // We only pad block_ibnd = block_loops[i]
      if (!is_zero(residual)) {
        paddable = false;
      }

      const IntImmNode* int_block_extent = block_loops[i]->extent.as<IntImmNode>();

      // Check divisibility
      if (int_block_extent) {
        if (int_block_extent->value == 1) {
          return NullOpt;
        }
        if (int_block_extent->value % int_desc_extent->value != 0) {
         padding[next_block_ind + 1] = (int_block_extent->value + int_desc_extent->value - 1) / int_desc_extent->value * int_desc_extent->value;
        }
      } else {
        return NullOpt;
      }

      ret->loop_map.Set(block_loop_sref, GetRef<tir::For>(desc_loop));
      break;
    }
  }

  if (!padding.empty()) {
    if (!paddable) return NullOpt;
    std::vector<IntImm> final_padding;
    for (size_t i = 0; i < block->block->iter_vars.size(); ++i) {
      const IterVar& iter = block->block->iter_vars[i];
      auto it = padding.find(i);
      if (it != padding.end()) {
        final_padding.push_back(IntImm(iter->dom->extent.dtype(), it->second));
      } else {
        const IntImmNode* extent = iter->dom->extent.as<IntImmNode>();
        final_padding.push_back(GetRef<IntImm>(extent));
      }
    }
    ret->padding = std::move(final_padding);
  }

  for (int i = 0, n = desc_loops.size(); i < n; ++i) {
    ret->desc_loop_indexer.Set(GetRef<tir::For>(desc_loops[i]), Integer(i));
  }
  return TensorizeInfo(ret);
}

TVM_REGISTER_GLOBAL("tir.schedule.IsSpatialPrimFunc").set_body_typed(IsSpatialPrimFunc);
TVM_REGISTER_GLOBAL("tir.schedule.GetTensorizeLoopMapping")
    .set_body_typed([](Schedule sch, BlockRV block, PrimFunc desc_func) {
      return GetTensorizeLoopMapping(sch->state(), sch->GetSRef(block), desc_func);
    });
}  // namespace tir
}  // namespace tvm
