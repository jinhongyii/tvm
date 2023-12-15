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
 * \file tir/analysis/verify_well_formed.cc
 * \brief Check if schedulable tir is well-formed.
 */

#include <tvm/runtime/registry.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

#include <exception>
#include <optional>
#include <variant>

#include "../ir/functor_common.h"
#include "tvm/ir/module.h"

namespace tvm {
namespace tir {

namespace {
class TIRVisitorWithPath : protected ExprFunctor<void(const PrimExpr&, ObjectPath)>,
                           protected StmtFunctor<void(const Stmt&, ObjectPath)> {
 public:
  template <typename TObjectRef>
  void operator()(TObjectRef&& obj) {
    Visit(std::forward<TObjectRef>(obj), ObjectPath::Root());
  }

 protected:
  inline void Visit(const IRModule& obj, ObjectPath path) { VisitIRModule(obj, path); }
  inline void Visit(const PrimFunc& obj, ObjectPath path) { VisitPrimFunc(obj, path); }
  inline void Visit(const GlobalVar& obj, ObjectPath path) { VisitGlobalVar(obj, path); }

  inline void Visit(const PrimExpr& obj, ObjectPath path) { VisitExpr(obj, path); }
  inline void Visit(const Stmt& obj, ObjectPath path) { VisitStmt(obj, path); }

  inline void Visit(const Range& obj, ObjectPath path) { VisitRange(obj, path); }
  inline void Visit(const Buffer& obj, ObjectPath path) { VisitBuffer(obj, path); }
  inline void Visit(const BufferRegion& obj, ObjectPath path) { VisitBufferRegion(obj, path); }
  inline void Visit(const MatchBufferRegion& obj, ObjectPath path) {
    VisitMatchBufferRegion(obj, path);
  }
  // inline void Visit(const IterVar& obj, ObjectPath path) { VisitIterVar(obj, path); }

  virtual void EnterDef(const GlobalVar& var, ObjectPath path);
  virtual void EnterDef(const Var& var, ObjectPath path);
  // virtual void EnterDef(const IterVar& var, ObjectPath path);
  virtual void EnterDef(const Buffer& buffer, ObjectPath path);

  virtual void ExitDef(const GlobalVar& var, ObjectPath path);
  virtual void ExitDef(const Var& var, ObjectPath path);
  // virtual void ExitDef(const IterVar& var, ObjectPath path);
  virtual void ExitDef(const Buffer& buffer, ObjectPath path);

  virtual void VisitIRModule(const IRModule& mod, ObjectPath path);
  virtual void VisitGlobalVar(const GlobalVar& gvar, ObjectPath path);
  virtual void VisitPrimFunc(const PrimFunc& func, ObjectPath path);

  virtual void VisitBuffer(const Buffer& buffer, ObjectPath path);
  virtual void VisitBufferRegion(const BufferRegion& region, ObjectPath path);
  virtual void VisitMatchBufferRegion(const MatchBufferRegion& region, ObjectPath path);

  // virtual void VisitIterVar(const IterVar& iter_var, ObjectPath path);

  virtual void VisitRange(const Range& range, ObjectPath path);

  template <typename T>
  inline void Visit(const Array<T>& arr, ObjectPath path) {
    for (size_t i = 0; i < arr.size(); i++) {
      Visit(arr[i], path->ArrayIndex(i));
    }
  }

  template <typename T>
  inline void Visit(const Optional<T>& opt, ObjectPath path) {
    if (opt) {
      Visit(opt.value(), path);
    }
  }

  using StmtFunctor::VisitStmt;
  void VisitStmt_(const AttrStmtNode* op, ObjectPath path) override;
  void VisitStmt_(const IfThenElseNode* op, ObjectPath path) override;
  void VisitStmt_(const LetStmtNode* op, ObjectPath path) override;
  void VisitStmt_(const ForNode* op, ObjectPath path) override;
  void VisitStmt_(const WhileNode* op, ObjectPath path) override;
  void VisitStmt_(const AllocateNode* op, ObjectPath path) override;
  void VisitStmt_(const AllocateConstNode* op, ObjectPath path) override;
  void VisitStmt_(const DeclBufferNode* op, ObjectPath path) override;
  void VisitStmt_(const BufferStoreNode* op, ObjectPath path) override;
  void VisitStmt_(const BufferRealizeNode* op, ObjectPath path) override;
  void VisitStmt_(const AssertStmtNode* op, ObjectPath path) override;
  void VisitStmt_(const ProducerStoreNode* op, ObjectPath path) override;
  void VisitStmt_(const ProducerRealizeNode* op, ObjectPath path) override;
  void VisitStmt_(const PrefetchNode* op, ObjectPath path) override;
  void VisitStmt_(const SeqStmtNode* op, ObjectPath path) override;
  void VisitStmt_(const EvaluateNode* op, ObjectPath path) override;
  void VisitStmt_(const BlockNode* op, ObjectPath path) override;
  void VisitStmt_(const BlockRealizeNode* op, ObjectPath path) override;

  using ExprFunctor::VisitExpr;
  void VisitExpr_(const VarNode* op, ObjectPath path) override;
  void VisitExpr_(const SizeVarNode* op, ObjectPath path) override;
  void VisitExpr_(const BufferLoadNode* op, ObjectPath path) override;
  void VisitExpr_(const ProducerLoadNode* op, ObjectPath path) override;
  void VisitExpr_(const LetNode* op, ObjectPath path) override;
  void VisitExpr_(const CallNode* op, ObjectPath path) override;
  void VisitExpr_(const AddNode* op, ObjectPath path) override;
  void VisitExpr_(const SubNode* op, ObjectPath path) override;
  void VisitExpr_(const MulNode* op, ObjectPath path) override;
  void VisitExpr_(const DivNode* op, ObjectPath path) override;
  void VisitExpr_(const ModNode* op, ObjectPath path) override;
  void VisitExpr_(const FloorDivNode* op, ObjectPath path) override;
  void VisitExpr_(const FloorModNode* op, ObjectPath path) override;
  void VisitExpr_(const MinNode* op, ObjectPath path) override;
  void VisitExpr_(const MaxNode* op, ObjectPath path) override;
  void VisitExpr_(const EQNode* op, ObjectPath path) override;
  void VisitExpr_(const NENode* op, ObjectPath path) override;
  void VisitExpr_(const LTNode* op, ObjectPath path) override;
  void VisitExpr_(const LENode* op, ObjectPath path) override;
  void VisitExpr_(const GTNode* op, ObjectPath path) override;
  void VisitExpr_(const GENode* op, ObjectPath path) override;
  void VisitExpr_(const AndNode* op, ObjectPath path) override;
  void VisitExpr_(const OrNode* op, ObjectPath path) override;
  void VisitExpr_(const ReduceNode* op, ObjectPath path) override;
  void VisitExpr_(const CastNode* op, ObjectPath path) override;
  void VisitExpr_(const NotNode* op, ObjectPath path) override;
  void VisitExpr_(const SelectNode* op, ObjectPath path) override;
  void VisitExpr_(const RampNode* op, ObjectPath path) override;
  void VisitExpr_(const BroadcastNode* op, ObjectPath path) override;
  void VisitExpr_(const ShuffleNode* op, ObjectPath path) override;
  void VisitExpr_(const IntImmNode* op, ObjectPath path) override;
  void VisitExpr_(const FloatImmNode* op, ObjectPath path) override;
  void VisitExpr_(const StringImmNode* op, ObjectPath path) override;
  void VisitExpr_(const AnyNode* op, ObjectPath path) override;

  template <typename T>
  class DefContext {
   public:
    DefContext(DefContext&& other) { swap(std::move(other)); }
    DefContext& operator=(DefContext&& other) {
      swap(std::move(other));
      return *this;
    }

    DefContext(const DefContext&) = delete;
    DefContext& operator=(const DefContext&) = delete;
    ~DefContext() {
      // Checks performed when a definition goes out of scope may
      // raise an exception.  If the stack is already being unwound
      // due to another exception being thrown, this would cause a
      // segfault and terminate the program.  By checking that no
      // additional exceptions have been thrown between the
      // construction of the DefContext and the destruction, we avoid
      // this case and allow the first error to propagate upward.
      LOG(DEBUG) << "At construction, " << uncaught_exceptions_
                 << " active exceptions,  and at destruction " << std::uncaught_exceptions();
      if (self_ && std::uncaught_exceptions() == uncaught_exceptions_) {
        LOG(DEBUG) << "Calling ExitDef for " << obj_;
        self_->ExitDef(obj_, path_);
      }
    }

   private:
    friend class TIRVisitorWithPath;

    DefContext(TIRVisitorWithPath* self, T obj, ObjectPath path)
        : self_(self), obj_(obj), path_(path), uncaught_exceptions_(std::uncaught_exceptions()) {
      self_->EnterDef(obj_, path_);
    }

    void swap(DefContext&& other) {
      std::swap(this->self_, other.self_);
      std::swap(this->obj_, other.obj_);
      std::swap(this->path_, other.path_);
      std::swap(this->uncaught_exceptions_, other.uncaught_exceptions_);
    }

    TIRVisitorWithPath* self_{nullptr};
    T obj_;
    ObjectPath path_{ObjectPath::Root()};
    int uncaught_exceptions_{-1};
  };

  template <typename T>
  DefContext<T> WithDef(T obj, ObjectPath path) {
    return DefContext(this, obj, path);
  }
};

void TIRVisitorWithPath::VisitIRModule(const IRModule& mod, ObjectPath path) {
  // To ensure deterministic order of visits, sort the GlobalVar first
  // by visibility (public then private), then alphabetically by name.
  std::vector<GlobalVar> gvars;
  std::unordered_set<GlobalVar, ObjectPtrHash, ObjectPtrEqual> externally_exposed;
  for (const auto& [gvar, func] : mod->functions) {
    gvars.push_back(gvar);
    if (func->GetAttr<String>(tvm::attr::kGlobalSymbol).defined()) {
      externally_exposed.insert(gvar);
    }
  }

  std::sort(gvars.begin(), gvars.end(),
            [&externally_exposed](const GlobalVar& a, const GlobalVar& b) {
              bool a_exposed = externally_exposed.count(a);
              bool b_exposed = externally_exposed.count(b);
              if (a_exposed != b_exposed) {
                return a < b;
              } else {
                return a->name_hint < b->name_hint;
              }
            });

  std::vector<DefContext<GlobalVar>> context;

  for (const auto& gvar : gvars) {
    context.push_back(WithDef(gvar, path->Attr("global_var_map_")->MapValue(gvar->name_hint)));
  }

  for (const auto& gvar : gvars) {
    auto base_func = mod->functions[gvar];
    if (auto prim_func = base_func.as<PrimFunc>()) {
      Visit(prim_func.value(), path->Attr("functions")->MapValue(gvar));
    }
  }

  while (context.size()) context.pop_back();
}

void TIRVisitorWithPath::VisitGlobalVar(const GlobalVar& gvar, ObjectPath path) {}

void TIRVisitorWithPath::VisitPrimFunc(const PrimFunc& func, ObjectPath path) {
  // The implicit definitions from a PrimFunc::buffer_map are pretty
  // weird.  They only apply if no previous definition of that
  // variable has occurred.  Therefore, to ensure that we only avoid
  // duplicate calls to VisitVarDef, these semantics need to be
  // checked.
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> defined_params;
  std::vector<std::variant<DefContext<Var>, DefContext<Buffer>>> context;

  auto ppath = path->Attr("params");
  for (size_t i = 0; i < func->params.size(); i++) {
    context.push_back(WithDef(func->params[i], ppath->ArrayIndex(i)));
    defined_params.insert(func->params[i]);
  }

  auto try_visit_implicit_var_def = [this, &defined_params, &context](const PrimExpr& expr,
                                                                      ObjectPath path) {
    if (auto opt = expr.as<Var>()) {
      auto var = opt.value();
      if (!defined_params.count(var)) {
        context.push_back(WithDef(var, path));
        defined_params.insert(var);
      }
    }
  };
  auto try_visit_implicit_var_def_array = [&try_visit_implicit_var_def](const Array<PrimExpr>& arr,
                                                                        ObjectPath path) {
    for (size_t i = 0; i < arr.size(); i++) {
      try_visit_implicit_var_def(arr[i], path->ArrayIndex(i));
    }
  };

  auto buffer_map_path = path->Attr("buffer_map");
  for (size_t i = 0; i < func->params.size(); i++) {
    if (auto opt = func->buffer_map.Get(func->params[i])) {
      auto buf = opt.value();
      auto buf_path = buffer_map_path->MapValue(ppath->ArrayIndex(i));

      // A buffer in the buffer_map always defines its data pointer
      context.push_back(WithDef(buf->data, buf_path->Attr("data")));

      // But other implicit definitions only apply if they weren't
      // provided as explicit parameters, and they weren't defined
      // implicitly by any previous buffer.
      try_visit_implicit_var_def_array(buf->shape, buf_path->Attr("shape"));
      try_visit_implicit_var_def_array(buf->strides, buf_path->Attr("strides"));
      try_visit_implicit_var_def(buf->elem_offset, buf_path->Attr("elem_offset"));
    }
  }

  // Only after all the implicit definitions have been visited can we
  // visit the buffer definition itself.
  for (size_t i = 0; i < func->params.size(); i++) {
    if (auto opt = func->buffer_map.Get(func->params[i])) {
      auto buf_path = buffer_map_path->MapValue(ppath->ArrayIndex(i));
      EnterDef(opt.value(), buf_path);
    }
  }

  Visit(func->body, path->Attr("body"));

  while (context.size()) context.pop_back();
}

void TIRVisitorWithPath::EnterDef(const GlobalVar& gvar, ObjectPath path) {}

void TIRVisitorWithPath::ExitDef(const GlobalVar& gvar, ObjectPath path) {}

void TIRVisitorWithPath::EnterDef(const Var& var, ObjectPath path) {}

void TIRVisitorWithPath::ExitDef(const Var& var, ObjectPath path) {}

// void TIRVisitorWithPath::EnterDef(const IterVar& iter_var, ObjectPath path) {
//   Visit(iter_var->dom, path->Attr("dom"));
//   EnterDef(iter_var->var, path->Attr("var"));
// }

// void TIRVisitorWithPath::ExitDef(const IterVar& iter_var, ObjectPath path) {
//   ExitDef(iter_var->var, path->Attr("var"));
// }

void TIRVisitorWithPath::EnterDef(const Buffer& buffer, ObjectPath path) {
  // Defining a buffer counts as using all parameters in the buffer
  // (e.g. shape/strides).
  Visit(buffer->data, path->Attr("data"));
  Visit(buffer->shape, path->Attr("shape"));
  Visit(buffer->strides, path->Attr("strides"));
  Visit(buffer->elem_offset, path->Attr("elem_offset"));
}
void TIRVisitorWithPath::ExitDef(const Buffer& buffer, ObjectPath path) {}

void TIRVisitorWithPath::VisitBuffer(const Buffer& buffer, ObjectPath path) {
  // Using a buffer *also* counts as using all parameters in the buffer.
  Visit(buffer->data, path->Attr("data"));
  Visit(buffer->shape, path->Attr("shape"));
  Visit(buffer->strides, path->Attr("strides"));
  Visit(buffer->elem_offset, path->Attr("elem_offset"));
}

void TIRVisitorWithPath::VisitBufferRegion(const BufferRegion& region, ObjectPath path) {
  Visit(region->buffer, path->Attr("path"));
  Visit(region->region, path->Attr("region"));
}

void TIRVisitorWithPath::VisitMatchBufferRegion(const MatchBufferRegion& match, ObjectPath path) {
  Visit(match->source, path->Attr("source"));

  // MatchBufferRegion define the match->buffer, but do not own the
  // body in which the match->buffer is defined.  Therefore, the
  // definitions are handled in the BlockNode visitor.
}

// void TIRVisitorWithPath::VisitIterVar(const IterVar& iter_var, ObjectPath path) {
//   Visit(iter_var->dom, path->Attr("dom"));
//   Visit(iter_var->var, path->Attr("var"));
// }

void TIRVisitorWithPath::VisitRange(const Range& range, ObjectPath path) {
  Visit(range->min, path->Attr("min"));
  Visit(range->extent, path->Attr("extent"));
}

void TIRVisitorWithPath::VisitStmt_(const LetStmtNode* op, ObjectPath path) {
  Visit(op->value, path->Attr("value"));
  auto context = WithDef(op->var, path->Attr("var"));
  Visit(op->body, path->Attr("body"));
}

void TIRVisitorWithPath::VisitStmt_(const AttrStmtNode* op, ObjectPath path) {
  Visit(op->value, path->Attr("value"));

  std::optional<DefContext<Var>> context = std::nullopt;
  if (auto ptr = op->node.as<IterVarNode>(); ptr && op->attr_key == attr::thread_extent) {
    // Some attributes serve as a source of definition for the
    // tir::Var they annotate.
    Visit(ptr->dom, path->Attr("node")->Attr("dom"));
    context = WithDef(ptr->var, path->Attr("node")->Attr("var"));
  } else if (auto expr = op->node.as<PrimExpr>()) {
    Visit(expr.value(), path->Attr("node"));
  }
  Visit(op->body, path->Attr("body"));
}

void TIRVisitorWithPath::VisitStmt_(const ForNode* op, ObjectPath path) {
  Visit(op->min, path->Attr("min"));
  Visit(op->extent, path->Attr("extent"));
  auto context = WithDef(op->loop_var, path->Attr("loop_var"));
  Visit(op->body, path->Attr("body"));
}

void TIRVisitorWithPath::VisitStmt_(const WhileNode* op, ObjectPath path) {
  Visit(op->condition, path->Attr("condition"));
  Visit(op->body, path->Attr("body"));
}

void TIRVisitorWithPath::VisitStmt_(const AllocateNode* op, ObjectPath path) {
  Visit(op->condition, path->Attr("condition"));
  Visit(op->extents, path->Attr("extents"));
  auto context = WithDef(op->buffer_var, path->Attr("buffer_var"));
  Visit(op->body, path->Attr("body"));
}

void TIRVisitorWithPath::VisitStmt_(const AllocateConstNode* op, ObjectPath path) {
  Visit(op->extents, path->Attr("extents"));
  auto context = WithDef(op->buffer_var, path->Attr("buffer_var"));
  Visit(op->body, path->Attr("body"));
}

void TIRVisitorWithPath::VisitStmt_(const DeclBufferNode* op, ObjectPath path) {
  auto context = WithDef(op->buffer, path->Attr("buffer"));
  Visit(op->body, path->Attr("body"));
}

void TIRVisitorWithPath::VisitStmt_(const BufferStoreNode* op, ObjectPath path) {
  Visit(op->value, path->Attr("value"));
  Visit(op->buffer, path->Attr("buffer"));
  Visit(op->indices, path->Attr("indices"));
}

void TIRVisitorWithPath::VisitStmt_(const BufferRealizeNode* op, ObjectPath path) {
  Visit(op->condition, path->Attr("condition"));
  Visit(op->bounds, path->Attr("bounds"));
  auto context = WithDef(op->buffer, path->Attr("buffer"));
  Visit(op->body, path->Attr("body"));
}

void TIRVisitorWithPath::VisitStmt_(const IfThenElseNode* op, ObjectPath path) {
  Visit(op->condition, path->Attr("condition"));
  Visit(op->then_case, path->Attr("then_case"));
  Visit(op->else_case, path->Attr("else_case"));
}

void TIRVisitorWithPath::VisitStmt_(const AssertStmtNode* op, ObjectPath path) {
  Visit(op->condition, path->Attr("condition"));
  Visit(op->message, path->Attr("message"));
  Visit(op->body, path->Attr("body"));
}

void TIRVisitorWithPath::VisitStmt_(const ProducerStoreNode* op, ObjectPath path) {
  Visit(op->indices, path->Attr("indices"));
  Visit(op->value, path->Attr("value"));
}

void TIRVisitorWithPath::VisitStmt_(const ProducerRealizeNode* op, ObjectPath path) {
  Visit(op->bounds, path->Attr("bounds"));
  Visit(op->body, path->Attr("body"));
  Visit(op->condition, path->Attr("condition"));
}

void TIRVisitorWithPath::VisitStmt_(const PrefetchNode* op, ObjectPath path) {
  Visit(op->bounds, path->Attr("bounds"));
}

void TIRVisitorWithPath::VisitStmt_(const SeqStmtNode* op, ObjectPath path) {
  Visit(op->seq, path->Attr("seq"));
}

void TIRVisitorWithPath::VisitStmt_(const EvaluateNode* op, ObjectPath path) {
  Visit(op->value, path->Attr("value"));
}

void TIRVisitorWithPath::VisitStmt_(const BlockNode* op, ObjectPath path) {
  std::vector<std::variant<DefContext<Var>, DefContext<Buffer>>> context;

  Visit(op->iter_vars, path->Attr("iter_vars"));
  Visit(op->reads, path->Attr("reads"));
  Visit(op->writes, path->Attr("writes"));

  {
    auto alloc_path = path->Attr("alloc_buffers");
    for (size_t i = 0; i < op->alloc_buffers.size(); i++) {
      auto buffer_path = alloc_path->ArrayIndex(i);
      auto buf = op->alloc_buffers[i];
      context.push_back(WithDef(buf->data, buffer_path->Attr("data")));
      context.push_back(WithDef(buf, buffer_path));
    }
  }

  {
    auto match_path = path->Attr("match_buffers");
    Visit(op->match_buffers, match_path);

    for (size_t i = 0; i < op->match_buffers.size(); i++) {
      auto buf = op->match_buffers[i]->buffer;
      auto buffer_path = match_path->ArrayIndex(i)->Attr("buffer");
      context.push_back(WithDef(buf->data, buffer_path->Attr("data")));
      context.push_back(WithDef(buf, buffer_path));
    }
  }

  Visit(op->init, path->Attr("init"));
  Visit(op->body, path->Attr("body"));

  while (context.size()) context.pop_back();
}

void TIRVisitorWithPath::VisitStmt_(const BlockRealizeNode* op, ObjectPath path) {
  Visit(op->iter_values, path->Attr("iter_values"));
  Visit(op->predicate, path->Attr("predicate"));
  Visit(op->block, path->Attr("block"));
}

void TIRVisitorWithPath::VisitExpr_(const VarNode* op, ObjectPath path) {}

void TIRVisitorWithPath::VisitExpr_(const SizeVarNode* op, ObjectPath path) {
  VisitExpr_(static_cast<const VarNode*>(op), path);
}

void TIRVisitorWithPath::VisitExpr_(const AnyNode* op, ObjectPath path) {}

void TIRVisitorWithPath::VisitExpr_(const BufferLoadNode* op, ObjectPath path) {
  Visit(op->buffer, path->Attr("buffer"));
  Visit(op->indices, path->Attr("indices"));
}

void TIRVisitorWithPath::VisitExpr_(const ProducerLoadNode* op, ObjectPath path) {
  Visit(op->indices, path->Attr("indices"));
}

void TIRVisitorWithPath::VisitExpr_(const LetNode* op, ObjectPath path) {
  Visit(op->value, path->Attr("value"));
  auto context = WithDef(op->var, path->Attr("var"));
  Visit(op->body, path->Attr("body"));
}

void TIRVisitorWithPath::VisitExpr_(const CallNode* op, ObjectPath path) {
  if (auto gvar = op->op.as<GlobalVar>()) {
    VisitGlobalVar(gvar.value(), path->Attr("op"));
  }
  Visit(op->args, path->Attr("args"));
}

#define DEFINE_BINOP_VISIT_(OP)                                        \
  void TIRVisitorWithPath::VisitExpr_(const OP* op, ObjectPath path) { \
    Visit(op->a, path->Attr("a"));                                     \
    Visit(op->b, path->Attr("b"));                                     \
  }

DEFINE_BINOP_VISIT_(AddNode);
DEFINE_BINOP_VISIT_(SubNode);
DEFINE_BINOP_VISIT_(MulNode);
DEFINE_BINOP_VISIT_(DivNode);
DEFINE_BINOP_VISIT_(ModNode);
DEFINE_BINOP_VISIT_(FloorDivNode);
DEFINE_BINOP_VISIT_(FloorModNode);
DEFINE_BINOP_VISIT_(MinNode);
DEFINE_BINOP_VISIT_(MaxNode);
DEFINE_BINOP_VISIT_(EQNode);
DEFINE_BINOP_VISIT_(NENode);
DEFINE_BINOP_VISIT_(LTNode);
DEFINE_BINOP_VISIT_(LENode);
DEFINE_BINOP_VISIT_(GTNode);
DEFINE_BINOP_VISIT_(GENode);
DEFINE_BINOP_VISIT_(AndNode);
DEFINE_BINOP_VISIT_(OrNode);

#undef DEFINE_BINOP_VISIT_

void TIRVisitorWithPath::VisitExpr_(const IntImmNode* op, ObjectPath path) {}
void TIRVisitorWithPath::VisitExpr_(const FloatImmNode* op, ObjectPath path) {}
void TIRVisitorWithPath::VisitExpr_(const StringImmNode* op, ObjectPath path) {}

void TIRVisitorWithPath::VisitExpr_(const ReduceNode* op, ObjectPath path) {
  Visit(op->axis, path->Attr("axis"));
  Visit(op->source, path->Attr("source"));
  Visit(op->init, path->Attr("init"));
  Visit(op->condition, path->Attr("condition"));
}

void TIRVisitorWithPath::VisitExpr_(const CastNode* op, ObjectPath path) {
  Visit(op->value, path->Attr("value"));
}

void TIRVisitorWithPath::VisitExpr_(const NotNode* op, ObjectPath path) {
  Visit(op->a, path->Attr("a"));
}

void TIRVisitorWithPath::VisitExpr_(const SelectNode* op, ObjectPath path) {
  Visit(op->condition, path->Attr("condition"));
  Visit(op->true_value, path->Attr("true_value"));
  Visit(op->false_value, path->Attr("false_value"));
}

void TIRVisitorWithPath::VisitExpr_(const RampNode* op, ObjectPath path) {
  Visit(op->base, path->Attr("base"));
  Visit(op->stride, path->Attr("stride"));
}

void TIRVisitorWithPath::VisitExpr_(const ShuffleNode* op, ObjectPath path) {
  Visit(op->indices, path->Attr("indices"));
  Visit(op->vectors, path->Attr("vectors"));
}

void TIRVisitorWithPath::VisitExpr_(const BroadcastNode* op, ObjectPath path) {
  Visit(op->value, path->Attr("value"));
}

template <typename DerivedVerifier>
class Verifier : protected TIRVisitorWithPath {
 public:
  template <typename TirNodeRef>
  static bool Verify(const TirNodeRef& node, bool assert_on_error) {
    DerivedVerifier verifier(assert_on_error);
    verifier(node);
    return !verifier.has_error_;
  }

 protected:
  Verifier(bool assert_on_error) : assert_on_error_(assert_on_error) {}

  /* \brief Helper class to handle the bool-or-assert handles
   *
   * Each verifier can either return a boolean, or assert on failure.
   * To avoid needing to duplicate this logic at every step, the
   * Verify() method can be used.  Similar to `LOG(FATAL)` or
   * `LOG(DEBUG)`, it returns an object that can accept streamed
   * context information.
   *
   * If the error should be raised, then the context is collected
   * identically to `LOG(FATAL)`.  If a boolean is returned, or if the
   * condition passes, then the streamed context is discarded.
   *
   * Usage:
   *
   *     Verify(value == expected_value)
   *            << "ValueError: " << value
   *            << " was not the expected value of " << expected_value;
   */
  class VerifyStream {
   public:
    VerifyStream(bool log_fatal) {
      if (log_fatal) {
        log_.emplace();
      }
    }

    VerifyStream(const VerifyStream&) = delete;
    VerifyStream& operator=(const VerifyStream&) = delete;
    VerifyStream(VerifyStream&& other) { std::swap(log_, other.log_); }
    VerifyStream& operator=(VerifyStream&& other) {
      std::swap(log_, other.log_);
      return *this;
    }

    template <typename T>
    VerifyStream& operator<<(T&& t) {
      if (log_.has_value()) {
        log_.value() << std::forward<T>(t);
      }
      return *this;
    }

    ~VerifyStream() noexcept(false) {
      if (log_.has_value()) {
        LOG(FATAL) << log_->str();
      }
    }

    std::optional<std::ostringstream> log_{std::nullopt};
  };

  // TODO(Lunderberg): Add the filename/linenum with
  // std::source_location when C++20 is available.
  VerifyStream Verify(bool condition) {
    has_error_ = has_error_ || !condition;
    return VerifyStream(!condition && assert_on_error_);
  }

  bool assert_on_error_;
  bool has_error_{false};
};

}  // namespace

/*! \brief Verify all Expr inside the block does not contain:
 *    1. loop vars outside the current block.
 *    2. block vars of parent blocks.
 */
class BlockVarAccessVerifier : public StmtExprVisitor {
 public:
  static bool Verify(const PrimFunc& func, bool assert_mode) {
    BlockVarAccessVerifier verifier(assert_mode);
    verifier(func->body);
    return !verifier.has_error_;
  }

 private:
  explicit BlockVarAccessVerifier(bool assert_mode) : assert_mode_(assert_mode) {}

  void VisitStmt(const Stmt& stmt) final {
    if (!has_error_) {
      StmtExprVisitor::VisitStmt(stmt);
    }
  }

  void VisitExpr(const PrimExpr& expr) final {
    if (!has_error_) {
      StmtExprVisitor::VisitExpr(expr);
    }
  }

  void VisitExpr_(const VarNode* op) final {
    auto it = loop_vars_.find(op);
    if (it != loop_vars_.end() && it->second < block_stack_.size()) {
      has_error_ = true;
      if (assert_mode_) {
        if (it->second == 0) {
          LOG(FATAL) << "Well-formedness check failed: "
                     << "Loop iterator var " << op->name_hint
                     << " is defined outside of any block, "
                     << "but is used inside the non-opaque current block \""
                     << block_stack_.back()->name_hint << "\".";
        } else {
          LOG(FATAL) << "Well-formedness check failed: "
                     << "Loop iterator var " << op->name_hint << " is defined in block \""
                     << block_stack_[it->second - 1]->name_hint << "\", "
                     << "but is used inside the non-opaque current block \""
                     << block_stack_.back()->name_hint << "\".";
        }
      }
    }
  }

  void VisitStmt_(const ForNode* op) final {
    ICHECK(loop_vars_.find(op->loop_var.get()) == loop_vars_.end());
    loop_vars_[op->loop_var.get()] = block_stack_.size();
    StmtExprVisitor::VisitStmt_(op);
    loop_vars_.erase(op->loop_var.get());
  }

  void VisitStmt_(const BlockNode* op) final {
    // Do not check boundary if it's a opaque block.
    bool is_non_opaque = op->iter_vars.size();
    if (is_non_opaque) {
      block_stack_.push_back(op);
    }

    // Step 0. Skip block iter var's domain

    // Step 1. Visit read/write regions
    auto fvisit_buffer_region = [this](const BufferRegion& s) {
      for (const auto& range : s->region) {
        this->VisitExpr(range->min);
        this->VisitExpr(range->extent);
      }
    };
    VisitArray(op->reads, fvisit_buffer_region);
    VisitArray(op->writes, fvisit_buffer_region);

    // Step 2. Visit match buffers
    VisitArray(op->match_buffers,
               [fvisit_buffer_region](const MatchBufferRegion& match_buffer_region) {
                 fvisit_buffer_region(match_buffer_region->source);
               });

    // Step 3. Visit init and body
    if (op->init.defined()) {
      this->VisitStmt(op->init.value());
    }
    this->VisitStmt(op->body);

    if (is_non_opaque) {
      block_stack_.pop_back();
    }
  }

 private:
  /*! \brief The map from outside loop vars to its corresponding block level. */
  std::unordered_map<const VarNode*, size_t> loop_vars_;
  /*! \brief Whether it's in assert mode. */
  bool assert_mode_;
  /*! \brief Current nested block stack level. */
  std::vector<const BlockNode*> block_stack_;
  /*! \brief Whether there is error. */
  bool has_error_{false};
};

class UndefinedVarVerifier : public Verifier<UndefinedVarVerifier> {
 public:
  // Until templated-this arrives in C++23, the CRTP can't inject a
  // constructor into the child class.  Therefore, must explicitly add
  // the constructor.
  using Verifier::Verifier;

 private:
  void EnterDef(const Var& var, ObjectPath path) override {
    LOG(DEBUG) << "Entering definition of " << var;
    {
      auto it = currently_defined_.find(var);
      Verify(it == currently_defined_.end())
          << "ValueError: "
          << "TIR is ill-formed, "
          << "due to duplicate nested definitions of variable " << var
          << ".  It was first defined at " << it->second << ", and was re-defined at " << path;
    }

    {
      auto it = currently_defined_.find(var);
      Verify(it == currently_defined_.end())
          << "ValueError: "
          << "TIR is ill-formed, "
          << "due to duplicate sequential definitions of variable " << var
          << ".  It was first defined at " << it->second << ", and was later re-defined at "
          << path;
    }

    currently_defined_.insert({var, path});
  }

  void ExitDef(const Var& var, ObjectPath path) override {
    LOG(DEBUG) << "Exiting definition of " << var;
    auto active_def = currently_defined_.find(var);
    auto prev_def = previously_defined_.find(var);

    Verify(prev_def == previously_defined_.end())
        << "Unmatched EnterDef/ExitDef for variable " << var << ", defined at " << path
        << " but previously defined at " << prev_def->second;

    Verify(active_def != currently_defined_.end())
        << "Unmatched EnterDef/ExitDef for variable " << var << ".  "
        << "ExitDef occurred at " << path << ", but no corresponding EnterDef.";
    Verify(active_def->second == path)
        << "Unmatched EnterDef/ExitDef for variable " << var << ", defined at "
        << active_def->second << ", but undefined at " << path;

    currently_defined_.erase(active_def);
    previously_defined_.insert({var, path});
  }

  void VisitExpr_(const VarNode* op, ObjectPath path) override {
    auto var = GetRef<Var>(op);

    auto prev_def = previously_defined_.find(var);
    Verify(prev_def == previously_defined_.end())
        << "ValueError: "
        << "Invalid use of variable " << var << " was used at " << path << ".  "
        << "While this variable was previously defined at " << prev_def->second
        << ", this definition is no longer in-scope.";

    auto active_def = currently_defined_.find(var);
    Verify(active_def != currently_defined_.end())
        << "ValueError: "
        << "Invalid use of undefined variable " << var << " at " << path;
  }

  std::unordered_map<Var, ObjectPath, ObjectPtrHash, ObjectPtrEqual> currently_defined_;
  std::unordered_map<Var, ObjectPath, ObjectPtrHash, ObjectPtrEqual> previously_defined_;
};

bool VerifyWellFormed(const PrimFunc& func, bool assert_mode) {
  if (!BlockVarAccessVerifier::Verify(func, assert_mode)) {
    return false;
  }
  // TODO(Siyuan): add more checks here.
  return true;
}

bool VerifyWellFormed(const IRModule& mod, bool assert_mode) {
  for (const auto& [gvar, base_func] : mod->functions) {
    if (auto prim_func = base_func.as<PrimFunc>()) {
      bool res = VerifyWellFormed(prim_func.value(), assert_mode);
      if (!res) {
        return false;
      }
    }
  }

  if (!UndefinedVarVerifier::Verify(mod, assert_mode)) return false;

  return true;
}

TVM_REGISTER_GLOBAL("tir.analysis.VerifyWellFormed")
    .set_body_typed([](const ObjectRef& obj, bool assert_mode) {
      if (auto opt = obj.as<PrimFunc>()) {
        return VerifyWellFormed(opt.value(), assert_mode);
      } else if (auto opt = obj.as<IRModule>()) {
        return VerifyWellFormed(opt.value(), assert_mode);
      } else {
        LOG(FATAL) << "Expected VerifyWellFormed argument to be a PrimFunc or IRModule, but found "
                   << obj->GetTypeKey();
      }
    });

}  // namespace tir
}  // namespace tvm
