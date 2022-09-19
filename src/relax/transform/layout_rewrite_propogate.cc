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

#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {

namespace tir{
    IndexMap ExtractIndexMap(const PrimFunc& func){
        struct BufferIndexFinder: public StmtExprVisitor{

            void VisitStmt_(const BlockNode* op) final{
              StmtVisitor::VisitStmt_(op);
              for(const auto& iter: op->iter_vars){
                initial_ranges_.push_back(iter->dom);
              }
            }

            void VisitStmt_(const BufferStoreNode* op) final {
              ExprVisitor::VisitExpr(op->value);
              if(need_inverse_){
                for(const PrimExpr& e: op->indices){
                  initial_indices_.push_back(Downcast<Var>(e));
                }
              }else{
                final_indices_ = op->indices;
              }
            }

             void VisitExpr_(const BufferLoadNode* op) final { 
                for (const PrimExpr& e: op->indices){
                  if(!e.as<VarNode>()){
                    need_inverse_ = true;
                    final_indices_ = op->indices;
                    initial_indices_.clear();
                    return;
                  }else{
                    initial_indices_.push_back(Downcast<Var>(e));
                  }
                } 
             }

             Array<Var> initial_indices_;
             Array<PrimExpr> final_indices_;
             Array<Range> initial_ranges_;
             bool need_inverse_ = false;
        };
        BufferIndexFinder finder;
        finder(func->body);
        if(finder.need_inverse_){
          return IndexMap(finder.initial_indices_, finder.final_indices_).Inverse(finder.initial_ranges_);
        }else{
          return IndexMap(finder.initial_indices_, finder.final_indices_);
        }
    }

    PrimFunc CreateLayoutRewriteFunc(const IndexMap& index_map, const Buffer& src_buffer, const Buffer& dst_buffer){
      Var src("src", DataType::Handle());
      Var dst("dst", DataType::Handle());
      Array<PrimExpr> load_indices;
      for(const auto& var: index_map->initial_indices){
        load_indices.push_back(var);
      }
      Stmt body = BufferStore(dst_buffer, BufferLoad(src_buffer, load_indices), index_map->final_indices);
      Array<IterVar> loop_vars_iter;
      for(int i = 0; i < index_map->initial_indices.size(); i++){
        loop_vars_iter.push_back(IterVar(Range::FromMinExtent(0, src_buffer->shape[i]), index_map->initial_indices[i], kDataPar));
      }
      body = Block(loop_vars_iter, {}, {}, "layout_rewrite", body, NullOpt, {}, {},
                   {{tir::attr::script_parsing_detect_access, Integer(3)}});
      Array<PrimExpr> loop_vars;
      for(int i = 0; i < index_map->initial_indices.size(); i++){
        loop_vars.push_back(index_map->initial_indices[i].copy_with_suffix("_o"));
      }
      body = BlockRealize(loop_vars, Bool(true), Downcast<Block>(body));
      ICHECK_EQ(src_buffer->shape.size(), index_map->initial_indices.size());
      for (int i = static_cast<int>(index_map->initial_indices.size()) - 1; i >= 0;i--){
        body = For(Downcast<Var>(loop_vars[i]), 0, src_buffer->shape[i], ForKind::kSerial, body);
      }
      body = Block({}, {}, {}, "root", body);
      body = BlockRealize({}, Bool(true), Downcast<Block>(body));
      Map<Var, Buffer> buffer_map{{src, src_buffer}, {dst, dst_buffer}};
      PrimFunc func = PrimFunc({src, dst}, std::move(body), VoidType(), buffer_map);
      return func;
    }

    PrimFunc FuseLayoutRewriteFuncs(const PrimFunc& func1, const PrimFunc& func2){
        IndexMap index_map1 = ExtractIndexMap(func1);
        IndexMap index_map2 = ExtractIndexMap(func2);
        ICHECK_EQ(index_map1->final_indices.size(), index_map2->initial_indices.size())
            << func1 << std::endl << func2;
        IndexMap final_index_map = index_map1.ComposeIndexMap(index_map2);
        Buffer src_buffer = func1->buffer_map[func1->params[0]];
        Buffer dst_buffer = func2->buffer_map[func2->params[1]];
        return CreateLayoutRewriteFunc(final_index_map, src_buffer, dst_buffer);
    }
} // namespace tir
namespace relax {


class Propogator : public ExprMutator {
  public:
  static IRModule Transform(IRModule mod){
    Propogator propogator(mod);
    for (const auto& kv : mod->functions) {
      const GlobalVar& gv = kv.first;
      const BaseFunc& func = kv.second;
      if (kv.second.as<relax::FunctionNode>()) {
        auto updated_func = Downcast<Function>(propogator(func));
        propogator.builder_->UpdateFunction(gv, updated_func);
      } else if (kv.second.as<tir::PrimFuncNode>()){
        propogator.builder_->UpdateFunction(gv, func);
      }
    }
    return propogator.builder_->GetContextIRModule();
  }
    private:
    explicit Propogator(const IRModule& mod) : mod_(mod) {}
    /*!
    * \brief Pattern match op to a TIR function and look it up.
    * \return The TIR function, or NullOpt if patter match fails.
    */
    Optional<tir::PrimFunc> MatchPrimFunc(const GlobalVar& gv) const {
        Optional<BaseFunc> base_func = mod_->functions.Get(gv);
        if (auto* pfunc = base_func.as<tir::PrimFuncNode>()) {
        return GetRef<tir::PrimFunc>(pfunc);
        } else {
        return NullOpt;
        }
    }

    Expr VisitExpr_(const CallNode* call) final {
        static const Op& call_tir_op = Op::Get("relax.call_tir");
        if (call->op == call_tir_op) {
        return VisitCallTIR(call);
        } else {
        return GetRef<Expr>(call);
        }
    }

    Expr GetCallArgForLayoutRewrite(Expr call_tir_arg){
        if(const auto* tuple = call_tir_arg.as<TupleNode>()){
            ICHECK(tuple->fields.size() == 1) << call_tir_arg;
            return tuple->fields[0];
        }else{
            return call_tir_arg;
        }
    }
    
    Array<Expr> GetTupleArgs(Expr call_tir_arg){
        if(const auto* tuple = call_tir_arg.as<TupleNode>()){
            return tuple->fields;
        }else{
            return {call_tir_arg};
        }
    }

    Expr VisitCallTIR(const CallNode* call) {
      Call new_call = Downcast<Call>(ExprMutator::VisitExpr_(call));
      static const Op& call_tir_op = Op::Get("relax.call_tir");
      ICHECK(new_call->op == call_tir_op);

      // Step 1. Get PrimFunc
      Optional<tir::PrimFunc> opt_f = MatchPrimFunc(Downcast<GlobalVar>(new_call->args[0]));
      CHECK(opt_f.defined()) << "Cannot find PrimFuncs used in call_tir";
      const tir::PrimFunc& f = opt_f.value();
      if(f->HasNonzeroAttr("layout_rewrite")){
        Array<Expr> tuple_args = GetTupleArgs(new_call->args[1]);
        ICHECK_EQ(tuple_args.size(), 1);                                                                                                                                                                                                                                                                                
        Expr arg = tuple_args[0];
        if (last_unresolved_layout_rewrite_.count(arg)) {
          Call last_call = last_unresolved_layout_rewrite_[arg];
          last_unresolved_layout_rewrite_.erase(arg);
          tir::PrimFunc last_func =
              MatchPrimFunc(Downcast<GlobalVar>(last_call->args[0]))
                  .value();
          tir::PrimFunc new_func = tir::FuseLayoutRewriteFuncs(last_func, f);
          GlobalVar new_func_var = builder_->AddFunction(new_func, "fused_layout_rewrite");
          new_call = Call(call_tir_op, {new_func_var, last_call->args[1], new_call->args[2]}, new_call->attrs, new_call->type_args);
          return new_call;
        } else {
          last_unresolved_layout_rewrite_.Set(new_call, new_call);
          return new_call;
        }
      } else {
        Array<Expr> tuple_args = GetTupleArgs(new_call->args[1]);
        for(const auto& arg : tuple_args){
          if (last_unresolved_layout_rewrite_.count(arg)) {
            builder_->Emit(VarBinding(Downcast<Var>(arg), last_unresolved_layout_rewrite_[arg]));
            last_unresolved_layout_rewrite_.erase(arg);
          }
        }
        return new_call;
      }
    }

    void VisitBinding_(const VarBindingNode* binding) final {
      Expr new_value = this->VisitExpr(binding->value);
      Var new_var = this->VisitVarDef(binding->var);
      if(last_unresolved_layout_rewrite_.count(new_value)){
        if(new_var.as<DataflowVarNode>()){
          last_unresolved_layout_rewrite_.Set(new_var, last_unresolved_layout_rewrite_[new_value]);
          return;
        } else{
          builder_->Emit(VarBinding(Downcast<Var>(new_value), last_unresolved_layout_rewrite_[new_value]));
        }
      }
      auto emit = [this](VarBinding b) {
        if (this->builder_->CurrentBlockIsDataFlow() && !b->var.as<DataflowVarNode>()) {
          this->builder_->EmitOutput(b);
        } else {
          this->builder_->Emit(b);
        }
      };

      if (new_var.same_as(binding->var) && new_value.same_as(binding->value)) {
        emit(GetRef<VarBinding>(binding));
        return;
      }

      Var temp = WithShapeAndType(new_var, new_value->shape_, new_value->checked_type_);
      if (!temp.same_as(new_var)) {
        new_var = temp;
        this->var_remap_[binding->var->vid] = new_var;
      }

      emit(VarBinding(new_var, new_value));
    }

    const IRModule& mod_;
    Map<Expr, Call> last_unresolved_layout_rewrite_;


};

namespace transform {

Pass LayoutRewritePropogate() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) { return Propogator::Transform(m); };
  return CreateModulePass(/*pass function*/ pass_func, /*opt level*/ 0,
                          /*pass name*/ "LayoutRewritePropogate",
                          /*required*/ {});
}

TVM_REGISTER_GLOBAL("relax.transform.LayoutRewritePropogate")
    .set_body_typed(LayoutRewritePropogate);

}  // namespace transform
} // namespace relax
} // namespace tvm
