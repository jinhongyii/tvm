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

#include <tvm/relax/attrs/memory.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include "../../printer/text_printer.h"
#include "../../relay/transforms/pattern_utils.h"

namespace tvm {
namespace relax {


class LayoutRewriteBlockFinder: public tir::StmtExprVisitor{
    public:
    static String FindLayoutRewriteBlock(const tir::PrimFunc& func) {
        LayoutRewriteBlockFinder finder;
        finder(func->body);
        return finder.block_name;
    }

    private:
    void VisitStmt_(const tir::BlockNode* block) final {
        if (block->annotations.count("from_preproc")) {
            if(block_name!=""){
                LOG(FATAL) << "Found multiple layout rewrite blocks";
            }
            block_name = block->name_hint;
            return;
        }
        StmtExprVisitor::VisitStmt_(block);
    }

    String block_name;
};

class InlineMutator : public ExprMutator {
 public:
  

  static IRModule Transform(IRModule mod){
    InlineMutator mutator(mod);
    for (const auto& kv : mod->functions) {
      const GlobalVar& gv = kv.first;
      const BaseFunc& func = kv.second;
      if (kv.second.as<relax::FunctionNode>()) {
        auto updated_func = Downcast<Function>(mutator(func));
        mutator.builder_->UpdateFunction(gv, updated_func);
      }
    }
    mod = mutator.builder_->GetContextIRModule();
    return mod;
  }
 private:

  explicit InlineMutator(const IRModule& mod): mod_(mod) {}

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
  Array<Expr> GetTIRArgs(const CallNode* call) {
    static const Op& call_tir_op = Op::Get("relax.call_tir");
    ICHECK_EQ(call->op, call_tir_op);
    Array<Expr> ret;
    if (call->args[1].as<TupleNode>()) {
      ret = Downcast<Tuple>(call->args[1])->fields;
    } else {
      ret = {call->args[1]};
    }
    return ret;
  }

  Expr VisitExpr_(const CallNode* call) final {
    // post-order mutation
    Expr expr = ExprMutator::VisitExpr_(call);
    call = expr.as<CallNode>();

    static const Op& call_tir_op = Op::Get("relax.call_tir");
    Array<Expr> args = call->args;
    if (call->op == call_tir_op) {
      GlobalVar gvar = Downcast<GlobalVar>(args[0]);
      Optional<tir::PrimFunc> opt_f = MatchPrimFunc(gvar);
      if (opt_f.defined()) {
        tir::PrimFunc f = opt_f.value();
        String block_name = LayoutRewriteBlockFinder::FindLayoutRewriteBlock(f);
        IRModule mod({{GlobalVar("main"), f}});
        tir::Schedule sch =
            tir::Schedule::Concrete(mod, 0, 0, tir::ScheduleErrorRenderLevel::kNone);
        try {
          sch->ReverseComputeInline(sch->GetBlock(block_name));
          tir::PrimFunc new_f = Downcast<tir::PrimFunc>(sch->mod()->Lookup("main"));
          const GlobalVar& new_f_var = builder_->AddFunction(new_f, gvar->name_hint);

          args.Set(0, new_f_var);
          return Call(call->op, args, call->attrs, call->type_args);
        } catch (const dmlc::Error& e) {
          LOG(INFO) << "Failed to inline layout rewrite block: " << e.what();
          builder_->UpdateFunction(gvar, f);
        }
      }
    }

    return GetRef<Expr>(call);
  }

  const IRModule& mod_;
};

namespace transform {

Pass ComputeInlineLayoutRewrite() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =  //
      [=](IRModule m, PassContext pc) { return InlineMutator::Transform(m); };
  return CreateModulePass(/*pass_function=*/pass_func,                //
                          /*opt_level=*/0,                            //
                          /*pass_name=*/"ComputeInlineLayoutRewrite",  //
                          /*required=*/{});
}

TVM_REGISTER_GLOBAL("relax.transform.ComputeInlineLayoutRewrite").set_body_typed(ComputeInlineLayoutRewrite);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
