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
 * \file tvm/relax/distributed/transform/lower_distir.cc
 * \brief Pass for lowering DistIR into Relax
 *  This pass assumes all the TensorIR functions are in local view, 
 *  so the pass only handles sharding tensor shape.
 */

#include <tvm/relax/distributed/axis_group_graph.h>
#include <tvm/relax/distributed/transform.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/attrs/ccl.h>
#include <tvm/tir/stmt_functor.h>
#include "../../../tir/schedule/transform.h"
#include "../../op/ccl/ccl.h"
#include "utils.h"

namespace tvm{
namespace relax{
namespace distributed{

class DistIRSharder : public ExprMutator {
public:

  static IRModule LowerDistIR(IRModule mod) {
    return DistIRSharder(mod).Lower();
  }


private:
  DistIRSharder(IRModule mod): ExprMutator(mod) {}

  IRModule Lower(){
    auto mod = builder_->GetContextIRModule();
    for (const auto& [gv, base_func] : mod->functions) {
      const auto* func_ = base_func.as<FunctionNode>();
      if (func_ == nullptr) {
        continue;
      }
      Function func = RewriteFunction(GetRef<Function>(func_));
      builder_->UpdateFunction(gv, func);
    }
    return builder_->GetContextIRModule();
  }

  ShapeExpr ShardShape(ShapeExpr orig_shape, DeviceMesh device_mesh, Placement placement){
    ShapeTuple device_mesh_shape = device_mesh->shape;
    Array<PrimExpr> new_tensor_shape_value = orig_shape->values;
    for (int i = 0; i < device_mesh_shape.size(); i++) {
      if(placement->dim_specs[i]->kind == PlacementSpecKind::kSharding){
        int shard_size = device_mesh_shape[i];
        int axis = placement->dim_specs[i]->axis;
        new_tensor_shape_value.Set(axis, floordiv(orig_shape->values[axis], shard_size));
      }
    }
    return ShapeExpr(new_tensor_shape_value);
  }

  TensorStructInfo ShardSinfo(DTensorStructInfo orig_sinfo){
    TensorStructInfo tensor_sinfo = orig_sinfo->tensor_sinfo;
    ICHECK(tensor_sinfo->shape);
    const auto* orig_shape = tensor_sinfo->shape.as<ShapeExprNode>();
    auto new_tensor_sinfo = make_object<TensorStructInfoNode>(*tensor_sinfo.get());
    new_tensor_sinfo->shape = ShardShape(GetRef<ShapeExpr>(orig_shape), orig_sinfo->device_mesh, orig_sinfo->placement);
    return TensorStructInfo(new_tensor_sinfo);
  }

  Expr ShardInputParamTensorAndConstant(Expr input){
    const auto* sinfo = GetStructInfoAs<DTensorStructInfoNode>(input);
    if(!sinfo){
      return input;
    }
    TensorStructInfo new_tensor_sinfo = ShardSinfo(GetRef<DTensorStructInfo>(sinfo));
    if (const auto* var = input.as<VarNode>()) {
      Var new_param(var->name_hint(), new_tensor_sinfo);
      return new_param;
    } else if (const auto* constant = input.as<ConstantNode>()) {
      for(const auto& spec: sinfo->placement->dim_specs){
        ICHECK(spec->kind == PlacementSpecKind::kReplica);
      }
      Constant new_constant(constant->data, new_tensor_sinfo);
      return new_constant;
    } else {
      LOG(FATAL) << "Cannot shard tensor which is not Var or Constant: " << input;
      throw;
    }
  }

  void InputPreprocessing(Function func) {
    Optional<Integer> num_inputs = func->GetAttr<Integer>("num_input");
    if(!num_inputs.defined()){
      return;
    }
    for (int i = 0; i < num_inputs.value()->value; i++) {
      auto sinfo = GetStructInfoAs<DTensorStructInfoNode>(func->params[i]);
      for(const auto& dim_spec: sinfo->placement->dim_specs){
        ICHECK(dim_spec->kind == PlacementSpecKind::kReplica)
            << "Input tensor sharding is not supported now";
      }
      Var old_input = param_tensor_remap_.at(func->params[i]);
      input_preprocessing_.Set(broadcast_from_worker0(old_input), func->params[i]);
    }
  }

  Function RewriteFunction(Function func){
    input_tensor_remap_.clear();
    param_tensor_remap_.clear();
    input_preprocessing_.clear();
    Array<Var> new_params;
    for (const Var& var : func->params) {
      if (GetStructInfoAs<DTensorStructInfoNode>(var)) {
        Var new_param = Downcast<Var>(ShardInputParamTensorAndConstant(var));
        param_tensor_remap_.Set(var, new_param);
        new_params.push_back(new_param);
      } else {
        new_params.push_back(var);
      }
    }
    InputPreprocessing(func);
    auto new_body = VisitWithNewScope(func->body, new_params);
    Function new_func(new_params, new_body, NullOpt, func->is_pure, func->attrs);
    return new_func;
  }

  BindingBlock VisitBindingBlock_(const BindingBlockNode* block) {
    builder_->BeginBindingBlock();
    for(const auto& pr: input_preprocessing_){
      Var new_var = builder_->Emit(pr.first, "broadcast");
      input_tensor_remap_.Set(pr.second, new_var);
    }
    for (Binding binding : block->bindings) {
      this->VisitBinding(binding);
    }
    return builder_->EndBlock();
  }

  BindingBlock VisitBindingBlock_(const DataflowBlockNode* block) {
    builder_->BeginDataflowBlock();
    for(const auto& pr: input_preprocessing_){
      Var new_var = builder_->Emit(pr.first, "broadcast");
      input_tensor_remap_.Set(pr.second, new_var);
    }
    for (auto binding : block->bindings) {
      this->VisitBinding(binding);
    }
    return builder_->EndBlock();
  }

  Expr VisitExpr_(const VarNode* var) final {
    auto it = input_tensor_remap_.find(GetRef<Var>(var));
    if (it != input_tensor_remap_.end()) {
      return (*it).second;
    }
    it = param_tensor_remap_.find(GetRef<Var>(var));
    if (it != param_tensor_remap_.end()) {
      return (*it).second;
    }
    return ExprMutator::VisitExpr_(var);
  }

  Call HandleSpecialCaseinDTensorLowering(const CallNode* call, Var binding_var){
    static Op reshape_op = Op::Get("relax.reshape");
    static Op call_tir_op = Op::Get("relax.call_tir");
    static Op call_tir_local_view_op = Op::Get("relax.dist.call_tir_local_view");
    if (call->op.same_as(reshape_op)) {
      ICHECK(call->args[1].as<ShapeExprNode>());
      const auto* out_sinfo = GetStructInfoAs<DTensorStructInfoNode>(binding_var);
      auto new_call_node = make_object<CallNode>(*call);
      new_call_node->args.Set(1, ShardShape(Downcast<ShapeExpr>(call->args[1]), out_sinfo->device_mesh, out_sinfo->placement));
      return Call(new_call_node);
    } else if(call->op.same_as(call_tir_local_view_op)){
      auto new_call_node = make_object<CallNode>(*call);
      new_call_node->op = call_tir_op;
      new_call_node->sinfo_args = {ShardSinfo(GetRef<DTensorStructInfo>(GetStructInfoAs<DTensorStructInfoNode>(binding_var)))};
      return Call(new_call_node);
    } else if(call->op.same_as(call_tir_op)){
      LOG(FATAL)<<"call_tir should be lowered to call_tir_local_view before lowering to relax";
    }
    return GetRef<Call>(call);
  }  

  void VisitBinding_(const VarBindingNode* binding, const CallNode* val){
    const auto* sinfo = GetStructInfoAs<DTensorStructInfoNode>(binding->var);
    if (!sinfo) {
      ExprMutator::VisitBinding_(binding, val);
      return;
    }
    Call new_call = Downcast<Call>(this->VisitExpr(HandleSpecialCaseinDTensorLowering(val, binding->var)));
    ReEmitBinding(binding, builder_->Normalize(new_call));
  }

  Map<Var, Var> param_tensor_remap_;
  Map<Expr, Var> input_preprocessing_;
  //todo: broadcast every "R" input
  //todo: for every "S" input, insert shard and scatter directly in the beginning
  //todo: postpone broadcast
  //      if the operands are "R" on the device mesh dim of the broadcast, then broadcast be moved across this operator
  //      broadcast can be fused with local scatter(slice) and become scatter_from_worker0
  Map<Var, Var> input_tensor_remap_;
};

namespace transform {

Pass LowerDistIR() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) { return DistIRSharder::LowerDistIR(m); };
  return CreateModulePass(pass_func, 1, "LowerDistIR", {}); 
}
TVM_REGISTER_GLOBAL("relax.distributed.transform.LowerDistIR")
    .set_body_typed(LowerDistIR);
}  // namespace transform

} // namespace distributed
} // namespace relax
} // namespace tvm