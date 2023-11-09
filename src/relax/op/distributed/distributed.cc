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
 * \file distributed.cc
 * \brief Redistribute operator.
 */

#include "distributed.h"
#include "../ccl/ccl.h"
#include <tvm/topi/einsum.h>
#include <tvm/relax/attrs/ccl.h>

#include <algorithm>
#include <utility>
#include <vector>

namespace tvm {
namespace relax {

/* relax.dist.annotate_sharding */
TVM_REGISTER_NODE_TYPE(DistributionAttrs);
Expr annotate_sharding(Expr input, distributed::DeviceMesh device_mesh,
                       distributed::Placement placement) {
  ObjectPtr<DistributionAttrs> attrs = make_object<DistributionAttrs>();
  attrs->device_mesh = device_mesh;
  attrs->placement = placement;

  static const Op& op = Op::Get("relax.dist.annotate_sharding");
  return Call(op, {std::move(input)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.dist.annotate_sharding").set_body_typed(annotate_sharding);

StructInfo InferStructInfoAnnotateSharding(const Call& call, const BlockBuilder& ctx) {
  return GetStructInfo(call->args[0]);
}

TVM_REGISTER_OP("relax.dist.annotate_sharding")
    .set_num_inputs(1)
    .add_argument("input", "Tensor", "The input tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoAnnotateSharding)
    .set_attr<FInferStructInfo>("dist.FInferStructInfo", InferStructInfoAnnotateSharding)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.dist.redistribute */
TVM_REGISTER_NODE_TYPE(DistributionAttrs);
Expr redistribute(Expr input, distributed::DeviceMesh device_mesh,
                  distributed::Placement placement) {
  ObjectPtr<DistributionAttrs> attrs = make_object<DistributionAttrs>();
  attrs->device_mesh = device_mesh;
  attrs->placement = placement;

  static const Op& op = Op::Get("relax.dist.redistribute");
  return Call(op, {std::move(input)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.dist.redistribute").set_body_typed(redistribute);

StructInfo InferDistStructInfoRedistribute(const Call& call, const BlockBuilder& ctx) {
  const auto* attrs = call->attrs.as<DistributionAttrs>();
  const auto* sinfo = GetStructInfoAs<distributed::DTensorStructInfoNode>(call->args[0]);
  ICHECK(sinfo);
  return distributed::DTensorStructInfo(sinfo->tensor_sinfo, attrs->device_mesh, attrs->placement);
}

TVM_REGISTER_OP("relax.dist.redistribute")
    .set_num_inputs(1)
    .add_argument("input", "Tensor", "The input tensor.")
    .set_attr<FInferStructInfo>("dist.FInferStructInfo", InferDistStructInfoRedistribute)
    .set_attr<Bool>("FPurity", Bool(true));

StructInfo InferStructInfoCallTIRLocalView(const Call& call, const BlockBuilder& ctx) {
  if (call->sinfo_args.size() != 1) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "sinfo_args should have exactly 1 output struct info.");
  }
  CHECK(call->args[0]->IsInstance<GlobalVarNode>())
      << "call_tir_local_view expects the first argument to be a GlobalVar referring to a TIR PrimFunc. "
      << "However, gets " << call->args[0];
  return call->sinfo_args[0];
}

RELAY_REGISTER_OP("relax.dist.call_tir_local_view")
    .set_num_inputs(3)
    .add_argument("func", "Expr", "The destination-passing-style function.")
    .add_argument("args", "Tuple", "The input arguments.")
    .add_argument("packed_ints", "Expr",
                  "ShapeExpr representing a tuple of ints to unpack during runtime. Omitted from "
                  "args if unused")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoCallTIRLocalView)
    .set_attr<Bool>("FPurity", Bool(true));

Expr MakeCallTIRLocalView(Expr func, Tuple args, Array<distributed::DTensorStructInfo> out_sinfo_list,
                 Optional<Expr> packed_ints) {
  for (const distributed::DTensorStructInfo& sinfo : out_sinfo_list) {
    const auto* shape = sinfo->tensor_sinfo->shape.as<ShapeExprNode>();
    CHECK(shape != nullptr) << "out_sinfo of call_tir_local_view should have defined ShapeExpr as shape. "
                               "However, one given structure info is "
                            << sinfo;
  }

  StructInfo out_sinfo{nullptr};
  if (out_sinfo_list.size() == 1) {
    out_sinfo = out_sinfo_list[0];
  } else {
    out_sinfo = TupleStructInfo({out_sinfo_list.begin(), out_sinfo_list.end()});
  }

  static const Op& op = Op::Get("relax.dist.call_tir_local_view");
  Call call;
  if (!packed_ints) {
    // don't use additional optional argument
    call = Call(op, {func, args}, {}, {out_sinfo});
  } else {
    call = Call(op, {func, args, packed_ints.value()}, {}, {out_sinfo});
  }
  return call;
}

TVM_REGISTER_GLOBAL("relax.op.dist.call_tir_local_view").set_body_typed(MakeCallTIRLocalView);

}  // namespace relax
}  // namespace tvm
