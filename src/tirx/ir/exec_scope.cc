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
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/exec_scope.h>
#include <tvm/tirx/op.h>

#include <queue>

namespace tvm {
namespace tirx {

std::string ScopeKindToString(ScopeKind kind) {
  switch (kind) {
    case ScopeKind::kWorld:
      return "world";
    case ScopeKind::kKernel:
      return "kernel";
    case ScopeKind::kCluster:
      return "cluster";
    case ScopeKind::kCta:
      return "cta";
    case ScopeKind::kWarpgroup:
      return "warpgroup";
    case ScopeKind::kWarp:
      return "warp";
    case ScopeKind::kThread:
      return "thread";
  }
  LOG(FATAL) << "Internal Error: unknown ScopeKind " << static_cast<int>(kind);
}

ScopeKind StringToScopeKind(const ffi::String& name) {
  if (name == "world") return ScopeKind::kWorld;
  if (name == "kernel") return ScopeKind::kKernel;
  if (name == "cluster") return ScopeKind::kCluster;
  if (name == "cta") return ScopeKind::kCta;
  if (name == "warpgroup") return ScopeKind::kWarpgroup;
  if (name == "warp") return ScopeKind::kWarp;
  if (name == "thread") return ScopeKind::kThread;
  LOG(FATAL) << "Unknown scope kind name: " << name;
}

std::pair<ffi::String, ffi::String> ScopeBindingToStringPair(ScopeBinding binding) {
  switch (binding) {
    case ScopeBinding::kKernelCluster:
      return {"kernel", "cluster"};
    case ScopeBinding::kKernelCta:
      return {"kernel", "cta"};
    case ScopeBinding::kClusterCta:
      return {"cluster", "cta"};
    case ScopeBinding::kCtaWarpgroup:
      return {"cta", "warpgroup"};
    case ScopeBinding::kCtaWarp:
      return {"cta", "warp"};
    case ScopeBinding::kWarpgroupWarp:
      return {"warpgroup", "warp"};
    case ScopeBinding::kWarpThread:
      return {"warp", "thread"};
    case ScopeBinding::kCtaThread:
      return {"cta", "thread"};
    case ScopeBinding::kWarpgroupThread:
      return {"warpgroup", "thread"};
    case ScopeBinding::kClusterCtaPair:
      return {"cluster", "cta_pair"};
  }
  LOG(FATAL) << "Internal Error: unknown ScopeBinding " << static_cast<int>(binding);
}

ScopeBinding StringPairToScopeBinding(const ffi::String& parent, const ffi::String& cur) {
  if (parent == "kernel" && cur == "cluster") return ScopeBinding::kKernelCluster;
  if (parent == "kernel" && cur == "cta") return ScopeBinding::kKernelCta;
  if (parent == "cluster" && cur == "cta") return ScopeBinding::kClusterCta;
  if (parent == "cta" && cur == "warpgroup") return ScopeBinding::kCtaWarpgroup;
  if (parent == "cta" && cur == "warp") return ScopeBinding::kCtaWarp;
  if (parent == "warpgroup" && cur == "warp") return ScopeBinding::kWarpgroupWarp;
  if (parent == "warp" && cur == "thread") return ScopeBinding::kWarpThread;
  if (parent == "cta" && cur == "thread") return ScopeBinding::kCtaThread;
  if (parent == "warpgroup" && cur == "thread") return ScopeBinding::kWarpgroupThread;
  if (parent == "cluster" && cur == "cta_pair") return ScopeBinding::kClusterCtaPair;
  LOG(FATAL) << "Unknown scope binding: parent=" << parent << " cur=" << cur;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ExecScopeNode::RegisterReflection();
  ScopeIdDefNode::RegisterReflection();
}

/******** Definition of Execution Scope ********/
bool ScopeNameHigher(const ffi::String& a, const ffi::String& b) {
  return ScopeKindHigher(StringToScopeKind(a), StringToScopeKind(b));
}

ExecScope::ExecScope(ScopeKind kind, ffi::Array<ScopeIdDef> scope_id_def) {
  auto n = ffi::make_object<ExecScopeNode>();
  n->kind = kind;
  n->scope_id_def = std::move(scope_id_def);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.ExecScope", [](ffi::String name) { return ExecScope(name); });
}

// ScopeIdDef
ScopeIdDef::ScopeIdDef(ffi::Array<Var> ids, ffi::Array<PrimExpr> extents, ScopeBinding scope,
                       ffi::Optional<ffi::Array<PrimExpr>> preferred_extents) {
  auto n = ffi::make_object<ScopeIdDefNode>();
  TVM_FFI_ICHECK_EQ(ids.size(), extents.size())
      << "ValueError: Number of dimensions must match, got " << ids.size() << " and "
      << extents.size();
  n->def_ids = std::move(ids);
  n->extents = std::move(extents);
  n->scope = scope;
  n->preferred_extents = std::move(preferred_extents);
  data_ = std::move(n);
}

PrimExpr ScopeIdDef::fused_extent() const {
  TVM_FFI_ICHECK_GT(get()->extents.size(), 0) << "ValueError: Cannot get extent of empty scope";
  PrimExpr ret = get()->extents[0];
  for (size_t i = 1; i < get()->extents.size(); ++i) {
    ret = ret * get()->extents[i];
  }
  return ret;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.ScopeIdDef",
                        [](ffi::Array<Var> vars, ffi::Array<PrimExpr> extents, ffi::String parent,
                           ffi::String cur, ffi::Optional<ffi::Array<PrimExpr>> preferred_extents) {
                          return ScopeIdDef(vars, extents, StringPairToScopeBinding(parent, cur),
                                            preferred_extents);
                        });
}

// Forward declarations for the file-static Compose/Compliment helpers used
// by ScopeIdDefVerifier::Verify below; defined further down in this file.
static ffi::Optional<ScopeIdDef> Compose(const ScopeIdDef& lhs, const ScopeIdDef& rhs);
static ffi::Optional<ScopeIdDef> Compliment(const ScopeIdDef& lhs, const ScopeIdDef& rhs);

bool ScopeIdDefVerifier::Verify(const ffi::Array<ScopeIdDef>& defs) {
  id_set.clear();
  arith::Analyzer ana;
  std::queue<ScopeIdDef> queue;

  auto insert_id = [&](const ScopeIdDef& id) {
    auto [it, inserted] = id_set.try_emplace(id->scope, id);
    if (!inserted) {
      TVM_FFI_ICHECK(ana.CanProveEqual(it->second.fused_extent(), id.fused_extent()))
          << "Inconsistent extents for scope binding " << static_cast<int>(id->scope);
    } else {
      queue.push(id);
    }
  };

  for (const auto& def : defs) {
    if (def->preferred_extents.defined()) {
      TVM_FFI_ICHECK(def->scope == ScopeBinding::kClusterCta)
          << "ValueError: preferred_extents is only valid for cluster→cta scope";
      TVM_FFI_ICHECK_EQ(def->preferred_extents.value().size(), def->extents.size())
          << "ValueError: preferred_extents must have the same size as extents, got "
          << def->preferred_extents.value().size() << " vs " << def->extents.size();
    }
    insert_id(def);
  }
  if (id_set.count(ScopeBinding::kClusterCtaPair)) {
    TVM_FFI_ICHECK(id_set.count(ScopeBinding::kClusterCta))
        << "ValueError: T.cta_id_in_pair() requires T.cta_id_in_cluster(...) in the same kernel";
  }

  while (!queue.empty()) {
    auto head = queue.front();
    queue.pop();

    // Snapshot to avoid iterator invalidation on insert
    std::vector<ScopeIdDef> snapshot;
    snapshot.reserve(id_set.size());
    for (const auto& [_, def] : id_set) snapshot.push_back(def);
    for (const auto& def : snapshot) {
      for (auto op : {Compose, Compliment}) {
        if (auto result = op(head, def)) insert_id(result.value());
        if (auto result = op(def, head)) insert_id(result.value());
      }
    }
  }
  return true;
}

namespace {
// The ScopeBinding enum is a closed set; these helpers project it back onto
// the (parent, cur) scope-kind pair so Compose/Compliment can operate on the
// hierarchy without reintroducing string plumbing.
std::pair<ffi::String, ffi::String> BindingParts(ScopeBinding b) {
  return ScopeBindingToStringPair(b);
}

ffi::Optional<ScopeBinding> TryStringPairToBinding(const ffi::String& parent,
                                                   const ffi::String& cur) {
  if (parent == "kernel" && cur == "cluster") return ScopeBinding::kKernelCluster;
  if (parent == "kernel" && cur == "cta") return ScopeBinding::kKernelCta;
  if (parent == "cluster" && cur == "cta") return ScopeBinding::kClusterCta;
  if (parent == "cta" && cur == "warpgroup") return ScopeBinding::kCtaWarpgroup;
  if (parent == "cta" && cur == "warp") return ScopeBinding::kCtaWarp;
  if (parent == "warpgroup" && cur == "warp") return ScopeBinding::kWarpgroupWarp;
  if (parent == "warp" && cur == "thread") return ScopeBinding::kWarpThread;
  if (parent == "cta" && cur == "thread") return ScopeBinding::kCtaThread;
  if (parent == "warpgroup" && cur == "thread") return ScopeBinding::kWarpgroupThread;
  if (parent == "cluster" && cur == "cta_pair") return ScopeBinding::kClusterCtaPair;
  return std::nullopt;
}
}  // namespace

static ffi::Optional<ScopeIdDef> Compose(const ScopeIdDef& lhs, const ScopeIdDef& rhs) {
  if (lhs->scope == ScopeBinding::kClusterCtaPair || rhs->scope == ScopeBinding::kClusterCtaPair) {
    return std::nullopt;
  }
  auto [l_parent, l_cur] = BindingParts(lhs->scope);
  auto [r_parent, r_cur] = BindingParts(rhs->scope);
  if (l_cur != r_parent) return std::nullopt;
  auto composed = TryStringPairToBinding(l_parent, r_cur);
  if (!composed.has_value()) return std::nullopt;
  return ScopeIdDef({Var("")}, {lhs.fused_extent() * rhs.fused_extent()}, composed.value());
}

static ffi::Optional<ScopeIdDef> Compliment(const ScopeIdDef& lhs, const ScopeIdDef& rhs) {
  if (lhs->scope == ScopeBinding::kClusterCtaPair || rhs->scope == ScopeBinding::kClusterCtaPair) {
    return std::nullopt;
  }
  if (is_zero(rhs.fused_extent())) return std::nullopt;
  arith::Analyzer ana;
  auto try_compliment = [&](PrimExpr lhs_ext, PrimExpr rhs_ext,
                            ScopeBinding scope) -> ffi::Optional<ScopeIdDef> {
    if (ana.CanProve(floormod(lhs_ext, rhs_ext) == 0)) {
      return ScopeIdDef({Var("")}, {floordiv(lhs_ext, rhs_ext)}, scope);
    }
    TVM_FFI_ICHECK(!ana.CanProve(floormod(lhs_ext, rhs_ext) != 0))
        << "ValueError: scope binding " << static_cast<int>(scope)
        << " has non-divisible extents: " << lhs_ext << " is not divisible by " << rhs_ext;
    return std::nullopt;
  };
  auto [l_parent, l_cur] = BindingParts(lhs->scope);
  auto [r_parent, r_cur] = BindingParts(rhs->scope);
  if (l_parent == r_parent && ScopeNameHigher(r_cur, l_cur)) {
    if (auto b = TryStringPairToBinding(r_cur, l_cur)) {
      return try_compliment(lhs.fused_extent(), rhs.fused_extent(), b.value());
    }
  }
  if (l_cur == r_cur && ScopeNameHigher(l_parent, r_parent)) {
    if (auto b = TryStringPairToBinding(l_parent, r_parent)) {
      return try_compliment(lhs.fused_extent(), rhs.fused_extent(), b.value());
    }
  }
  return std::nullopt;
}

/******** ScopeIdResolve: closed-enum static dispatch ********/
namespace {
using LaunchParams = ScopeIdResolve::LaunchParams;

std::pair<PrimExpr, PrimExpr> GetThread(const std::string& tag, const LaunchParams& params,
                                        bool allow_missing = false) {
  auto it = params.find(tag);
  if (it == params.end()) {
    TVM_FFI_ICHECK(allow_missing) << "Cannot find thread var: " << tag;
    return {0, 1};
  }
  return {(*it).second->var, (*it).second->dom->extent};
}

PrimExpr GetLinearThreadIndex(const LaunchParams& params) {
  PrimExpr tx, ty, tz, ex, ey, ez;
  std::tie(tx, ex) = GetThread("threadIdx.x", params, true);
  std::tie(ty, ey) = GetThread("threadIdx.y", params, true);
  std::tie(tz, ez) = GetThread("threadIdx.z", params, true);
  return tx + ty * ex + tz * ex * ey;
}

ffi::Array<PrimExpr> Trivial3DResolve(const LaunchParams& params, const char* prefix, int out_dim) {
  ffi::Array<PrimExpr> ret;
  for (int i = 0; i < out_dim; ++i) {
    ret.push_back(GetThread(std::string(prefix) + static_cast<char>('x' + i), params).first);
  }
  return ret;
}

ffi::Array<PrimExpr> ResolveCuda(ScopeBinding binding,
                                 const ffi::Optional<ffi::Array<PrimExpr>>& extents, int out_dim,
                                 const LaunchParams& params) {
  arith::Analyzer ana;
  switch (binding) {
    case ScopeBinding::kKernelCta:
      return Trivial3DResolve(params, "blockIdx.", out_dim);
    case ScopeBinding::kClusterCta:
      return Trivial3DResolve(params, "clusterCtaIdx.", out_dim);
    case ScopeBinding::kCtaThread:
      return Trivial3DResolve(params, "threadIdx.", out_dim);
    case ScopeBinding::kKernelCluster: {
      TVM_FFI_ICHECK_LE(out_dim, 3)
          << "ValueError: kernel->cluster can only have 3 dimensions for now";
      ffi::Array<PrimExpr> ret;
      for (int i = 0; i < out_dim; ++i) {
        ret.push_back(tirx::Call(
            DataType::Int(32), builtin::ptx_fetch_register(),
            {IntImm(DataType::Int(32), 32), StringImm("clusterid." + std::string(1, 'x' + i))}));
      }
      return ret;
    }
    case ScopeBinding::kCtaWarpgroup: {
      TVM_FFI_ICHECK_EQ(out_dim, 1) << "ValueError: cta->warpgroup must be 1D";
      return {ana.Simplify(FloorDiv(GetThread("warp_id_in_cta", params).first, 4))};
    }
    case ScopeBinding::kCtaWarp: {
      TVM_FFI_ICHECK_EQ(out_dim, 1) << "ValueError: cta->warp must be 1D";
      return {ana.Simplify(GetThread("warp_id_in_cta", params).first)};
    }
    case ScopeBinding::kWarpgroupWarp: {
      TVM_FFI_ICHECK_EQ(out_dim, 1) << "ValueError: warpgroup->warp must be 1D";
      return {ana.Simplify(FloorMod(GetThread("warp_id_in_cta", params).first, 4))};
    }
    case ScopeBinding::kWarpgroupThread: {
      TVM_FFI_ICHECK_EQ(out_dim, 1) << "ValueError: warpgroup->thread must be 1D";
      return {ana.Simplify(FloorMod(GetLinearThreadIndex(params), 128))};
    }
    case ScopeBinding::kWarpThread: {
      TVM_FFI_ICHECK_EQ(out_dim, 1) << "ValueError: warp->thread must be 1D";
      return {ana.Simplify(FloorMod(GetLinearThreadIndex(params), 32))};
    }
    case ScopeBinding::kClusterCtaPair: {
      TVM_FFI_ICHECK_EQ(out_dim, 1) << "ValueError: cluster->cta_pair must be 1D";
      PrimExpr cbx, cby, cbz, ex, ey, ez;
      std::tie(cbx, ex) = GetThread("clusterCtaIdx.x", params, true);
      std::tie(cby, ey) = GetThread("clusterCtaIdx.y", params, true);
      std::tie(cbz, ez) = GetThread("clusterCtaIdx.z", params, true);
      return {ana.Simplify(FloorMod(cbx + cby * ex + cbz * ex * ey, 2))};
    }
  }
  LOG(FATAL) << "Internal Error: unknown ScopeBinding " << static_cast<int>(binding);
}
}  // namespace

ffi::Array<PrimExpr> ScopeIdResolve::Resolve(ScopeBinding binding,
                                             const ffi::Optional<ffi::Array<PrimExpr>>& extents,
                                             int out_dim, const ffi::String& target_kind,
                                             const LaunchParams& params) {
  if (target_kind == "cuda") return ResolveCuda(binding, extents, out_dim, params);
  LOG(FATAL) << "Cannot resolve ScopeIdDef for target=" << target_kind
             << " binding=" << static_cast<int>(binding);
}

PrimExpr ScopeIdResolve::ComputeWarpIdInCta(const LaunchParams& params) {
  PrimExpr warp_id = FloorDiv(GetLinearThreadIndex(params), 32);
  PrimExpr mask = IntImm(DataType::UInt(32), 0xffffffff);
  return Call(warp_id.dtype(), builtin::tvm_warp_shuffle(),
              {mask, warp_id, IntImm(DataType::Int(32), 0), IntImm(DataType::Int(32), 32),
               IntImm(DataType::Int(32), 32)});
}

}  // namespace tirx
}  // namespace tvm
