# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""copy_async dispatch variant: tmem<->local."""

from tvm.tirx import PrimFunc
from tvm.tirx.operator.tile_primitive_dispatch import DispatchContext, predicate, register_dispatch
from tvm.tirx.stmt import TilePrimitiveCall

from ..copy import _is_valid_copy, _scope_allowed, copy_tmem_local_impl
from ..exec_scope_utils import exec_scope_ok


# === Variant: copy_async/tmem<->local (priority=10) ===
#
# Same as copy/tmem<->local but async (async_op=True).
# After: Tx.ptx.tcgen05.ld/st with async completion signaling.
@register_dispatch(
    "copy_async",
    "cuda",
    variant="tmem<->local",
    priority=10,
    when=[
        predicate("validate_copy_op", _is_valid_copy),
        predicate("exec_scope", exec_scope_ok, expected_scopes=["warpgroup"]),
        predicate(
            "storage_scope", _scope_allowed, allowed_pairs=[("tmem", "local"), ("local", "tmem")]
        ),
    ],
)
def copy_async_schedule_tmem_local_async(
    op_call: TilePrimitiveCall, sctx: DispatchContext
) -> PrimFunc:
    return copy_tmem_local_impl(op_call, sctx, async_op=True)
