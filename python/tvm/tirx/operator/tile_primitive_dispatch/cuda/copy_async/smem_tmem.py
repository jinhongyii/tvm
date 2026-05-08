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

"""copy_async dispatch variant: smem->tmem.

Same algorithm as copy/smem_tmem.py but does NOT emit ``tcgen05.commit``
after the cp loop — caller is responsible for managing completion.
"""

from tvm.tirx import PrimFunc
from tvm.tirx.operator.tile_primitive_dispatch import DispatchContext, predicate, register_dispatch
from tvm.tirx.stmt import TilePrimitiveCall

from ..copy import _is_valid_smem_tmem_copy, _single_thread_exec
from ..copy.smem_tmem import copy_smem_tmem_impl


@register_dispatch(
    "copy_async",
    "cuda",
    variant="smem->tmem",
    priority=10,
    when=[
        predicate("validate_smem_tmem_copy", _is_valid_smem_tmem_copy),
        predicate("exec_scope", _single_thread_exec),
    ],
)
def copy_async_schedule_smem_tmem(op_call: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    return copy_smem_tmem_impl(op_call, sctx, async_op=True)
