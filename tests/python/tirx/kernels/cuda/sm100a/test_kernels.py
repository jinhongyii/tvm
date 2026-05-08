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
"""Unified sm100a kernel tests — thin wrapper over tirx-kernels registry.

Discovers all kernels registered in the ``tirx_kernels`` package (installed
via ``pip install -e /path/to/tirx-kernels``) and generates pytest
parametrized test cases from their ``CONFIGS``.
"""

import pytest
from tirx_kernels.registry import discover_kernels
from tirx_kernels.runner import run_kernel_test

import tvm.testing

# Discover all sm100a kernels
_registry = discover_kernels(min_compute_capability=10)

# Build (kernel_name, config) pairs for parametrize
_test_cases = []
_test_ids = []
for name, mod in sorted(_registry.items()):
    for cfg in getattr(mod, "CONFIGS", []):
        _test_cases.append((name, cfg))
        _test_ids.append(f"{name}-{cfg.get('label', 'default')}")


@tvm.testing.requires_cuda_compute_version(10, exact=True)
@pytest.mark.parametrize("kernel_name,config", _test_cases, ids=_test_ids)
def test_kernel(kernel_name, config):
    run_kernel_test(kernel_name, config, registry=_registry)
