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

"""Configure pytest-xdist to distribute parametrized tests across workers.

The default 'loadfile' strategy sends all tests from the same file to one
worker.  Since test_kernels.py contains all 247 parametrized kernel tests,
this would serialize them.  Switching to 'load' distributes individual test
cases round-robin across workers, restoring parallelism.
"""


def pytest_configure(config):
    if config.pluginmanager.hasplugin("xdist"):
        dist = getattr(config.option, "dist", None)
        if dist in (None, "no", "loadfile"):
            config.option.dist = "load"
