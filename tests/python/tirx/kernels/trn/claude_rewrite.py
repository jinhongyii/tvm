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


"""
This script is a workaround for a bug in the Neuron compiler.
"""

import hashlib
import os

import anthropic

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


def extract_output(text):
    start_tag = "<output_start>"
    end_tag = "<output_end>"

    start_index = text.find(start_tag)
    if start_index == -1:
        return None  # Start tag not found

    # Move past the start tag
    start_index += len(start_tag)

    end_index = text.find(end_tag, start_index)
    if end_index == -1:
        return None  # End tag not found

    # Extract the substring between tags
    return text[start_index:end_index]


def rewrite_program(program, func_name):
    if not os.path.exists("cache"):
        os.makedirs("cache")
    program_hash = hashlib.sha256(program.encode()).hexdigest()
    if os.path.exists(f"cache/output_{func_name}_{program_hash}.txt"):
        # Read content from the cache if it exists
        with open(f"cache/output_{func_name}_{program_hash}.txt") as file:
            content_str = file.read()
        print("Cache loaded successfully")
    else:
        print("Cache not found, rewriting program, this may take a while...")
        print(f"original program: {program}")
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        message = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=30000,
            messages=[
                {
                    "role": "user",
                    "content": f"""
            Below is a DSL program. For each psum tensor (variable allocated using nl.ndarray(..., buffer=ncc.psum.mod_alloc)), do the following things:
    1. Find out nisa.nc_matmul instructions that uses this tensor as output. For each appearance, calculate the tripcount by multipling the extent of all the spatial loops above it.
    We define spatial loop and reduction loop as follows (slightly different from normal understanding):
    Spatial loop: A loop that the matmul(producer) and consumer are both under it.
    Reduction loop: A loop that only the matmul(producer) is under it.
    Sum up the tripcounts of **spatial loops**, and this number represents the total number of instructions that writes value to this tensor.
    2. Give each matmul instruction that access this psum tensor an unique instruction id, and rewrite the psum tensor access (along with its consumer) to tensor[id, *original_indices]
    3.  rewrite the tensor allocation to nl.ndarray(shape=[total_num_write_instr, *original_shape], ..., buffer=ncc.psum.mod_alloc(..., num_bank_tiles=(1, 8))

    For example,

    ```
    a_psum = nl.ndarray(shape=(8, nl.par_dim(128), 512], buffer=ncc.psum.mod_alloc(base_bank=0, base_addr=0, num_bank_tiles=(8, ))

    for i in Tx.sequential_range(4, body_no_reorder=True):
      for j in Tx.sequential_range(32, body_no_reorder=True):
        for k in Tx.sequential_range(16, body_no_reorder=True):
          a_psum[j % 8, :, :] += nisa.nc_matmul(...)
        b[...] = nl.copy(a_psum[j%8, :, :])
    ```
    Here i and j are spatial loops, and k is reduction loop, because nisa.nc_matmul and its consumer(nl.copy) are both under the i loop and j loop, while the consumer is not under the k loop.
    The tripcount is calculated as 4*32=128.

    Say we have a more complex program

    ```
    a_psum = nl.ndarray(shape=(8, nl.par_dim(128), 512], buffer=ncc.psum.mod_alloc(base_bank=0, base_addr=0, num_bank_tiles=(8, ))

    for i in Tx.sequential_range(4, body_no_reorder=True):
      for j in Tx.sequential_range(32, body_no_reorder=True):
        for k in Tx.sequential_range(16, body_no_reorder=True):
          a_psum[j % 8, :, :] += nisa.nc_matmul(...)
        b[...] = nl.copy(a_psum[j%8, :, :])
      for k in Tx.sequential_range(16, body_no_reorder=True):
        a_psum[0, :, :] += nisa.nc_matmul(...)
      b[...] = nl.copy(a_psum[0, :, :])
    ```
    The tripcount of first matmul is 128, and the tripcount of second matmul is 4.
    So the unique id of the first matmul is 0-127, and the unique id of the second matmul is 128-131.
    We would rewrite the program to
    ```
    a_psum = nl.ndarray(shape=(132, 8, nl.par_dim(128), 512], buffer=ncc.psum.mod_alloc(base_bank=0, base_addr=0, num_bank_tiles=(1, 8, ))

    for i in Tx.sequential_range(4, body_no_reorder=True):
      for j in Tx.sequential_range(32, body_no_reorder=True):
        for k in Tx.sequential_range(16, body_no_reorder=True):
          a_psum[i*32 + j, j % 8, :, :] += nisa.nc_matmul(...)
        ... = nl.copy(a_psum[i*32 + j,  j%8, :, :])
      for k in Tx.sequential_range(16, body_no_reorder=True):
        a_psum[128+i, 0, :, :] += nisa.nc_matmul(...)
      ... = nl.copy(a_psum[128+i, 0, :, :])
    ```

    Do not rewrite other part of the program. Do not write a new script to do the rewrite. Here's the program, please print the rewritten program after "<output_start>", and print "<output_end>" after the rewritten program.
        ```
            {program}
        ```
            """,  # noqa: E501
                }
            ],
            thinking={"type": "enabled", "budget_tokens": 10000},
            timeout=900,
        )

        content_str = message.content[1].text
        with open(f"cache/output_{func_name}_{program_hash}.txt", "w") as f:
            f.write(content_str)

    return extract_output(content_str)
