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
# ruff: noqa: E501


import numpy as np
import pytest
import torch

import tvm
import tvm.testing

generator = None


def get_seed_and_offset(increment: int) -> tuple[int, int]:
    global generator
    if generator is None:
        generator = torch.Generator(device=torch.device("cuda"))
    state = generator.get_state()
    seed, offset = state.view(torch.int64)
    offset += (increment + 3) // 4 * 4
    generator.set_state(torch.tensor([seed, offset], dtype=torch.int64).view(torch.uint8))
    return int(seed), int(offset)


@pytest.mark.skip(reason="Requires FlashInfer enabled and proper setup")
def test_sampling():
    # Test configuration
    batch_size = 10
    vocab_size = 5
    num_iterations = 1000
    tol_atol = 0.02
    tol_rtol = 0.05  # relative tolerance

    # Probability tensor (each row sums to 1)
    probs_np = np.array([[0.1, 0.2, 0.3, 0.2, 0.2] for _ in range(batch_size)], dtype="float32")
    top_p_arr_np = np.array([0.8] * batch_size, dtype="float32")

    dev = tvm.cuda(0)
    prob_tvm = tvm.runtime.tensor(probs_np, device=dev)
    top_p_arr_tvm = tvm.runtime.tensor(top_p_arr_np, device=dev)
    output_tvm = tvm.runtime.empty((batch_size,), "int32", device=dev)

    device = tvm.cuda()
    tvm.target.Target.from_device(device)
    sampling_func = tvm.get_global_func("flashinfer.top_p_sampling_from_prob")

    counts = np.zeros((batch_size, vocab_size), dtype="int32")

    for _ in range(num_iterations):
        # Generate seed and a random offset.
        philox_seed, philox_offset = get_seed_and_offset(32 * batch_size)

        # the kernel expects (probs, output, maybe_indices, deterministic, philox_seed, philox_offset, cuda_stream)
        sampling_func(
            prob_tvm,
            output_tvm,
            top_p_arr_tvm,
            philox_seed,
            philox_offset,
            torch.cuda.current_stream().cuda_stream,
        )

        out = output_tvm.numpy()
        for i in range(batch_size):
            sampled_token = out[i]
            counts[i, sampled_token] += 1

    # Convert counts to frequencies.
    frequencies = counts / float(num_iterations)
    print(f"frequencies: {frequencies}")

    # For each row, check that the empirical frequency is close to the input probability.
    for row in range(batch_size):
        tvm.testing.assert_allclose(frequencies[row], probs_np[row], rtol=tol_rtol, atol=tol_atol)


if __name__ == "__main__":
    # Run the test standalone (if not using pytest)
    test_sampling()
