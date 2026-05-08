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
import os
import sys
from collections.abc import Callable

import numpy as np
import paramiko
import pytest
import torch

import tvm
from tvm.tirx import PrimFunc

from .claude_rewrite import rewrite_program

HOST_IP = ""
USERNAME = "ubuntu"


dtype_map = {"float32": "np.float32", "float16": "np.float16"}

np_dtype_map = {"float32": np.float32, "float16": np.float16}


def generate_std_output(func: PrimFunc, std_f_output: Callable):
    num_inputs = int(func.attrs["num_inputs"])
    np.random.seed(0)
    inputs = []
    for i in range(num_inputs):
        int_shape = [int(x) for x in func.buffer_map[func.params[i]].shape]
        inputs.append(
            torch.from_numpy(
                ((np.random.rand(*int_shape) - 0.5) * 2).astype(
                    np_dtype_map[func.buffer_map[func.params[i]].dtype]
                )
            )
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = [x.to(device) for x in inputs]
    outputs = [o.cpu().numpy() for o in std_f_output(*inputs)]
    return outputs


def generate_test_function(func: PrimFunc, target: tvm.target.Target):
    mod = tvm.IRModule({"main": func})
    mod = tvm.compile(mod, target=target, tir_pipeline="trn")
    src = mod.mod.imports[0].inspect_source()
    func_str = src
    func_name = func.attrs["global_symbol"]
    func_str += "def test_func():\n"
    cnt = 0

    for param in func.params:
        if param not in func.buffer_map:
            continue
        buffer = func.buffer_map[param]
        assert buffer.dtype in dtype_map, f"Unsupported dtype {buffer.dtype}"
        func_str += f"  param_{cnt} = ((np.random.rand(*{buffer.shape}) - 0.5) * 2).astype({dtype_map[buffer.dtype]})\n"  # noqa: E501
        cnt += 1
    func_str += f"  {func_name}_kernel({', '.join([f'param_{i}' for i in range(cnt)])})\n"
    num_inputs = int(func.attrs["num_inputs"])
    total_params = len(func.params)
    for i in range(num_inputs, total_params):
        func_str += f"  np.save('output_{i - num_inputs}.npy', param_{i})\n"
    func_str += "np.random.seed(0)\n"
    func_str += "test_func()\n"
    func_str = rewrite_program(func_str, func_name)
    return func_str


def create_remote_temp_dir(ssh_client):
    """
    Create a temporary directory on the remote machine.
    Returns the path to the created directory.
    """
    print("Creating a remote temporary directory...")
    stdin, stdout, stderr = ssh_client.exec_command("mktemp -d")
    tmp_dir = stdout.read().decode().strip()
    error = stderr.read().decode().strip()
    if error:
        print("Error creating temporary directory:", error)
        sys.exit(1)
    print("Remote temporary directory created:", tmp_dir)
    return tmp_dir


def send_script_string(ssh_client, script_str, remote_file):
    """
    Write the provided Python script (as a string) to a file on the remote machine.
    """
    sftp = ssh_client.open_sftp()
    print(f"Uploading Python script to {remote_file}...")
    with sftp.open(remote_file, "w") as remote_file_handle:
        remote_file_handle.write(script_str)
    sftp.close()


def run_remote_script(ssh_client, remote_script, remote_dir):
    """
    Execute the remote Python script.
    The command changes to the remote directory and runs the script.
    """
    command = (
        f"source ~/test-schedule-venv/bin/activate && cd {remote_dir} && python {remote_script}"
    )
    print(f"Executing remote command: {command}")
    stdin, stdout, stderr = ssh_client.exec_command(command)

    # Optionally, print the output and error streams.
    output = stdout.read().decode()
    error = stderr.read().decode()
    if output:
        print("Remote output:")
        print(output)
    if error:
        raise RuntimeError(f"Error running remote script: {error}")


def fetch_npy_files(ssh_client, remote_dir, local_dir):
    """
    Download all files ending with .npy from the remote directory to the local directory.
    """
    sftp = ssh_client.open_sftp()
    print(f"Listing files in remote directory: {remote_dir}")
    try:
        remote_files = sftp.listdir(remote_dir)
    except OSError as e:
        print(f"Error listing directory {remote_dir}: {e}")
        sftp.close()
        return

    npy_files = [file for file in remote_files if file.endswith(".npy")]
    if not npy_files:
        print("No .npy files found on the remote machine.")
    else:
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
        for file in npy_files:
            remote_path = os.path.join(remote_dir, file)
            local_path = os.path.join(local_dir, file)
            print(f"Downloading {remote_path} to {local_path}...")
            sftp.get(remote_path, local_path)
    sftp.close()


def cleanup_local_dir(local_dir):
    """Remove the local directory containing downloaded .npy files."""
    print(f"Cleaning up local directory: {local_dir}")
    os.system(f"rm -rf {local_dir}")


def cleanup_remote_dir(ssh_client, remote_dir):
    """Remove the temporary directory from the remote machine."""
    command = f"rm -rf {remote_dir}"
    print(f"Cleaning up remote directory: {remote_dir}")
    ssh_client.exec_command(command)


def run_on_remote_and_check_correct(func, std_f_output, target):
    # Remote server configuration
    port = 22
    # Python script as a string
    python_script_str = generate_test_function(func, target)
    print(python_script_str)
    # Name for the remote script file
    remote_script = "remote_script.py"
    # Local directory to store downloaded .npy files
    local_output_dir = "./downloaded_npy_files"

    # Create SSH client and set policy to auto-add remote host key
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    print("Connecting to the remote machine...")
    ssh_client.connect(HOST_IP, port=port, username=USERNAME)

    try:
        # Create a temporary directory on the remote machine
        remote_temp_dir = create_remote_temp_dir(ssh_client)

        # Define the remote path for the script within the temporary directory
        remote_script_path = os.path.join(remote_temp_dir, remote_script)

        # Upload the Python script (from the string) to the remote temporary directory
        send_script_string(ssh_client, python_script_str, remote_script_path)

        # Run the remote Python script
        run_remote_script(ssh_client, remote_script, remote_temp_dir)

        # Retrieve all .npy files from the remote temporary directory
        fetch_npy_files(ssh_client, remote_temp_dir, local_output_dir)

        std_outputs = generate_std_output(func, std_f_output)
        # Check the correctness of the output
        for i, std_output in enumerate(std_outputs):
            output = np.load(f"{local_output_dir}/output_{i}.npy")
            np.testing.assert_allclose(output, std_output, rtol=1e-2, atol=1e-2)

    finally:
        # Clean up the remote temporary directory
        cleanup_remote_dir(ssh_client, remote_temp_dir)
        cleanup_local_dir(local_output_dir)
        ssh_client.close()
        print("Operation completed and remote temporary directory cleaned up.")


def check_ssh_agent_connection(hostname, username, port=22):
    """
    Attempt to establish an SSH connection to the given hostname using the SSH agent's keys.

    Parameters:
        hostname (str): The remote host to connect to.
        username (str): The username for the remote connection.
        port (int): The SSH port (default: 22).

    Returns:
        paramiko.PKey: The SSH key that was used for a successful connection.

    Raises:
        RuntimeError: If no keys are loaded in the SSH agent or if all connection attempts fail.
    """
    agent = paramiko.Agent()
    keys = agent.get_keys()
    if not keys:
        raise RuntimeError("No keys loaded in the SSH agent. Please add keys using ssh-add.")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    last_exception = None
    for key in keys:
        try:
            client.connect(hostname, port=port, username=username, pkey=key, timeout=10)
            # If connection is successful, close and return the key used.
            client.close()
            return key
        except Exception as e:
            last_exception = e
            continue

    raise RuntimeError(f"SSH connection failed using agent keys. Last error: {last_exception}")


@pytest.fixture(scope="session")
def ssh_client():
    """
    Pytest fixture that verifies an SSH connection can be made using keys from the SSH agent.

    Update the 'hostname' and 'username' variables as needed.
    If no connection can be made, tests depending on this fixture will be skipped.
    """
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(hostname=HOST_IP, username=USERNAME, timeout=5)
        yield client
    except Exception as e:
        pytest.skip(f"SSH connection failed: {e}")
    finally:
        client.close()


@pytest.mark.dependency(name="ssh_success")
def test_ssh_connection(ssh_client):
    """Test SSH connection and mark it as a dependency."""
    assert ssh_client.get_transport().is_active(), "SSH transport is not active"
