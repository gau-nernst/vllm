# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import json
import multiprocessing
import os
import sys
from shutil import which

try:
    # Try to get CUDA_HOME/ROCM_HOME from the PyTorch installation, which is
    # the most reliable source of truth for vLLM's build (a ROCm torch build
    # sets ROCM_HOME and leaves CUDA_HOME unset, and vice versa).
    from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME
except ImportError:
    print("Warning: PyTorch not found. Falling back to CUDA_HOME/ROCM_HOME "
          "environment variables.")
    CUDA_HOME = os.environ.get("CUDA_HOME")
    ROCM_HOME = os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH")

# GPU compiler detection per device backend.
GPU_BACKENDS = {
    "cuda": {
        "cmake_lang": "CUDA",
        "compilers": ("nvcc",),
        "home": CUDA_HOME,
        "example": "/usr/local/cuda/bin/nvcc",
    },
    "rocm": {
        "cmake_lang": "HIP",
        "compilers": ("amdclang++", "hipcc"),
        "home": ROCM_HOME,
        "example": "/opt/rocm/bin/amdclang++",
    },
}


def detect_device():
    """The GPU backend to build for.

    Follows the installed torch build (a CUDA build sets CUDA_HOME, a ROCm
    build sets ROCM_HOME); otherwise probes the host for a CUDA or ROCm
    toolchain. Defaults to cuda (upstream behavior: the compiler prompt then
    asks for nvcc).
    """
    if CUDA_HOME is not None:
        return "cuda"
    if ROCM_HOME is not None:
        return "rocm"
    has_cuda = os.path.isdir("/usr/local/cuda") or which("nvcc") or which("nvidia-smi")
    has_rocm = os.path.isdir("/opt/rocm") or which("rocminfo")
    if has_rocm and not has_cuda:
        return "rocm"
    return "cuda"


def detect_gpu_compiler(device):
    """Find the device backend's GPU compiler (nvcc / amdclang++ / hipcc)."""
    backend = GPU_BACKENDS[device]
    names, home, example = backend["compilers"], backend["home"], backend["example"]
    if home:
        path = os.path.join(home, "bin", names[0])
        if os.path.exists(path):
            print(f"Found {names[0]} via torch.utils.cpp_extension: {path}")
            return path
    for name in names:
        path = which(name)
        if path:
            print(f"Found {name} in PATH: {path}")
            return path
    return input(
        f"Could not automatically find {'/'.join(names)}. Please provide the "
        f"full path to the GPU compiler (e.g., {example}): "
    ).strip()


def get_python_executable():
    """Get the current Python executable, which is used to run this script."""
    return sys.executable


def get_cpu_cores():
    """Get the number of CPU cores."""
    return multiprocessing.cpu_count()


def generate_presets(output_path="CMakeUserPresets.json", force_overwrite=False):
    """Generates the CMakeUserPresets.json file."""

    print("Attempting to detect your system configuration...")

    # Detect the GPU backend and its compiler
    device = detect_device()
    print(f"Target device: {device}")
    gpu_compiler = detect_gpu_compiler(device)
    print(f"Using GPU compiler: {gpu_compiler}")

    # Detect Python executable
    python_executable = get_python_executable()
    if python_executable:
        print(f"Found Python via sys.executable: {python_executable}")
    else:
        python_executable_prompt = (
            "Could not automatically find Python executable. Please provide "
            "the full path to your Python executable for vLLM development "
            "(typically from your virtual environment, e.g., "
            "/home/user/venvs/vllm/bin/python): "
        )
        python_executable = input(python_executable_prompt).strip()
        if not python_executable:
            raise ValueError(
                "Could not determine Python executable. Please provide it manually."
            )

    print(f"Using Python executable: {python_executable}")

    # Get CPU cores
    cpu_cores = get_cpu_cores()
    nvcc_threads = min(4, cpu_cores)
    cmake_jobs = max(1, cpu_cores // nvcc_threads)
    nvcc_note = f"NVCC_THREADS={nvcc_threads} and " if device == "cuda" else ""
    print(f"Detected {cpu_cores} CPU cores. "
          f"Setting {nvcc_note}CMake jobs={cmake_jobs}.")

    # Get vLLM project root (assuming this script is in vllm/tools/)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print(f"VLLM project root detected as: {project_root}")

    # Ensure python_executable path is absolute or resolvable
    if not os.path.isabs(python_executable) and which(python_executable):
        python_executable = os.path.abspath(which(python_executable))
    elif not os.path.isabs(python_executable):
        print(
            f"Warning: Python executable '{python_executable}' is not an "
            "absolute path and not found in PATH. CMake might not find it."
        )

    cmake_lang = GPU_BACKENDS[device]["cmake_lang"]
    cache_variables = {
        "CMAKE_BUILD_TYPE": "Release",
        "VLLM_PYTHON_EXECUTABLE": python_executable,
        "CMAKE_INSTALL_PREFIX": "${sourceDir}",
        f"CMAKE_{cmake_lang}_COMPILER": gpu_compiler,
    }
    if device == "rocm":
        cache_variables["VLLM_TARGET_DEVICE"] = "rocm"
        if ROCM_HOME:
            cache_variables["ROCM_PATH"] = ROCM_HOME
    else:
        cache_variables["CMAKE_CUDA_FLAGS"] = ""
        cache_variables["NVCC_THREADS"] = str(nvcc_threads)

    # Detect compiler cache
    if which("sccache"):
        print("Using sccache for compiler caching.")
        for launcher in ("C", "CXX", "CUDA", "HIP"):
            cache_variables[f"CMAKE_{launcher}_COMPILER_LAUNCHER"] = "sccache"
    elif which("ccache"):
        print("Using ccache for compiler caching.")
        for launcher in ("C", "CXX", "CUDA", "HIP"):
            cache_variables[f"CMAKE_{launcher}_COMPILER_LAUNCHER"] = "ccache"
    else:
        print("No compiler cache ('ccache' or 'sccache') found.")

    configure_preset = {
        "name": "release",
        "binaryDir": "${sourceDir}/cmake-build-release",
        "cacheVariables": cache_variables,
    }
    if which("ninja"):
        print("Using Ninja generator.")
        configure_preset["generator"] = "Ninja"
        cache_variables["CMAKE_JOB_POOLS"] = f"compile={cmake_jobs}"
    else:
        print("Ninja not found, using default generator. Build may be slower.")

    presets = {
        "version": 6,
        # Keep in sync with CMakeLists.txt and requirements/build/cuda.txt
        "cmakeMinimumRequired": {"major": 3, "minor": 26, "patch": 1},
        "configurePresets": [configure_preset],
        "buildPresets": [
            {
                "name": "release",
                "configurePreset": "release",
                "jobs": cmake_jobs,
            }
        ],
    }

    output_file_path = os.path.join(project_root, output_path)

    if os.path.exists(output_file_path):
        if force_overwrite:
            print(f"Overwriting existing file '{output_file_path}'")
        else:
            overwrite = (
                input(f"'{output_file_path}' already exists. Overwrite? (y/N): ")
                .strip()
                .lower()
            )
            if overwrite != "y":
                print("Generation cancelled.")
                return

    try:
        with open(output_file_path, "w") as f:
            json.dump(presets, f, indent=4)
        print(f"Successfully generated '{output_file_path}'")
        print("\nTo use this preset:")
        print(f"1. Ensure you are in the vLLM root directory: cd {project_root}")
        print("2. Initialize CMake: cmake --preset release")
        print("3. Build+install: cmake --build --preset release --target install")

    except OSError as e:
        print(f"Error writing file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force-overwrite",
        action="store_true",
        help="Force overwrite existing CMakeUserPresets.json without prompting",
    )

    args = parser.parse_args()
    generate_presets(force_overwrite=args.force_overwrite)
