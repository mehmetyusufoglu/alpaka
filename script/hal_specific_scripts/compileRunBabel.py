import os
import shutil
import subprocess
import multiprocessing
from datetime import datetime
import time


def run_command(command, capture_output=False, timeout=120):
    """ Helper function to run a system command with a timeout and optionally capture output """
    env_setup = "source /etc/profile.d/modules.sh && source /opt/spack/share/spack/setup-env.sh && "
    bash_command = f"bash -c '{env_setup} {command}'"
    try:
        print(f"Running command: {command}")
        result = subprocess.run(
            bash_command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout
        )
        output = result.stdout.decode().strip()
        error = result.stderr.decode().strip()
        if output:
            print("Output:", output)
        if error:
            print("Error:", error)
        return output
    except subprocess.TimeoutExpired:
        print(f"Command timed out: {command}")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}\nError: {e.stderr.decode()}")
        return None


def run_command_with_retries(command, retries=3, delay=10):
    """ Helper function to retry a command in case of failure """
    for attempt in range(retries):
        output = run_command(command)
        if output is not None:
            return output
        print(f"Retrying command ({attempt + 1}/{retries}) after {delay} seconds...")
        time.sleep(delay)
    print(f"Command failed after {retries} retries: {command}")
    return None


def setup_environment(preset):
    """ Load necessary modules and set environment variables with retries and timeout """
    
    if preset == "gpu-cuda-nvcc": 
        commands = [
            "spack load cmake@3.25",
            "spack load /u3oct6d",  # Specific hash for Boost
            "spack load cuda@12.2"
        ]
        # Verify if nvcc exists
        nvcc_path = run_command("which nvcc", capture_output=True)
        if not nvcc_path:
            print("Error: nvcc not found. Ensure CUDA is loaded properly.")
            return False

        # Set LD_LIBRARY_PATH
        cuda_lib_path = os.path.join(os.path.dirname(nvcc_path), "../lib64")
        os.environ["LD_LIBRARY_PATH"] = cuda_lib_path + os.pathsep + os.getenv("LD_LIBRARY_PATH", "")
        print(f"Updated LD_LIBRARY_PATH with CUDA libraries: {cuda_lib_path}")
    elif preset == "gpu-hip":
        commands = [
            "spack load cmake@3.25",
            "spack load /u3oct6d",  # Specific hash for Boost
            "module load rocm-5.7.2" ]
    elif preset == "gpu-sycl-intel":
        commands = [
            "spack load cmake@3.25",
            "spack load /u3oct6d",  # Specific hash for Boost
            "spack load intel-oneapi-compilers@2023.1.0",
            "spack load intel-oneapi-dpl@2022.2.0"]

    for cmd in commands:
        if run_command_with_retries(cmd, retries=3, delay=10) is None:
            print(f"Failed to execute: {cmd}")
            return False

    # Verify environment
    print("Environment setup completed successfully.")
    return True

def clone_or_update_alpaka():
    """ Clone the Alpaka repository or update it if it already exists """
    if os.path.exists("alpaka"):
        os.chdir("alpaka")
        run_command("git checkout develop")
        run_command("git pull origin develop")
        os.chdir("..")
    else:
        run_command("git clone https://github.com/alpaka-group/alpaka.git --branch develop")


def switch_to_alpaka_root():
    """ Ensure the current working directory is the root of the Alpaka repository """
    if not os.path.exists("alpaka/CMakePresets.json"):
        print("Error: Script must be executed from the alpaka root directory.")
        exit(1)
    os.chdir("alpaka")  # Switch to the alpaka directory


def build_and_run_preset(preset):
    """ Configure, build, and run the benchmark for a specific preset """
    num_cores = max(1, multiprocessing.cpu_count() - 2)  # Use all cores minus 2, but at least 1

    # Get Boost include directory dynamically
    boost_path = subprocess.run(
        "spack location -i /u3oct6d",
        shell=True,
        stdout=subprocess.PIPE,
        check=True
    ).stdout.decode().strip() + "/include"
    print(f"Using Boost include directory: {boost_path}")

    # Verify if nvcc exists
    nvcc_path = run_command("which nvcc", capture_output=True)
    if not nvcc_path:
        print("Error: nvcc not found. Ensure CUDA is loaded properly.")
        return False

    # Verify if hipcc exists
    hipcc_path = run_command("which clang++", capture_output=True)
    if not nvcc_path:
        print("Error: hipcc not found. Ensure HIP is loaded properly.")
        return False


    # Backend-specific flags
    extra_flags = ""
    if preset == "gpu-cuda-nvcc":
        extra_flags = f"-Dalpaka_ACC_GPU_CUDA_ENABLE=ON -Dalpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE=OFF -DCMAKE_CUDA_COMPILER={nvcc_path} -DCMAKE_CUDA_ARCHITECTURES=52"
    elif preset == "gpu-hip":
        extra_flags = f"-Dalpaka_ACC_GPU_HIP_ENABLE=ON -Dalpaka_ACC_GPU_HIP_ONLY_MODE=ON -Dalpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE=OFF -DCMAKE_HIP_COMPILER={hipcc_path}"
    elif preset == "gpu-sycl-intel":
        extra_flags = "-Dalpaka_ACC_SYCL_ENABLE=ON -Dalpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE=OFF"

    # Configure
    print(f"Configuring for preset: {preset}")
    run_command(f"cmake --preset {preset} -DBoost_INCLUDE_DIR={boost_path} -Dalpaka_BUILD_BENCHMARKS=ON {extra_flags}", capture_output=False)

    # Build
    print(f"Building for preset: {preset}")
    run_command(f"cmake --build --preset {preset} --target babelstream -j {num_cores}", capture_output=False)

    # Run
    print(f"Running benchmark for preset: {preset}")
    datetime_now = datetime.now()
    current_datetime = datetime_now.strftime("%Y-%m-%d_%H-%M-%S")
    os.chdir(f"build/{preset}/benchmarks/babelstream")

    # Generate result file with timestamp and preset name
    results_file = f"../../../babelstream-{preset}-{current_datetime}.txt"
    run_command(f"./babelstream --array-size=33554432 --number-runs=100 > {results_file}")
    print(f"Results saved to {results_file}")
    os.chdir("../../..")


if __name__ == "__main__":
    presets = ["gpu-hip"]  # Replace or add other presets as needed

    # Step 2: Clone or update Alpaka
    clone_or_update_alpaka()

    # Step 3: Ensure we are in the alpaka root directory
    switch_to_alpaka_root()

    # Step 4: Configure, build, and run for the selected presets
    for preset in presets:
        # Step 1: Setup environment
        if not setup_environment(preset):
            print("Failed to set up the environment. Exiting.")
            exit(1)
        print(f"Processing preset: {preset}")
        try:
            build_and_run_preset(preset)
        except Exception as e:
            print(f"Failed for preset {preset} with error: {e}")

