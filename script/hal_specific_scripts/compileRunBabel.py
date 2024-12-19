import os
import shutil  # Added shutil for directory operations
import subprocess
import multiprocessing
from datetime import datetime

def run_command(command, capture_output=False):
    """ Helper function to run a system command and optionally capture output """
    env_setup = "source /etc/profile.d/modules.sh && source /opt/spack/share/spack/setup-env.sh && "
    bash_command = f"bash -c '{env_setup} {command}'"
    try:
        print(f"Running command: {command}")
        result = subprocess.run(bash_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = result.stdout.decode().strip()
        error = result.stderr.decode().strip()
        if output:
            print("Output:", output)
        if error:
            print("Error:", error)
        return output
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}\nError: {e.stderr.decode()}")
        return None

def setup_environment():
    """ Load necessary modules and set environment variables """
    commands = [
        "module load rocm-5.7.2",
        "spack load cmake@3.25",
        "spack load /u3oct6d",  # Specific hash for Boost
        "spack load intel-oneapi-compilers@2023.2.1",
        "spack load intel-oneapi-dpl@2022.2.0",
        "spack load cuda@12.2"
    ]
    for cmd in commands:
        output = run_command(cmd)
        if "Failed" in output or "Error" in output:
            print(f"Failed to execute: {cmd}")
            return False

    # Set LD_LIBRARY_PATH for CUDA
    cuda_lib_path = subprocess.run("spack location -i cuda@12.2", shell=True, stdout=subprocess.PIPE).stdout.decode().strip() + "/lib64"
    os.environ['LD_LIBRARY_PATH'] = os.getenv('LD_LIBRARY_PATH', '') + os.pathsep + cuda_lib_path
    print(f"Updated LD_LIBRARY_PATH with CUDA libraries: {cuda_lib_path}")
    
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

def build_babelstream():
    """ Build only the babelstream benchmark with benchmarks enabled and appropriate backend flags """
    num_cores = max(1, multiprocessing.cpu_count() - 2)
    os.chdir("alpaka")
    boost_path = subprocess.run("spack location -i /u3oct6d", shell=True, stdout=subprocess.PIPE).stdout.decode().strip() + "/include"

     # Clean the build directory to start fresh
    build_dir = os.path.join(os.getcwd(), "build")
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    os.makedirs(build_dir)

    run_command(f"cmake -S . -B build -Dalpaka_ACC_GPU_CUDA_ENABLE=ON -Dalpaka_ACC_GPU_CUDA_ONLY_MODE=ON -Dalpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE=OFF -Dalpaka_BUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release -DBoost_INCLUDE_DIR={boost_path}")
    run_command(f"cmake --build build --target babelstream -j {num_cores}")
    os.chdir("..")

def build_babelstream_rocm():
    """ Build the BabelStream benchmark for ROCm/HIP backend """
    num_cores = max(1, multiprocessing.cpu_count() - 2)
    os.chdir("alpaka")
    boost_path = subprocess.run("spack location -i /u3oct6d", shell=True, stdout=subprocess.PIPE).stdout.decode().strip() + "/include"
    
    # Clean the build directory to start fresh
    build_dir = os.path.join(os.getcwd(), "build")
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    os.makedirs(build_dir)

    run_command(f"cmake -S . -B build -Dalpaka_ACC_GPU_HIP_ENABLE=ON -Dalpaka_ACC_GPU_HIP_ONLY_MODE=ON -Dalpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE=OFF -Dalpaka_BUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release -DBoost_INCLUDE_DIR={boost_path}")
    run_command(f"cmake --build build --target babelstream -j {num_cores}")
    os.chdir("..")

#  extra_flags = "-DALPAKA_ACC_SYCL_ENABLED=ON -Dalpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE=OFF -Dalpaka_SYCL_ONEAPI_GPU=ON alpaka_SYCL_ONEAPI_GPU_DEVICES=\"spir64\""

#
#
def build_babelstream_sycl():
    """ Build the BabelStream benchmark for Intel SYCL backend """
    num_cores = max(1, multiprocessing.cpu_count() - 2)
    os.chdir("alpaka")
    boost_path = subprocess.run("spack location -i /u3oct6d", shell=True, stdout=subprocess.PIPE).stdout.decode().strip() + "/include"
    
    # Clean the build directory to start fresh
    build_dir = os.path.join(os.getcwd(), "build")
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    os.makedirs(build_dir)

    run_command(f"cmake -S . -B build -Dalpaka_ACC_SYCL_ENABLE=ON -Dalpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE=OFF -Dalpaka_ACC_CPU_B_SEQ_T_THREADS=OFF -Dalpaka_BUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release -Dalpaka_SYCL_ONEAPI_GPU=ON -Dalpaka_SYCL_ONEAPI_GPU_DEVICES=spir64 -DBoost_INCLUDE_DIR={boost_path}")
    run_command(f"cmake --build build --target babelstream -j {num_cores}")
    os.chdir("..")

def run_babelstream():
    """ Navigate to the build directory and run babelstream, saving the output to a file """
    datetime_now = datetime.now()
    current_datetime = datetime_now.strftime("%Y%m%d%H%M%S")
    os.chdir("alpaka/build/benchmarks/babelstream")
    filename = f"{current_datetime}-babelstream-output.txt"
    run_command(f"./babelstream --array-size=33554432 --number-runs=100 > ../../../../{filename}")
    os.chdir("../../../..")
    print(f"Results saved to {filename}")

if __name__ == "__main__":
    if setup_environment():
        clone_or_update_alpaka()
#        build_babelstream()
#        run_babelstream()
#        build_babelstream_rocm()
#        run_babelstream()
        build_babelstream_sycl()
        run_babelstream()
#        build_and_test_babelstream_sycl();
    else:
        print("Environment setup failed. Please check the errors and try again.")


