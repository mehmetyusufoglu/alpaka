#!/bin/bash

# Function to print directory information
print_directory_info() {
    echo "Current Working Directory: $(pwd)"
    echo "Script Directory: $(dirname "$(realpath "$0")")"
}

# Function to clone or update the Alpaka repository
clone_or_update_alpaka() {
    if [ -d "alpaka" ]; then
        echo "Updating Alpaka repository..."
        cd alpaka || exit 1
        git checkout develop
        git pull origin develop
        cd ..
    else
        echo "Cloning Alpaka repository..."
        git clone https://github.com/alpaka-group/alpaka.git --branch develop
    fi
}

# Function to set up the environment for gpu-hip
setup_environment_gpu_hip() {
    echo "Setting up environment for gpu-hip..."

    # Load necessary modules and Spack packages
    source /etc/profile.d/modules.sh
    source /opt/spack/share/spack/setup-env.sh

    spack load cmake@3.25 || { echo "Failed to load cmake@3.25"; return 1; }
    spack load /u3oct6d || { echo "Failed to load Boost"; return 1; }  # Specific hash for Boost
    module load rocm-5.7.2 || { echo "Failed to load ROCm-5.7.2"; return 1; }

    # Verify if hipcc exists
    which hipcc > /dev/null || { echo "Error: hipcc not found. Ensure HIP is loaded properly."; return 1; }

    # Set LD_LIBRARY_PATH for ROCm
    # export LD_LIBRARY_PATH=/opt/rocm-5.7.2/lib:/opt/rocm-5.7.2/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    echo "Environment setup completed successfully."
}

# Function to configure, build, and run the benchmark for gpu-hip
build_and_run_gpu_hip() {
    local preset="gpu-hip"
    local num_cores=$(( $(nproc) - 2 ))
    num_cores=$(( num_cores < 1 ? 1 : num_cores ))  # Ensure at least 1 core is used

    # Get Boost include directory dynamically
    boost_path=$(spack location -i /u3oct6d)/include
    echo "Using Boost include directory: $boost_path"

    # Backend-specific flags
    extra_flags="-Dalpaka_ACC_GPU_HIP_ENABLE=ON \
                 -Dalpaka_ACC_GPU_HIP_ONLY_MODE=ON \
                 -Dalpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE=OFF"		   
    #            -DCMAKE_HIP_COMPILER=$(which hipcc)"
    # Switch to Alpaka root directory
    if [ "$(basename "$(pwd)")" == "build" ]; then
        cd .. || exit 1
    fi
    if [ ! -f "CMakePresets.json" ]; then
        echo "Error: Script must be executed from the alpaka root directory."
        exit 1
    fi

    echo "Current Working Directory: $(pwd)"
    
    # Create build directory for the preset
    build_dir="build/$preset"
    #echo "Make Directory: $build_dir"
    #mkdir -p "$build_dir"

    # Configure
    echo "Current Working Directory: $(pwd)"
    echo "Configuring for preset: $preset"
    cmake --preset "$preset" \
          -DBoost_INCLUDE_DIR="$boost_path" \
          -Dalpaka_BUILD_BENCHMARKS=ON \
          $extra_flags

    # Build
    echo "Current Working Directory: $(pwd)"
    cd "$build_dir" || exit 1
    echo "Current Working Directory: $(pwd)"
    echo "Building for preset: $preset"
    cmake --build . --target babelstream -j "$num_cores"

    echo "Current Working Directory: $(pwd)"
    # Run benchmark
    echo "Running benchmark for preset: $preset"
    datetime_now=$(date +"%Y-%m-%d_%H-%M-%S")
    echo "Results directory: $(pwd)/.."
    results_file="../babelstream-$preset-$datetime_now.txt"
    echo "Run $(pwd)/benchmarks/babelstream/babelstream --array-size=33554432 --number-runs=100 > $results_file"
    $(pwd)/benchmarks/babelstream/babelstream --array-size=33554432 --number-runs=100 > "$results_file"
    echo "Results saved to $results_file"
}

# Main script execution
print_directory_info

# Step 1: Clone or update Alpaka
clone_or_update_alpaka

echo "Current Working Directory: $(pwd)"
# Step 2: Ensure we are in the Alpaka root directory
if [ "$(basename "$(pwd)")" != "alpaka" ]; then
    echo "Error: Script must be executed from the alpaka root directory."
    exit 1
fi

# Step 3: Set up environment for gpu-hip
if ! setup_environment_gpu_hip; then
    echo "Failed to set up the environment. Exiting."
    exit 1
fi

# Step 4: Configure, build, and run for gpu-hip
build_and_run_gpu_hip

echo "Script execution completed."
