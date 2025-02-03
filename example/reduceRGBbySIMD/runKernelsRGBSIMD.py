import subprocess
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Define the executable path
executable_path = "/home/yusufo81/projects/alpaka-dir/alpaka/build/example/reduceRGBbySIMD/reduceRGBbySIMD"

# Define the range of powers of 2
powers_of_2 = range(10, 29, 2)

# Lists to store the results
num_elements_list = []
simd_times = []
simd_times2 = []
non_simd_times = []

# Run the executable for each power of 2
for power in powers_of_2:
    num_elements = 2 ** power
    command = [executable_path, f"numElements={num_elements}"]

    # Run the command and capture the output
    result = subprocess.run(command, capture_output=True, text=True)
    output = result.stdout

    # Initialize variables to store times
    simd_time = None
    simd_time2 = None
    non_simd_time = None

    # Extract the execution times from the output
    for line in output.split('\n'):
        if "SIMD1to1:" in line:
            simd_time = float(line.split(':')[1].strip().split('s')[0])
        elif "NonSIMD:" in line:
            non_simd_time = float(line.split(':')[1].strip().split('s')[0])
        elif "SIMD1toN:" in line:
            simd_time2 = float(line.split(':')[1].strip().split('s')[0])

    # Check if both times were captured
    if simd_time is None:
        print(f"Error: Could not capture SIMD execution time for numElements: 2^{power} ({num_elements})")
        continue
    if non_simd_time is None:
        print(f"Error: Could not capture Non-SIMD execution time for numElements: 2^{power} ({num_elements})")
        continue
    if simd_time2 is None:
        print(f"Error: Could not capture SIMD execution time for numElements: 2^{power} ({num_elements})")
        continue

    # Append the results to the lists
    num_elements_list.append(num_elements)
    simd_times.append(simd_time)
    non_simd_times.append(non_simd_time)
    simd_times2.append(simd_time2)

    # Print the captured times
    print(f"numElements: 2^{power} ({num_elements})")
    print(f"  SIMD Execution Time: {simd_time:.4f}s")
    print(f"  SIMD 1toN Execution Time: {simd_time2:.4f}s")
    print(f"  Non-SIMD Execution Time: {non_simd_time:.4f}s")

# Print the results summary
print("\nSummary Results:")
for i, num_elements in enumerate(num_elements_list):
    print(f"numElements: 2^{powers_of_2[i]} ({num_elements})")
    print(f"  SIMD Execution Time: {simd_times[i]:.4f}s")
    print(f"  SIMD 1toN Execution Time: {simd_times2[i]:.4f}s")
    print(f"  Non-SIMD Execution Time: {non_simd_times[i]:.4f}s")

# Plot the results
plt.figure(figsize=(12, 8))
plt.plot(num_elements_list, simd_times2, marker='o', label='SIMD1toN')
plt.plot(num_elements_list, simd_times, marker='o', label='SIMD1to1')
plt.plot(num_elements_list, non_simd_times, marker='x', label='Non-SIMD')

plt.title('Execution Times of SIMD and Non-SIMD Kernels')
plt.xlabel('Number of Elements (numElements)')
plt.ylabel('Execution Time (s)')
plt.xscale('log', base=2)
plt.xticks(num_elements_list, [f'2^{p}' for p in powers_of_2], rotation=45)
plt.grid(True, which="both", ls="--")
plt.legend()
plt.tight_layout()

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
plot_file_name = f"{current_time}_Runtimes.png"
# Save the plot as an image
plt.savefig(plot_file_name)
print(f"Plot saved as: {plot_file_name}")
# Show the plot
# plt.show()
