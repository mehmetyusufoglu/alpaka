import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import re
from datetime import datetime
import matplotlib.dates as mdates

def parse_filename(filename):
    """ Extract the timestamp from the filename based on its pattern """
    pattern = r'(\d{6})-(\d{2})-(\w+)-(\d{4})-babelstream-(\w+).txt'
    match = re.match(pattern, filename)
    if match:
        time_str = f"{match.group(1)}-{match.group(2)}-{match.group(3)}-{match.group(4)}"
        return datetime.strptime(time_str, '%H%M%S-%d-%B-%Y')
    return None

def parse_file_content(filepath):
    """ Parse file to extract bandwidth for each kernel under 'Precision:single' """
    data = {}
    parsing_data = False
    single_precision = False
    with open(filepath, 'r') as file:
        for line in file:
            if 'Precision:single' in line:
                single_precision = True  # Start parsing data after this marker
            elif 'Precision:double' in line:
                single_precision = False  # Stop parsing if double precision starts
            elif single_precision and 'Bandwidths(GB/s)' in line:
                parsing_data = True  # Enable data parsing on the next line
                continue
            elif parsing_data and single_precision:
                # Check for kernel data lines only, avoiding headers and summary lines
                if re.match(r"\s*\w+Kernel\s+\d+\.\d+", line.strip()):
                    parts = line.strip().split()
                    kernel_name = parts[0]
                    bandwidth = float(parts[1])
                    if kernel_name not in data:
                        data[kernel_name] = []
                    data[kernel_name].append(bandwidth)
                elif 'Kernels:' in line or '====' in line or 'All tests passed' in line:
                    parsing_data = False  # Stop parsing at end of section or irrelevant lines
    return data

def plot_data(agg_data, directory):
    """ Plot bandwidth over time for each kernel """
    for kernel, bandwidths in agg_data.items():
        timestamps = [mdates.date2num(date) for date in sorted(bandwidths.keys())]
        bw_values = [np.mean(bandwidths[date]) for date in sorted(bandwidths.keys())]  # Average if multiple entries per date

        plt.figure(figsize=(10, 5))
        plt.plot(mdates.num2date(timestamps), bw_values, marker='o', linestyle='-')
        plt.title(f'Bandwidth Over Time for {kernel}')
        plt.xlabel('Time')
        plt.ylabel('Bandwidth (GB/s)')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.grid(True)
        plt.tight_layout()
        plot_filename = os.path.join(directory, f"{kernel}_bandwidth.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"Saved plot to {plot_filename}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py /path/to/your/files")
        sys.exit(1)

    directory = sys.argv[1]
    agg_data = {}

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        timestamp = parse_filename(filename)
        if timestamp:
            kernel_data = parse_file_content(filepath)
            for kernel, bandwidths in kernel_data.items():
                if kernel not in agg_data:
                    agg_data[kernel] = {}
                if timestamp not in agg_data[kernel]:
                    agg_data[kernel][timestamp] = []
                agg_data[kernel][timestamp].extend(bandwidths)

    plot_data(agg_data, directory)
