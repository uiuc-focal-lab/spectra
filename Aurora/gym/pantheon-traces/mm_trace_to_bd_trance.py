import os


def calculate_bandwidth(trace_file, output_file, packet_size=1500, interval=0.5):
    bytes_per_megabit = 125000  # 1 megabit = 125,000 bytes

    with open(trace_file, 'r') as file:
        times = [float(line.strip()) / 1000 for line in file]  # Convert ms to seconds

    bandwidths = []

    if len(set(times)) == 1:  # Check if all timestamps are the same
        time_interval = times[0]
        packets_per_interval = len(times) * (interval / time_interval)
        bandwidth_mbps = (packets_per_interval * packet_size * 8) / (
                    interval * 1000000)  # Convert bytes to bits and seconds to Mbps
        bandwidths.append(bandwidth_mbps)
    else:
        # More than one timestamp, or varying timestamps, process every 0.5 seconds
        current_interval_start = 0
        interval_packet_count = 0
        last_time = 0

        for time in times:
            while time >= current_interval_start + interval:
                if interval_packet_count > 0:
                    bandwidth_mbps = (interval_packet_count * packet_size * 8) / (interval * 1000000)
                    bandwidths.append(bandwidth_mbps)
                current_interval_start += interval
                interval_packet_count = 0
            interval_packet_count += 1
            last_time = time

        # Handle the last calculated interval
        if interval_packet_count > 0:
            bandwidth_mbps = (interval_packet_count * packet_size * 8) / (interval * 1000000)
            bandwidths.append(bandwidth_mbps)

    # Write bandwidths to the output file
    with open(output_file, 'w') as file:
        for bandwidth in bandwidths:
            file.write(f'{bandwidth:.2f}\n')
# Example usage
def process_trace_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.trace') and not filename.startswith("bandwidth"):
            trace_file = os.path.join(directory, filename)
            output_file = os.path.join(directory, f'bandwidth_{filename}')
            calculate_bandwidth(trace_file, output_file)
            print(f'Processed {filename} -> {output_file}')

# Example usage
process_trace_files("./")
