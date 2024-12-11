import pyshark
import numpy as np
import os
from tqdm import tqdm
import asyncio

DATA_DIR = 'training'  # or 'testing', depending on which dataset you're processing

def process_pcap(file_path):
    try:
        print(f"Processing file: {file_path}")

        # Use 'with' statement to ensure the capture is properly closed
        with pyshark.FileCapture(file_path, keep_packets=False) as cap:
            packet_times = []
            packet_sizes = []
            packet_directions = []

            # Replace with your actual local IP address
            LOCAL_IP = '192.168.0.15'

            packet_count = 0

            for packet in cap:
                packet_count += 1
                try:
                    # Get packet timestamp
                    timestamp = float(packet.sniff_timestamp)
                    packet_times.append(timestamp)

                    # Get packet size
                    size = int(packet.length)
                    packet_sizes.append(size)

                    # Determine packet direction
                    if 'IP' in packet:
                        src = packet.ip.src
                        dst = packet.ip.dst
                    else:
                        continue  # Skip if no IP layer

                    if src == LOCAL_IP:
                        packet_directions.append(1)   # Outgoing
                    else:
                        packet_directions.append(-1)  # Incoming

                except Exception as e:
                    print(f"Error processing packet {packet_count} in {file_path}: {e}")
                    continue

            print(f"Processed {packet_count} packets in {file_path}")

            if not packet_times:
                print(f"No packets processed in {file_path}.")
                return None

            # Ensure packet_times has at least two timestamps for np.diff
            if len(packet_times) < 2:
                print(f"Not enough packet times for inter-arrival time calculation in {file_path}.")
                return None

            # Calculate inter-packet arrival times
            inter_arrival_times = np.diff(packet_times)

            # Define intervals
            TIME_INTERVAL = 0.5
            MAX_INTERVALS = 20
            start_time = packet_times[0]
            intervals = np.arange(start_time, start_time + TIME_INTERVAL * MAX_INTERVALS, TIME_INTERVAL)
            num_packets_per_interval = np.zeros(MAX_INTERVALS)
            data_sent_per_interval = np.zeros(MAX_INTERVALS)
            data_received_per_interval = np.zeros(MAX_INTERVALS)

            for i in range(len(packet_times)):
                interval_index = int((packet_times[i] - start_time) / TIME_INTERVAL)
                if 0 <= interval_index < MAX_INTERVALS:
                    num_packets_per_interval[interval_index] += 1
                    if packet_directions[i] == 1:
                        data_sent_per_interval[interval_index] += packet_sizes[i]
                    else:
                        data_received_per_interval[interval_index] += packet_sizes[i]
                else:
                    print(f"Interval index {interval_index} out of bounds for {file_path}")
                    continue

            # Aggregate statistics
            features = {
                'num_packets_per_interval': num_packets_per_interval,
                'data_sent_per_interval': data_sent_per_interval,
                'data_received_per_interval': data_received_per_interval,
                'mean_inter_arrival_time': np.mean(inter_arrival_times),
                'std_inter_arrival_time': np.std(inter_arrival_times),
            }

            return features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None



def extract_features_from_directory(data_dir):
    feature_list = []
    labels = []
    file_names = os.listdir(data_dir)
    for file_name in tqdm(file_names):
        if file_name.endswith('.pcap'):
            file_path = os.path.join(data_dir, file_name)
            features = process_pcap(file_path)
            if features is not None:
                # Extract website label from file name
                label = file_name.split('-')[0]
                # Combine all features into a single array
                feature_array = np.concatenate([
                    features['num_packets_per_interval'],
                    features['data_sent_per_interval'],
                    features['data_received_per_interval'],
                    [features['mean_inter_arrival_time']],
                    [features['std_inter_arrival_time']]
                ])
                print(f"Feature array shape: {feature_array.shape}")
                feature_list.append(feature_array)
                labels.append(label)
            else:
                print(f"No features extracted from {file_path}")
    return np.array(feature_list), np.array(labels)

if __name__ == "__main__":
    features, labels = extract_features_from_directory(DATA_DIR)
    print(f"Extracted features shape: {features.shape}")
    print(f"Extracted labels shape: {labels.shape}")
    np.save('features.npy', features)
    np.save('labels.npy', labels)
