import pyshark
import numpy as np
import os
import re
from tqdm import tqdm

DATA_DIR = 'training'  # Change to 'testing' as needed

def process_pcap(file_path):
    try:
        print(f"Processing file: {file_path}")
        with pyshark.FileCapture(file_path, keep_packets=False) as cap:
            numPackets = 0
            total_size = 0
            max_size = 0
            large_packet_count = 0
            unique_ips = set()

            for packet in cap:
                # Each packet has a length field
                size = int(packet.length)
                numPackets += 1
                total_size += size
                if size > max_size:
                    max_size = size
                if size > 1000:
                    large_packet_count += 1

                # Extract IP addresses if available
                # Note: Not all packets may have an IP layer (e.g., ARP),
                # so we check first.
                if hasattr(packet, 'ip'):
                    src_ip = packet.ip.src
                    dst_ip = packet.ip.dst
                    unique_ips.add(src_ip)
                    unique_ips.add(dst_ip)

            if numPackets == 0:
                # No packets in this capture
                return None

            avg_packet_size = total_size / numPackets
            fraction_large_packets = large_packet_count / numPackets
            unique_ip_count = len(unique_ips)

            features = {
                'numPackets': numPackets,
                'avg_packet_size': avg_packet_size,
                'max_packet_size': max_size,
                'fraction_large_packets': fraction_large_packets,
                'unique_ip_count': unique_ip_count
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
                # Extract label from file name using regex
                label = re.split(r'[-.]', file_name)[0]

                feature_array = np.array([
                    features['numPackets'],
                    features['avg_packet_size'],
                    features['max_packet_size'],
                    features['fraction_large_packets'],
                    features['unique_ip_count']
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
