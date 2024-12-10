import pyshark
import os
import re
import statistics
from sklearn.preprocessing import MinMaxScaler


def extract_summary_stats():
    # Extracts and outputs summary stats for each packet capture into
    # output/summary_stats_training_raw, output/summary_stats_training_normalized, output/summary_stats_testing_raw, 
    # and output/summary_stats_testing_normalized in the format:
    # [label] [numPackets] [total_data_sent] [stdev_arrival_times] [avg_inter_arrival_time] [median_arrival_time]
    # All data is normalized with min/max scaling
    # The scaler is fit and applied on the training data only, and just applied on testing data

    print("Extracting Summary Stats:")

    print("\nTraining Data Processing:")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    training_dir = os.path.join(root_dir, "training")
    training_data = []
    training_labels = []
    i = 0
    numfiles = len(os.listdir(training_dir))
    for file_name in sorted(os.listdir(training_dir)):
        i += 1
        print(str(i) + "/" + str(numfiles))

        capture_path = os.path.join(training_dir, file_name)

        with pyshark.FileCapture(capture_path) as capture:
            total_data_sent = 0
            numPackets = 0
            arrival_times = []
            for packet in capture:
                arrival_times.append(packet.sniff_time.timestamp())
                total_data_sent += int(packet.length)
                numPackets += 1
        
        if len(arrival_times) > 1:
            inter_arrival_times = [arrival_times[i] - arrival_times[i - 1] for i in range(1, len(arrival_times))]
            avg_inter_arrival_time = sum(inter_arrival_times) / len(inter_arrival_times)
        else:
            avg_inter_arrival_time = 0.0

        median_arrival_time = statistics.median(inter_arrival_times)
        stdev_arrival_times = statistics.stdev(inter_arrival_times)

        # print(re.split(r'[-.]', file_name)[0] + "\nNum Packets: " +  str(numPackets) + "\nData Sent: " + str(total_data_sent) + "\nStd Deviation Inter Arrival Time: " + str(stdev_arrival_times) + "\nAverage Inter Arrival Time: " + str(avg_inter_arrival_time) + "\nMedian Inter Arrival Time: " + str(median_arrival_time) + "\n")
        
        training_data.append([numPackets, total_data_sent,stdev_arrival_times,avg_inter_arrival_time,median_arrival_time])
        training_labels.append(re.split(r'[-.]', file_name)[0])
    
    #ftesting = open("summary_testing", "w")

    scaler = MinMaxScaler()
    normalized_train = scaler.fit_transform(training_data)

    ftraining = open("output/summary_stats_training_normalized", "w")
    for i, arr in enumerate(normalized_train):
        line = f"{training_labels[i]} " + " ".join(map(str, arr)) + "\n"
        ftraining.write(line)
    ftraining.close()

    ftraining2 = open("output/summary_stats_training_raw", "w")
    for i, arr in enumerate(training_data):
        line = f"{training_labels[i]} " + " ".join(map(str, arr)) + "\n"
        ftraining2.write(line)
    ftraining2.close()

    print("\nTesting Data Processing:")
    testing_dir = os.path.join(root_dir, "testing")
    testing_data = []
    testing_labels = []
    i = 0
    numfiles = len(os.listdir(testing_dir))
    for file_name in sorted(os.listdir(testing_dir)):
        i += 1
        print(str(i) + "/" + str(numfiles))

        capture_path = os.path.join(testing_dir, file_name)

        with pyshark.FileCapture(capture_path) as capture:
            total_data_sent = 0
            numPackets = 0
            arrival_times = []
            for packet in capture:
                arrival_times.append(packet.sniff_time.timestamp())
                total_data_sent += int(packet.length)
                numPackets += 1
        
        if len(arrival_times) > 1:
            inter_arrival_times = [arrival_times[i] - arrival_times[i - 1] for i in range(1, len(arrival_times))]
            avg_inter_arrival_time = sum(inter_arrival_times) / len(inter_arrival_times)
        else:
            avg_inter_arrival_time = 0.0

        median_arrival_time = statistics.median(inter_arrival_times)
        stdev_arrival_times = statistics.stdev(inter_arrival_times)

        # print(re.split(r'[-.]', file_name)[0] + "\nNum Packets: " +  str(numPackets) + "\nData Sent: " + str(total_data_sent) + "\nStd Deviation Inter Arrival Time: " + str(stdev_arrival_times) + "\nAverage Inter Arrival Time: " + str(avg_inter_arrival_time) + "\nMedian Inter Arrival Time: " + str(median_arrival_time) + "\n")
        
        testing_data.append([numPackets, total_data_sent,stdev_arrival_times,avg_inter_arrival_time,median_arrival_time])
        testing_labels.append(re.split(r'[-.]', file_name)[0])

    normalized_test = scaler.transform(testing_data)

    ftesting = open("output/summary_stats_testing_normalized", "w")
    for i, arr in enumerate(normalized_test):
        line = f"{testing_labels[i]} " + " ".join(map(str, arr)) + "\n"
        ftesting.write(line)
    ftesting.close()

    ftesting2 = open("output/summary_stats_testing_raw", "w")
    for i, arr in enumerate(testing_data):
        line = f"{testing_labels[i]} " + " ".join(map(str, arr)) + "\n"
        ftesting2.write(line)
    ftesting2.close()

# Already ran - no need to rerun
# extract_summary_stats()

def extract_aggregated_data():
    # Extracts and outputs aggregated data (data per interval) for each packet capture into
    # output/aggregated_data_training_raw, output/aggregated_data_training_normalized, output/aggregated_data_testing_raw, 
    # and output/aggregated_data_testing_normalized in the format:
    # [label] [data_sent] [data_sent] ... 
    # All data is normalized with min/max scaling
    # The scaler is fit and applied on the training data only, and just applied on testing data

    print("Extracting Aggreggated Data:")

    print("\nTraining Data Processing:")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    training_dir = os.path.join(root_dir, "training")
    training_data = []
    training_labels = []
    i = 0
    numfiles = len(os.listdir(training_dir))
    
    bucket_interval = 0.1
    total_window = 3.0

    for file_name in sorted(os.listdir(training_dir)):
        i += 1
        print(f"{i}/{numfiles}")

        capture_path = os.path.join(training_dir, file_name)

        with pyshark.FileCapture(capture_path) as capture:
            data_buckets = []
            starting_time = None

            for packet in capture:
                packet_time = packet.sniff_time.timestamp()

                if starting_time is None:
                    starting_time = packet_time

                # find the current bucket index
                bucket_index = round((packet_time - starting_time) / bucket_interval)

                # Cap bucket index to the total number of buckets
                total_buckets = int(total_window / bucket_interval)
                bucket_index = min(bucket_index, total_buckets - 1)

                while len(data_buckets) <= bucket_index:
                    data_buckets.append(0)

                # add to appropriate bucket
                data_buckets[bucket_index] += int(packet.length)

                # stop if we reach the total window
                if (packet_time - starting_time) > total_window:
                    break

            # pad with 0's if the packet capture ended early
            while len(data_buckets) < total_buckets:
                data_buckets.append(0)

        training_data.append(data_buckets)
        training_labels.append(re.split(r'[-.]', file_name)[0])

    # Normalize and save training data
    scaler = MinMaxScaler()
    normalized_train = scaler.fit_transform(training_data)

    with open("output/aggregated_data_training_normalized", "w") as ftraining:
        for i, arr in enumerate(normalized_train):
            line = f"{training_labels[i]} " + " ".join(map(str, arr)) + "\n"
            ftraining.write(line)

    with open("output/aggregated_data_training_raw", "w") as ftraining2:
        for i, arr in enumerate(training_data):
            line = f"{training_labels[i]} " + " ".join(map(str, arr)) + "\n"
            ftraining2.write(line)

    print("\nTesting Data Processing:")
    testing_dir = os.path.join(root_dir, "testing")
    testing_data = []
    testing_labels = []
    i = 0
    numfiles = len(os.listdir(testing_dir))

    for file_name in sorted(os.listdir(testing_dir)):
        i += 1
        print(f"{i}/{numfiles}")

        capture_path = os.path.join(testing_dir, file_name)

        with pyshark.FileCapture(capture_path) as capture:
            data_buckets = []
            starting_time = None

            for packet in capture:
                packet_time = packet.sniff_time.timestamp()

                if starting_time is None:
                    starting_time = packet_time

                # find the current bucket index
                bucket_index = round((packet_time - starting_time) / bucket_interval)

                # Cap bucket index to the total number of buckets
                total_buckets = int(total_window / bucket_interval)
                bucket_index = min(bucket_index, total_buckets - 1)

                while len(data_buckets) <= bucket_index:
                    data_buckets.append(0)

                # add to appropriate bucket
                data_buckets[bucket_index] += int(packet.length)

                # stop if we reach the total window
                if (packet_time - starting_time) > total_window:
                    break

            # pad with 0's if the packet capture ended early
            while len(data_buckets) < total_buckets:
                data_buckets.append(0)

        testing_data.append(data_buckets)
        testing_labels.append(re.split(r'[-.]', file_name)[0])

    normalized_test = scaler.transform(testing_data)

    with open("output/aggregated_data_testing_normalized", "w") as ftesting:
        for i, arr in enumerate(normalized_test):
            line = f"{testing_labels[i]} " + " ".join(map(str, arr)) + "\n"
            ftesting.write(line)

    with open("output/aggregated_data_testing_raw", "w") as ftesting2:
        for i, arr in enumerate(testing_data):
            line = f"{testing_labels[i]} " + " ".join(map(str, arr)) + "\n"
            ftesting2.write(line)

# Already ran - no need to rerun
# extract_aggregated_data()


def extract_summary_stats_v2():
    # Extracts and outputs summary stats for each packet capture into
    # output/summary_stats_v2_training_raw, output/summary_stats_v2_training_normalized, output/summary_stats_v2_testing_raw, 
    # and output/summary_stats_v2_testing_normalized in the format:
    # [label] [numPackets] [total_data_sent] [TLS handshake size] [num large packets (> 1000 bytes)] [num small packets (< 100 bytes)]
    # All data is normalized with min/max scaling
    # The scaler is fit and applied on the training data only, and just applied on testing data

    print("Extracting Summary Stats V2:")

    print("\nTraining Data Processing:")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    training_dir = os.path.join(root_dir, "training")
    training_data = []
    training_labels = []
    i = 0
    numfiles = len(os.listdir(training_dir))
    for file_name in sorted(os.listdir(training_dir)):
        i += 1
        print(str(i) + "/" + str(numfiles))

        capture_path = os.path.join(training_dir, file_name)

        with pyshark.FileCapture(capture_path) as capture:
            total_data_sent = 0
            numPackets = 0
            handshake_size = 0
            numlargepackets = 0
            numsmallpackets = 0
            for packet in capture:
                total_data_sent += int(packet.length)
                numPackets += 1

                if 'TLS' in packet:
                    tls_layer = packet.tls
                    if hasattr(tls_layer, 'handshake_type'):
                        handshake_size += int(packet.length)

                if int(packet.length) > 1000:
                    numlargepackets += 1
                elif int(packet.length) < 100:
                    numsmallpackets += 1
        
        
        training_data.append([numPackets, total_data_sent, handshake_size, numlargepackets, numsmallpackets])
        training_labels.append(re.split(r'[-.]', file_name)[0])

    scaler = MinMaxScaler()
    normalized_train = scaler.fit_transform(training_data)

    ftraining = open("output/summary_stats_training_v2_normalized", "w")
    for i, arr in enumerate(normalized_train):
        line = f"{training_labels[i]} " + " ".join(map(str, arr)) + "\n"
        ftraining.write(line)
    ftraining.close()

    ftraining2 = open("output/summary_stats_training_v2_raw", "w")
    for i, arr in enumerate(training_data):
        line = f"{training_labels[i]} " + " ".join(map(str, arr)) + "\n"
        ftraining2.write(line)
    ftraining2.close()

    print("\nTesting Data Processing:")
    testing_dir = os.path.join(root_dir, "testing")
    testing_data = []
    testing_labels = []
    i = 0
    numfiles = len(os.listdir(testing_dir))
    for file_name in sorted(os.listdir(testing_dir)):
        i += 1
        print(str(i) + "/" + str(numfiles))

        capture_path = os.path.join(testing_dir, file_name)

        with pyshark.FileCapture(capture_path) as capture:
            total_data_sent = 0
            numPackets = 0
            handshake_size = 0
            numlargepackets = 0
            numsmallpackets = 0
            for packet in capture:
                total_data_sent += int(packet.length)
                numPackets += 1

                if 'TLS' in packet:
                    tls_layer = packet.tls
                    if hasattr(tls_layer, 'handshake_type'):
                        handshake_size += int(packet.length)

                if int(packet.length) > 1000:
                    numlargepackets += 1
                elif int(packet.length) < 100:
                    numsmallpackets += 1
        
        
        testing_data.append([numPackets, total_data_sent, handshake_size, numlargepackets, numsmallpackets])
        testing_labels.append(re.split(r'[-.]', file_name)[0])

    normalized_test = scaler.transform(testing_data)

    ftesting = open("output/summary_stats_testing_v2_normalized", "w")
    for i, arr in enumerate(normalized_test):
        line = f"{testing_labels[i]} " + " ".join(map(str, arr)) + "\n"
        ftesting.write(line)
    ftesting.close()

    ftesting2 = open("output/summary_stats_testing_v2_raw", "w")
    for i, arr in enumerate(testing_data):
        line = f"{testing_labels[i]} " + " ".join(map(str, arr)) + "\n"
        ftesting2.write(line)
    ftesting2.close()

extract_summary_stats_v2()