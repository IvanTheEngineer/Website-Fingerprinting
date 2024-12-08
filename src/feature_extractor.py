import pyshark
import os
import re
import statistics
from sklearn.preprocessing import MinMaxScaler


def extract_summary_stats():
    # Extracts and outputs summary stats for each packet capture into
    # output/summary_stats_training and output/summary_stats_testing in the format 
    # [label] [numPackets] [total_data_sent] [stdev_arrival_times] [avg_inter_arrival_time] [median_arrival_time]
    # All data is normalized with min/max scaling
    # The scaler is fit and applied on the training data only, and just applied on testing data

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

    ftraining = open("output/summary_stats_training", "w")
    for i, arr in enumerate(normalized_train):
        line = f"{training_labels[i]} " + " ".join(map(str, arr)) + "\n"
        ftraining.write(line)
    ftraining.close()

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
    ftesting = open("output/summary_stats_testing", "w")
    for i, arr in enumerate(normalized_test):
        line = f"{testing_labels[i]} " + " ".join(map(str, arr)) + "\n"
        ftesting.write(line)
    ftesting.close()

extract_summary_stats()