import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

print("\nEach model is a keras sequential model which was ran with 150 epochs each")

# Aggregated Data Feature Model
testing_file = "output/aggregated_data_testing_normalized"
model_file = "trained_model_aggregated_data.keras"
columns = ["label"] + [f"data{i+1}" for i in range(30)]
data_testing = pd.read_csv(testing_file, sep=" ", header=None, names=columns)

values_testing = data_testing.iloc[:, 1:].values
labels_testing = data_testing.iloc[:, 0].values

label_encoder = LabelEncoder()
labels_testing = label_encoder.fit_transform(labels_testing)
labels_testing = to_categorical(labels_testing)

values_testing = np.array(values_testing)
labels_testing = np.array(labels_testing)

if os.path.exists(model_file):
    model = load_model(model_file)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
else:
    print(f"model {model_file} not found")
    exit()

loss, accuracy = model.evaluate(values_testing, labels_testing, verbose=0)
print("\nAggregated Data Extraction Model Results:")
print("(Features are data sent per 0.1 second interval within the first 3 seconds)")
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

# Summary Stats Feature Model
testing_file = "output/summary_stats_testing_normalized"
model_file = "trained_model_summary_stats.keras"
columns = ["label", "numPackets", "total_data_sent", "stdev_arrival_times", "avg_inter_arrival_time", "median_arrival_time"]
data_testing = pd.read_csv(testing_file, sep=" ", header=None, names=columns)

values_testing = data_testing.iloc[:, 1:].values
labels_testing = data_testing.iloc[:, 0].values

label_encoder = LabelEncoder()
labels_testing = label_encoder.fit_transform(labels_testing)
labels_testing = to_categorical(labels_testing)

values_testing = np.array(values_testing)
labels_testing = np.array(labels_testing)

if os.path.exists(model_file):
    model = load_model(model_file)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
else:
    print(f"model {model_file} not found")
    exit()

loss, accuracy = model.evaluate(values_testing, labels_testing, verbose=0)
print("\nSummary Stats Extraction Model Results:")
print("(Features are numPackets, total_data_sent, stdev_arrival_times, avg_inter_arrival_time, and median_arrival_time)")
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

# Summary Stats v2 Feature Model
testing_file = "output/summary_stats_testing_v2_normalized"
model_file = "trained_model_summary_stats_v2.keras"
columns = ["label", "numPackets", "total_data_sent", "TLS_handshake_size", "num_large_packets", "num_small_packets"]
data_testing = pd.read_csv(testing_file, sep=" ", header=None, names=columns)

values_testing = data_testing.iloc[:, 1:].values
labels_testing = data_testing.iloc[:, 0].values

label_encoder = LabelEncoder()
labels_testing = label_encoder.fit_transform(labels_testing)
labels_testing = to_categorical(labels_testing)

values_testing = np.array(values_testing)
labels_testing = np.array(labels_testing)

if os.path.exists(model_file):
    model = load_model(model_file)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
else:
    print(f"model {model_file} not found")
    exit()

loss, accuracy = model.evaluate(values_testing, labels_testing, verbose=0)
print("\nSummary Stats V2 Extraction Model Results:")
print("(Features are numPackets, total_data_sent, TLS_handshake_size, num_large_packets (> 1000 bytes), and num_small_packets (< 100 bytes))")
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

# Summary Stats v3 Feature Model
testing_file = "output/summary_stats_testing_v2_normalized"
model_file = "trained_model_summary_stats_v3.keras"
columns = ["label", "numPackets", "total_data_sent", "TLS_handshake_size", "num_large_packets", "num_small_packets"]
data_testing = pd.read_csv(testing_file, sep=" ", header=None, names=columns)

# trying with just the numPackets and total_data_sent
data_testing = data_testing.drop(columns=["TLS_handshake_size", "num_large_packets", "num_small_packets"])

values_testing = data_testing.iloc[:, 1:].values
labels_testing = data_testing.iloc[:, 0].values

label_encoder = LabelEncoder()
labels_testing = label_encoder.fit_transform(labels_testing)
labels_testing = to_categorical(labels_testing)

values_testing = np.array(values_testing)
labels_testing = np.array(labels_testing)

if os.path.exists(model_file):
    model = load_model(model_file)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
else:
    print(f"model {model_file} not found")
    exit()

loss, accuracy = model.evaluate(values_testing, labels_testing, verbose=0)
print("\nSummary Stats V3 Extraction Model Results:")
print("(Features are just numPackets and total_data_sent)")
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

# Summary Stats v4 Feature Model
model_file = "trained_model_summary_stats_v4.keras"

# File paths and column definitions for the two datasets
file_1_testing = "output/summary_stats_testing_v2_normalized"
columns_1 = ["label", "numPackets", "total_data_sent", "TLS_handshake_size", "num_large_packets", "num_small_packets"]

file_2_testing = "output/summary_stats_testing_normalized"
columns_2 = ["label", "numPackets", "total_data_sent", "stdev_arrival_times", "avg_inter_arrival_time", "median_arrival_time"]

# Load both datasets
data_1_testing = pd.read_csv(file_1_testing, sep=" ", header=None, names=columns_1)
data_2_testing = pd.read_csv(file_2_testing, sep=" ", header=None, names=columns_2)

# Merge the datasets
final_testing = pd.concat([data_1_testing, data_2_testing.iloc[:, 3:]], axis=1)

# Separate features and labels for training and testing
values_testing = final_testing.iloc[:, 1:].values
labels_testing = final_testing.iloc[:, 0].values

label_encoder = LabelEncoder()
labels_testing = label_encoder.fit_transform(labels_testing)
labels_testing = to_categorical(labels_testing)

values_testing = np.array(values_testing)
labels_testing = np.array(labels_testing)

if os.path.exists(model_file):
    model = load_model(model_file)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
else:
    print(f"model {model_file} not found")
    exit()

loss, accuracy = model.evaluate(values_testing, labels_testing, verbose=0)
print("\nSummary Stats V4 Extraction Model Results:")
print("(Features are numPackets, total_data_sent, stdev_arrival_times, avg_inter_arrival_time, median_arrival_time, TLS_handshake_size, num_large_packets (> 1000 bytes), and num_small_packets (< 100 bytes))")
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

# All Feature Extractions Combined Model
model_file = "trained_model_all_extractions_combined.keras"
# load testing datasets
# Datasets were already normalized using min-max so it doesn't need to be done again
files_and_columns = [
    (
        "output/summary_stats_testing_v2_normalized",
        ["label", "numPackets", "total_data_sent", "TLS_handshake_size", "num_large_packets", "num_small_packets"]
    ),
    (
        "output/summary_stats_testing_normalized",
        ["label", "numPackets", "total_data_sent", "stdev_arrival_times", "avg_inter_arrival_time", "median_arrival_time"]
    ),
    (
        "output/aggregated_data_testing_normalized",
        ["label"] + [f"data{i+1}" for i in range(30)]
    )
]

testing_dfs = []

# combining all 3 feature extractions
for testing_file, columns in files_and_columns:
    testing_df = pd.read_csv(testing_file, sep=" ", header=None, names=columns)
    testing_dfs.append(testing_df)

combined_testing = pd.concat([df.iloc[:, 1:] for df in testing_dfs], axis=1)
combined_testing.insert(0, "label", testing_dfs[0]["label"])

values_testing = combined_testing.iloc[:, 1:].values
labels_testing = combined_testing.iloc[:, 0].values

label_encoder = LabelEncoder()
labels_testing = label_encoder.fit_transform(labels_testing)
labels_testing = to_categorical(labels_testing)

values_testing = np.array(values_testing)
labels_testing = np.array(labels_testing)

if os.path.exists(model_file):
    model = load_model(model_file)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
else:
    print(f"model {model_file} not found")
    exit()

loss, accuracy = model.evaluate(values_testing, labels_testing, verbose=0)
print("\nAll Feature Extractions Combined Model Results:")
print("Features are numPackets,total_data_sent,TLS_handshake_size,num_large_packets,num_small_packets,numPackets,total_data_sent,stdev_arrival_times,avg_inter_arrival_time,median_arrival_time, data sent per 0.1 second interval within the first 3 seconds")
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")