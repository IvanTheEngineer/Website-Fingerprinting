import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

# Load model, training and testing datasets
# Datasets for aggregated data dataset were already normalized using min-max so it doesn't need to be done again
training_file = "output/aggregated_data_training_normalized"
testing_file = "output/aggregated_data_testing_normalized"
model_file = "trained_model_aggregated_data.keras"
columns = ["label"] + [f"data{i+1}" for i in range(30)]
data_training = pd.read_csv(training_file, sep=" ", header=None, names=columns)
data_testing = pd.read_csv(testing_file, sep=" ", header=None, names=columns)

# Separate features and labels for training and testing
values_training = data_training.iloc[:, 1:].values
labels_training = data_training.iloc[:, 0].values

values_testing = data_testing.iloc[:, 1:].values
labels_testing = data_testing.iloc[:, 0].values

# print(pd.Series(labels_train).value_counts())

# Split training dataset in 3:1 ratio to training and validation
# validation serves as an independent dataset during training
values_training, values_validation, labels_training, labels_validation = train_test_split(
    values_training, labels_training,
    test_size=0.25,
    random_state=42,
    stratify=labels_training
)

# Encode labels as integers
label_encoder = LabelEncoder()
labels_training = label_encoder.fit_transform(labels_training)
labels_validation = label_encoder.transform(labels_validation)
labels_testing = label_encoder.transform(labels_testing)
labels_training = to_categorical(labels_training)
labels_validation = to_categorical(labels_validation)
labels_testing = to_categorical(labels_testing)

# Convert to np array
values_training = np.array(values_training)
labels_training = np.array(labels_training)
values_validation = np.array(values_validation)
labels_validation = np.array(labels_validation)
values_testing = np.array(values_testing)
labels_testing = np.array(labels_testing)

# Check if a saved model exists
if os.path.exists(model_file):
    model = load_model(model_file)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])  # Recompile the model
else:
    model = Sequential([
        Dense(128, activation="relu", input_shape=(values_training.shape[1],)),
        Dense(64, activation="relu"),                              
        Dense(32, activation="relu"),                         
        Dense(labels_training.shape[1], activation="softmax")            
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

# Train model
history = model.fit(values_training, labels_training, validation_data=(values_validation, labels_validation), epochs=100, batch_size=32, verbose=1)

# Evaluate on testing dataset
loss, accuracy = model.evaluate(values_testing, labels_testing, verbose=0)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

model.save(model_file)
print("Model saved")
