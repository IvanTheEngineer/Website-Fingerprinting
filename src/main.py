import numpy as np

# Replace 'your_file.npy' with the path to your .npy file
length = len('features.npy')
data = np.load('features.npy')

print(data)
print(length)
