import os
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

# Set the directory containing the flower images
base_dir = 'Flowers\Flowers'

# Define the names of the folders containing the different flower types
flower_names = ['Babi', 'Calimerio', 'Chrysanthemum', 'Hydrangeas', 'Lisianthus', 'Pingpong', 'Rosy', 'Tana']

# Define the size of the input images
img_size = (128, 128)

# Load the flower images and labels into arrays
images = []
labels = []
for i, flower_name in enumerate(flower_names):
    folder = os.path.join(base_dir, flower_name)
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        img = Image.open(file_path)
        img = img.resize(img_size)
        x = np.array(img)
        images.append(x)
        label = to_categorical(i, num_classes=len(flower_names))
        labels.append(label)
images = np.array(images)
labels = np.array(labels)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Preprocess the images
X_train = X_train / 255.0
X_val = X_val / 255.0

#KNN import nessecary packages
from sklearn.neighbors import NearestNeighbors

# Define the number of nearby images 
k = 10

# Flatten the images
images_flattened = images.reshape(images.shape[0], -1)

#Train the model
knn = NearestNeighbors(n_neighbors=k, metric='cosine')
knn.fit(images_flattened)

# Load the input image
input_img = Image.open(os.path.join(base_dir, r"Babi\babi_1.jpg"))
input_img = input_img.resize(img_size)
input_x = np.array(input_img)
input_x_flattened = input_x.reshape(1, -1)

# Find the k-nearest neighbors of the input image in the dataset
distances, indices = knn.kneighbors(input_x_flattened)

for i in range(k):
    index = indices[0][i]
    print(i)
    print(len(flower_names))
    print(index)
    print(len(indices))
    print(indices)
    if index >= len(flower_names):
        break
    flower_name = flower_names[index] # This will result in error due to indices and different number of images in each folder. 
    print('Nearest neighbor {}: {}'.format(i+1, flower_name))
    neighbor_img = Image.fromarray(images[index])
    neighbor_img.show()
