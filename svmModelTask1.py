#preparing the data
#%%
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

#import SVM
from sklearn.svm import SVC

#transform to 2D
X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_val_flattened = X_val.reshape(X_val.shape[0], -1)

# Convert labels to a 1D array
y_train = np.argmax(y_train, axis=1)
y_val = np.argmax(y_val, axis=1)

# Train the SVM model
svm = SVC(kernel='linear')
svm.fit(X_train_flattened, y_train)

# Evaluate the model on the validation set
accuracy = svm.score(X_val_flattened, y_val)
print('Validation accuracy:', accuracy)
#Validation accuracy: 0.387027027027027 (from 0-1)