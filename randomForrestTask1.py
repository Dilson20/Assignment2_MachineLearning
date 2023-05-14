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

#import random forrest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

#transform to 2D
X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_val_flattened = X_val.reshape(X_val.shape[0], -1)

randForrest = RandomForestClassifier(100)

randForrest.fit(X_train_flattened, y_train) #epoch

score = randForrest.score(X_val_flattened, y_val)
print("Validation accuracy: {:.2f}%".format(score*100))
#Accuracy around 15%, so not great