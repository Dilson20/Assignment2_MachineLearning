# Assignment2_MachineLearning
 
## How to run
- Place the notebook file in the same folder with the Flower dataset folder (There are 2 Flowers dataset after unzip. Put it in the inner one)
- Delete the MAC folder
- When you already ran my code before and generated Train and Test folder and you wanna re-run, delete the Train and Test folder first

# Libraries
- There might or might not unused libraries. You can delete them if you want

# EDA
- I suggest adding another graph display 10 random images from one class (Lisianthus will be the clearest). The color from each image should be different and you can give some comments like the color feature might be misleading and grayscale can be better

# Data preprocessing
- The dataset i use only support Keras and Tensorflow. You can use SVM, RandomForest from Tensorflow for testing. The syntax is similar to SKlearn
https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/RandomForestModel
- Check the comment to see which dataset generated to use
- We can use my CNN Base model to test with different dataset to see which image resolution, grayscale/RGB will be the best. If the accuracy is not clear, try to change epoch in fit() function in every model from 10 to 20

# Base model
- I already implement the base model. If u find bug, contact me
- These codes will not take so long to run (about few hours in the worst case)

Cheers !!
