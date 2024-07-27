## Plant Leaf Disease Detection

##### About Dataset
This dataset is recreated using offline augmentation from the original dataset.
The original dataset can be found on this github repo.
This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes. 
The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
A new directory containing 33 test images is created later for prediction purpose.

The dataset is divided into three parts:
- `train`: Used for training the model.
- `valid`: Used for validating the model during training.
- `test`: Used for testing the model after training.

Project Description
Plant Leaf Disease Detection
This project involves developing a machine learning model to identify and classify plant leaf diseases from images using Convolutional Neural Networks (CNNs). 
The goal is to create a model that can accurately classify plant leaf images into various disease categories, 
assisting farmers and agricultural professionals in early detection and management of plant diseases.

Objectives
Data Collection and Preprocessing:

Collect a dataset of plant leaf images categorized into various disease classes.
Preprocess the images to ensure consistent input size and normalization, and apply data augmentation techniques to enhance model generalization.
Model Development:

Develop a CNN model using TensorFlow and Keras to classify plant leaf images.
Build a deep learning architecture consisting of multiple convolutional layers, max-pooling layers, and dense layers.
Model Training and Evaluation:

The training data is used to train the CNN model. The model is trained using the following parameters:
- **Image Size**: 256x256
- **Batch Size**: 32
- **Epochs**: 10
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy

Train the model using the training dataset and validate it using a separate validation dataset.
Evaluate the model performance on a test dataset to assess accuracy, recall, and F1 score.
Deployment:

Deploy the trained model using Streamlit to create an interactive web application.
Implement a user-friendly interface that allows users to upload leaf images and receive predictions on disease classification.
Dataset
The dataset used in this project is sourced from the PlantVillage dataset, which includes images of plant leaves categorized into various disease classes. The dataset is divided into three parts:

Training Set: Used for training the model.
Validation Set: Used to tune model parameters and prevent overfitting.
Test Set: Used to evaluate the final performance of the model.
Model Architecture
The CNN model developed for this project includes the following layers:

****  Preprocessing Layers:

Resizing and rescaling of images to ensure uniform input dimensions.
Convolutional Layers:

Multiple convolutional layers with ReLU activation functions to extract features from images.
Max-Pooling Layers:

Downsampling layers to reduce the spatial dimensions and capture essential features.
Fully Connected Layers:

Dense layers to classify the extracted features into disease categories.
Output Layer:

A dense layer with softmax activation to produce probability scores for each class.
Training and Evaluation
Training: The model is trained for 10 epochs using the Adam optimizer and Sparse Categorical Crossentropy loss function.
Validation: Early stopping and model checkpoint callbacks are used to save the best model based on validation loss.
Testing: The model is evaluated on the test set, achieving an accuracy of 96%, recall of 95%, and F1 score of 95%.

Results
The model achieved the following metrics on the test set:

Accuracy: 96%
Recall: 95%
F1 Score: 95%

###  Streamlit App Deployment
The project includes a Streamlit web application that allows users to upload images of plant leaves and receive predictions on the presence of diseases.
The app provides a simple interface for interacting with the model:

Upload: Users can drag and drop or upload an image file.
Prediction: The model predicts the disease category and displays the result along with the uploaded image.
