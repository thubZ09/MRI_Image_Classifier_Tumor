## MRI Image Classifier
This repository contains a project for classifying MRI images into different categories using a deep learning model based on the VGG16 architecture. The project is implemented in Python using TensorFlow.

##✅ Project Structure

✔️Files and Directories:
model.h5: The pre-trained TensorFlow model for MRI classification.

✔️requirements.txt: 
Contains the Python dependencies required to run the project.

✔️.gitignore: 
Specifies files and directories to exclude from Git tracking.

✔️Directories:
Training/: Contains training images categorized into subfolders by label.
Testing/: Contains testing images categorized into subfolders by label.

##✅ Dependencies

✔️Ensure you have the following libraries installed:
TensorFlow
NumPy
Matplotlib
Pillow
Scikit-learn
Seaborn

✔️Install the required dependencies using: pip install -r requirements.txt

##✅Model Details

✔️Base Model:
VGG16: A pre-trained convolutional neural network (CNN) model from TensorFlow's applications module.

✔️Input shape: 
(128, 128, 3)

✔️Pre-trained on ImageNet
✔️Last three convolutional layers are unfrozen for fine-tuning.

✔️Custom Layers:
Flatten layer to convert feature maps into a single vector.

✔️Fully connected dense layer with ReLU activation.
✔️Dropout layers for regularization.
✔️Output layer with a softmax activation function for classification.

✔️Hyperparameters:
Optimizer: 
Adam with a learning rate of 0.0001.

Loss Function: 
Sparse Categorical Crossentropy.

Metrics: 
Sparse Categorical Accuracy.

##✅ Run the Project

✔️Clone the Repository
✔️Train the Model
If you want to retrain the model, ensure you have the training data in the correct folder structure and modify the train_dir and test_dir variables in the script accordingly.
✔️Run the training script in an environment with the required libraries installed.

##✅Features
✔️Model Training and Testing
Loads and preprocesses MRI images.
Applies data augmentation (brightness and contrast adjustments).
Trains a fine-tuned VGG16 model.

![image](https://github.com/user-attachments/assets/4e2b0b7a-2032-476e-9b40-92007bdcaba4)

✔️Evaluation Metrics

   ![image](https://github.com/user-attachments/assets/9c9887f8-6a9c-4762-ae2e-35cb0924b02f)

The script includes functions for model evaluation:

Training and Validation Plots: Visualize accuracy and loss.
Classification Report: Precision, recall, and F1 score.
Confusion Matrix: Detailed view of predictions.
ROC Curve and AUC: Performance of the classifier.







