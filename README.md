## MRI Image Classifier
This repository contains a project for classifying MRI images into different categories using a deep learning model based on the VGG16 architecture. The project is implemented in Python using TensorFlow.

✅ Project Structure

✔️Files and Directories:
model.h5: The pre-trained TensorFlow model for MRI classification.

✔️requirements.txt: 
Contains the Python dependencies required to run the project.

✔️.gitignore: 
Specifies files and directories to exclude from Git tracking.

✔️Directories:
Training/: Contains training images categorized into subfolders by label.
Testing/: Contains testing images categorized into subfolders by label.


✅ Dependencies

✔️Ensure you have the following libraries installed:
TensorFlow
NumPy
Matplotlib
Pillow
Scikit-learn
Seaborn

✔️Install the required dependencies using: pip install -r requirements.txt


✅Model Details

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


✅ Run the Project

✔️Clone the Repository
✔️Train the Model
If you want to retrain the model, ensure you have the training data in the correct folder structure and modify the train_dir and test_dir variables in the script accordingly.
✔️Run the training script in an environment with the required libraries installed.


✅Features
✔️Model Training and Testing
Loads and preprocesses MRI images.
Applies data augmentation (brightness and contrast adjustments).
Trains a fine-tuned VGG16 model.

✔️Evaluation Metrics
The script includes functions for model evaluation:

Training and Validation Plots: Visualize accuracy and loss.
Classification Report: Precision, recall, and F1 score.
Confusion Matrix: Detailed view of predictions.
ROC Curve and AUC: Performance of the classifier.


✅Model Evaluation

✔️Training and Validation Plots:

Visualize training history:

![image](https://github.com/user-attachments/assets/4e2b0b7a-2032-476e-9b40-92007bdcaba4)


✔️Classification Report and Confusion Matrix:

<img width="509" alt="Screenshot 2025-01-12 112401" src="https://github.com/user-attachments/assets/6cee3fba-cf2e-4e74-99f6-df9397cdd2d4" />


✔️Evaluate model predictions on the test set:

![image](https://github.com/user-attachments/assets/efd06eed-4b97-42f5-8880-90c62b2e4234)


✔️ROC Curve:

![image](https://github.com/user-attachments/assets/cb246c92-35e0-4efd-9be9-bea1439d3cbb)









