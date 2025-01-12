import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Load the trained model
model = load_model('model.h5')  # Ensure your model file is in the same directory
IMAGE_SIZE = 128

# Define class labels
class_names = ['Class1', 'Class2', 'Class3']  # Replace with actual class names

# Function to preprocess the image and make predictions
def predict(image):
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    prediction_dict = {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}
    return prediction_dict

# Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.inputs.Image(type="pil"),
    outputs=gr.outputs.Label(num_top_classes=3),
    title="MRI Image Classifier",
    description="Upload an MRI image to classify it.",
)

if __name__ == "__main__":
    interface.launch()
