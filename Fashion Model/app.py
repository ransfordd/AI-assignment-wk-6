import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Fashion MNIST Image Classifier (.tflite)")

uploaded_file = st.file_uploader("Choose an image...", type="png")
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L').resize((28,28))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1,28,28,1).astype('float32')

    # Load the TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path="fashion_mnist_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    pred = np.argmax(output, axis=1)[0]

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    st.image(image, caption=f"Prediction: {class_names[pred]}")