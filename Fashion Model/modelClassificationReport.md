# Image Classification Model Report

## Model Overview

A convolutional neural network (CNN) was trained on the Fashion MNIST dataset for image classification. The model was trained for 5 epochs and evaluated on a held-out test set.

---

## Training and Validation Metrics

| Epoch | Training Accuracy | Training Loss | Validation Accuracy | Validation Loss |
|-------|-------------------|---------------|---------------------|-----------------|
| 1     | 0.7690            | 0.6438        | 0.8677              | 0.3726          |
| 2     | 0.8812            | 0.3250        | 0.8828              | 0.3070          |
| 3     | 0.9021            | 0.2686        | 0.9017              | 0.2746          |
| 4     | 0.9128            | 0.2382        | 0.9022              | 0.2765          |
| 5     | 0.9229            | 0.2070        | 0.9072              | 0.2574          |

---

## Test Set Performance

- **Test Accuracy:** 0.9065 (90.65%)
- **Test Loss:** 0.2665

The model demonstrates strong generalization, with test accuracy closely matching validation accuracy.

---

## Deployment Steps

### 1. Model Conversion

The trained Keras model was converted to TensorFlow Lite format for lightweight deployment:

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('fashion_mnist_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

---

### 2. Web Application Deployment with Streamlit

A Streamlit app was created to allow users to upload images and receive predictions using the `.tflite` model.

**Example `app.py` code:**
```python
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

    interpreter = tf.lite.Interpreter(model_path="fashion_mnist_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    pred = np.argmax(output, axis=1)[0]

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    st.image(image, caption=f"Prediction: {class_names[pred]}")
```

---

### 3. Running the App

- Place `fashion_mnist_model.tflite` and `app.py` in the same directory.
- Start the app with:
  ```sh
  streamlit run app.py
  ```

---

## Conclusion

The image classification model achieved over **90% accuracy** on the Fashion MNIST test set and was successfully deployed as a web application using