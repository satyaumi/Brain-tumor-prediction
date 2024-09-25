import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from flask import Flask,request,render_template,request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.utils import normalize
from  tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Precision,Recall
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.layers import Dense,Dropout,Flatten
from tensorflow.keras.models import Model

img_shape = (299, 299, 3)

base_model = tf.keras.applications.Xception(include_top=False, weights="imagenet",
                                            input_shape=img_shape, pooling='max')

# Define the final model architecture
model = Sequential([
    base_model,
    Flatten(),
    Dropout(rate=0.3),
    Dense(128, activation='relu'),
    Dropout(rate=0.25),
    Dense(4, activation='softmax')
])

# Compile the model
model.compile(Adamax(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy', Precision(), Recall()])
model.load_weights('Xception_model.keras')
app =Flask(__name__)

print('modeel loaded. Check http://127.0.0.1:5000/')


# Function to get the class name based on the predicted class index
def get_class_name(predicted_class_idx):
    class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
    return class_labels[predicted_class_idx]

def getResult(img):
    image=cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((299, 299))
    image=np.array(image)
    image=image/255.0
    input_img = np.expand_dims(image, axis=0)
    result=model.predict(input_img)
    result01=np.argmax(result,axis=1)
    return int(result01[0])


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        predicted_class_idx=getResult(file_path)
        result=get_class_name(predicted_class_idx) 
        return result
    return None

if __name__=="__main__":
    app.run(debug=True)
