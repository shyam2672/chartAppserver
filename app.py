import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = tf.keras.models.load_model("model.h5")

print("sadsff")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image):
    # Resize image to match model input shape
    resized_image = image.resize((224, 224))
    # Convert image to numpy array
    img_array = np.asarray(resized_image)
    # Normalize pixel values to range [0, 1]
    img_array = img_array / 255.0
    # Expand dimensions to match model input shape
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.route('/')
def index():
    return "ffff"


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        print("asdfghgf")
        if 'file' not in request.files:
            return "no file"
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            # filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print(filename)

            # Load the trained model

            # Preprocess the uploaded image
            img = Image.open(filepath)
            img_array = preprocess_image(img)

            # Perform inference using the model
            predictions = model.predict(img_array)
            predictions_list = predictions.tolist()
            
            print(predictions_list)

            # Convert predictions to a human-readable format
            # (e.g., class labels, probabilities)
            # processed_output = process_predictions(predictions)

            # Return the processed output as JSON response
            return jsonify({'result': predictions_list})
        else:
            return 'Invalid file format'
    except Exception as e:
        print(e)
        return 'An error occurred'


if __name__ == '__main__':
    app.run(debug=True)
