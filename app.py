from __future__ import division, print_function

import os
import pickle

import numpy as np
from flask import Flask, jsonify, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from werkzeug.utils import secure_filename

from constants import DiseaseIdentifiers, ModelPaths, ScalerPaths

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({"status": "ok", "service": "medCheck"})


@app.route('/heartDisease')
def heartDisease():
    return render_template('heartDisease.html')


@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')


@app.route('/breastCancer')
def breastCancer():
    return render_template('breastCancer.html')


@app.route('/alzheimer')
def alzheimer():
    return render_template('alzheimer.html')


@app.route('/pneumonia')
def pneumonia():
    return render_template('pneumonia.html')


@app.route('/brainTumour')
def brainTumour():
    return render_template('brainTumour.html')


@app.route('/covid')
def covid():
    return render_template('covid.html')


def model_predict(file_path, model):
    img = image.load_img(file_path, target_size=(200, 200))
    img_array = image.img_to_array(img, dtype=float)
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array, verbose=0)
    return preds


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def prediction_message(prediction_value, result):
    if prediction_value >= 0.5:
        return (
            f'The model did not flag a likely indication of {result} based on the '
            'information provided. This does not rule out a medical condition.'
        )
    return (
        f'The model flagged a possible indication associated with {result}. '
        'This is not a diagnosis. Please consult a qualified healthcare professional '
        'if you have concerns.'
    )


@app.route('/predict/<string:checkup>', methods=['POST'])
def predict(checkup):
    if checkup == DiseaseIdentifiers.HEART_DISEASE:
        model = pickle.load(open(ModelPaths.HEART_DISEASES, 'rb'))
        scaler = pickle.load(open(ScalerPaths.HEART_DISEASE_SCALER, 'rb'))
        result = 'Heart Disease'
        htmlpage = 'heartDisease.html'
    elif checkup == DiseaseIdentifiers.ALZHEIMER:
        model = pickle.load(open(ModelPaths.ALZHEIMER_MODEL, 'rb'))
        scaler = pickle.load(open(ScalerPaths.ALZHEIMER_SCALER, 'rb'))
        result = "Alzheimer's disease"
        htmlpage = 'alzheimer.html'
    elif checkup == DiseaseIdentifiers.DIABETES:
        model = pickle.load(open(ModelPaths.DIABETES_MODEL, 'rb'))
        scaler = pickle.load(open(ScalerPaths.DIABETES_SCALER, 'rb'))
        result = 'Diabetes'
        htmlpage = 'diabetes.html'
    elif checkup == DiseaseIdentifiers.BREAST_CANCER:
        model = pickle.load(open(ModelPaths.BREAST_CANCER_MODEL, 'rb'))
        scaler = pickle.load(open(ScalerPaths.BREAST_CANCER_SCALER, 'rb'))
        result = 'Breast Cancer'
        htmlpage = 'breastCancer.html'
    elif checkup == DiseaseIdentifiers.PNEUMONIA:
        model = load_model(ModelPaths.PNEUMONIA_MODEL)
        result = 'Pneumonia'
        htmlpage = 'pneumonia.html'
    elif checkup == DiseaseIdentifiers.BRAIN_TUMOR:
        model = load_model(ModelPaths.BRAIN_TUMOR_MODEL)
        result = 'Brain Tumour'
        htmlpage = 'brainTumour.html'
    elif checkup == DiseaseIdentifiers.COVID:
        model = load_model(ModelPaths.COVID_MODEL)
        result = 'COVID-19'
        htmlpage = 'covid.html'
    else:
        return 'Invalid disease type', 400

    if checkup in [DiseaseIdentifiers.HEART_DISEASE, DiseaseIdentifiers.ALZHEIMER, DiseaseIdentifiers.DIABETES, DiseaseIdentifiers.BREAST_CANCER]:
        features = [float(value) for value in request.form.values()]
        array_features = np.asarray(features, dtype=float).reshape(1, -1)
        scaled_features = scaler.transform(array_features)
        prediction = model.predict(scaled_features)
        prediction_value = float(np.squeeze(prediction))
        return render_template(htmlpage, result=prediction_message(prediction_value, result))

    file = request.files.get('file')
    if file is None or file.filename == '':
        return render_template(htmlpage, result='Please upload an image for diagnosis.')

    if not allowed_file(file.filename):
        return render_template(htmlpage, result='Please upload a valid image file (.png, .jpg, .jpeg).')

    upload_dir = os.path.join(os.path.dirname(__file__), 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    filename = secure_filename(file.filename)
    file_path = os.path.join(upload_dir, filename)
    file.save(file_path)

    preds = model_predict(file_path, model)
    prediction_array = np.asarray(preds)
    prediction_value = float(prediction_array.reshape(-1)[0])
    return render_template(htmlpage, result=prediction_message(prediction_value, result))


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
