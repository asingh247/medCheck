from __future__ import division, print_function
import os
import pickle
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import Model, load_model
from keras.preprocessing import image

#Flask utils
from flask import Flask, render_template, request, url_for, jsonify, Response
from gevent.pywsgi import WSGIServer
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

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
    
    xtest_image = image.image_utils.load_img(file_path,target_size=(200, 200))
    xtest_image = image.image_utils.img_to_array(xtest_image,dtype=float)
    xtest_image = np.expand_dims(xtest_image, axis=0)
    preds = Model.predict(model ,xtest_image)
    return preds


@app.route('/predict/<string:checkup>', methods =['POST'])
def predict(checkup):
    if checkup=="heartDisease":
        model =  pickle.load(open('static/models/HeartDiseaseSVM.pkl', 'rb')) 
        scaler = pickle.load(open('static/scalers/heartDiseaseScaler.sav', 'rb')) 
        result = "Heart Disease"
        htmlpage = "heartDisease.html"
    elif checkup=="alzheimer":
        model = pickle.load(open('static/models/AlzheimerRfc.pkl', 'rb'))
        scaler = pickle.load(open('static/scalers/alzheimerScaler.sav','rb'))
        result = "Alzheimer"
        htmlpage = "alzheimer.html"
    elif checkup=="diabetes":
        model = pickle.load(open('static/models/diabetesClf.pkl', 'rb'))
        scaler = pickle.load(open('static/scalers/diabetesScaler.sav', 'rb'))
        result ="Diabetes"
        htmlpage ="diabetes.html"
    elif checkup=="breastCancer":
        model =  pickle.load(open('static/models/breastCancerRfc.pkl', 'rb'))
        scaler = pickle.load(open('static/scalers/breastCancerScaler.sav', 'rb'))
        result ="Breast Cancer"
        htmlpage ="breastCancer.html"
    elif checkup=="pnuemonia":
        model = load_model('static/models/PneumoniaDetection.h5') 
        Model.make_predict_function(model)
        result = "Pneumonia"
        htmlpage = "pneumonia.html"
    elif checkup=="brainTumour":
        model = load_model('static/models/BrainTumorDetection.h5') 
        Model.make_predict_function(model)
        result = "Brain Tumour"
        htmlpage = "brainTumour.html"
    elif checkup=="covid":
        model = load_model('static/models/CovidDetection.h5') 
        Model.make_predict_function(model)
        result = "Covid-19"
        htmlpage = "covid.html"

    if checkup in ['heartDisease','alzheimer','diabetes','breastCancer']:
        features = [float(i) for i in request.form.values()]
        array_features = [np.array(features)]
        scaled_features = scaler.transform(array_features)
        prediction = model.predict(scaled_features)
        output = prediction

        if output == 1:
            return render_template(htmlpage, 
                               result = f'The patient is not likely to have {result} and maintain current health!')
        else:
            return render_template(htmlpage, 
                               result = f'The patient is likely to have {result} and must consult a doctor for diagnosis!')
    
    else:

        f= request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        # img = Image.open(file_path)
        preds = model_predict(file_path, model)
        

        if preds[0][0]==0:
            return render_template(htmlpage, result = f'The patient is not likely to have {result} and maintain current health!')
        else:
            return render_template(htmlpage, result = f'The patient is likely to have {result} and must consult a doctor for diagnosis!')

if __name__ == '__main__':
    app.run(debug=True)    