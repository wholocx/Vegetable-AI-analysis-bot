# import fastbook
# import fastai
# import os
# import pathlib
# from fastbook import *
# from fastai import *
# from flask import Flask,request,jsonify
# from PIL import Image

# app = Flask(__name__)
# root = os.getcwd()
# filename = 'trained_model.pkl'
# learn = load_learner(os.path.join(root,filename))

# print("  * loading keras model...")
# get_model(learn)


# @app.route('/')
# def index():
#     return ("Server is running")

# @app.route('/predict', methods=['POST'])
# def predict():
    
#     return single_image_predict(request.files['image'])
    

# #function to predict image
# def single_image_predict(image):
#     img_PIL = Image.open(image)
#     img = tensor(img_PIL) #converting PIL image to tensor
#     learn_inf = learn.predict(img)
#     return jsonify({'Vegetable_status': learn_inf[0][10:],
#                     'probability': str(max(learn_inf[2].numpy()))})

# if __name__=='__main__':
#     app.run(debug=True)



import base64
import numpy as np
import io
from PIL import Image
from flask import request
from flask import jsonify
from flask import Flask
import json
from pathlib import Path
import tensorflow._api.v2.compat.v1 as tf

import fastai
from fastai import *
from fastai.vision.all import *
from fastai.callback import *

app = Flask(__name__)

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# def get_model():
learn = load_learner('trained_model.pkl')

print("  * loading keras model...")
# get_model()

@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(decoded))

    img_resized = img.resize((224, 224))
    pred, pred_idx, probs = learn.predict(img_resized)

    print("pred")
    print("probs[pred_idx]")

    response = {
        'prediction' : {
            'output' : str(pred),
            'probability' : str(probs[pred_idx])
        }
    }

    return jsonify(response)

if __name__=='__main__':
    app.run(debug=True)

# link of localhost
# http://localhost:5000/static/predict.html

def predict():
    global sess
    global graph
    with graph.as_default():
        message = request.get_json(force=True)