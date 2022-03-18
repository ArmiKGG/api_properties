# -*- coding: utf-8 -*-
"""IMPORTS"""
import json
import os
import sys

sys.path.append("monk_v1-master/")
sys.path.append("monk_v1-master/monk/")
from monk.gluon_prototype import prototype
from dotenv import load_dotenv
from logger_arqisoft import BaseMLLogger, make_formatter
import requests
import numpy as np
from keras.applications.vgg16 import VGG16
from PIL import Image
import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import pandas as pd
from flask_restful import reqparse, Api, Resource
from werkzeug.security import check_password_hash
from flask import *

"""pickles (loads)"""

gtf = prototype(verbose=1)
gtf.Prototype("Task", "gluon_resnet18_v1_train_all_layers", eval_infer=True)

SIZE = 256
VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))
le = pickle.load(open('static/img_classifier_1/encoder.pkl', 'rb'))
model = pickle.load(open('static/img_classifier_2/model_v1.pkl', 'rb'))
for layer in VGG_model.layers:
    layer.trainable = False

"""Uncomment for local usage"""
nltk.download('stopwords')
nltk.download('punkt')

model_nlp = pickle.load(open('static/description_nlp_classifier/text_classifier.pickle', "rb"))
vectorizer_nlp = pickle.load(open('static/description_nlp_classifier/tfidf.pickle', "rb"))

labels_nlp = ["auction", "good", "rehab"]

make_formatter()
load_dotenv()

logger = BaseMLLogger(
    log_name='logger', log_file_name=f"eza-ml")

"""functions"""


def make_prediction(path):  # monk
    img_name = path
    predictions = gtf.Infer(img_name=img_name)
    return predictions


def classify_image(img_path):  # keras classif
    img_resized = Image.open(img_path)
    img_resized = img_resized.resize((SIZE, SIZE), Image.ANTIALIAS)
    input_img = np.expand_dims(img_resized, axis=0)
    input_img_feature = VGG_model.predict(input_img)
    input_img_features = input_img_feature.reshape(input_img_feature.shape[0], -1)
    prediction = model.predict(input_img_features)[0]
    prediction = le.inverse_transform([prediction])
    return prediction[0]


def nlp_process_text(text):  # nlp
    text = text.lower().replace('\n', ' ').replace('\r', '').strip()
    text = re.sub(' +', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)

    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if w not in stop_words]

    text = " ".join(filtered_sentence)
    return text


def rehab_no_rehab(rehabs: int, all_count: int) -> bool:
    if int(rehabs / all_count * 100) >= 10:
        return True
    return False


def nlp_make_prediction(_model, vectorizer, labels, text):  # nlp
    index_class = _model.predict(vectorizer.transform(pd.Series(text).apply(nlp_process_text)))[0]
    logger.info(f'nlp {labels[index_class]}')
    return labels[index_class]


def classify_image_worker(path_to_image, description_nlp):
    first_img_class = classify_image(path_to_image)
    second_img_class = make_prediction(path_to_image)
    third_nlp_class = nlp_make_prediction(model_nlp, vectorizer_nlp, labels_nlp, description_nlp)
    is_rehab = False
    first_class = first_img_class
    if (third_nlp_class == 'rehab' or first_img_class == 'rehab') and ('as-is' in description_nlp
                                                                       or 'investment opportunity' in description_nlp):
        is_rehab = True

    second_class, third_class = 'inside', None
    if second_img_class['predicted_class'] == 'Exterior':
        second_class = 'outside'

    else:
        third_class = second_img_class['predicted_class'].replace('_', ' ').lower()
        if third_class == 'interior':
            third_class = None
    if third_class:
        classes = [f'{first_class} {second_class}', third_class]
    else:
        classes = [f'{first_class} {second_class}']
    return {'classes': classes,
            'is_rehab': is_rehab}


app = Flask(__name__)
api = Api(app)


if not os.path.exists('./tmp'):
    os.makedirs('./tmp')


class Health(Resource):
    def get(self):
        return make_response(jsonify({"status": "OK"}), 200)

    def post(self):
        return make_response(jsonify({"status": "OK"}), 200)


class Image(Resource):

    def post(self):
        f = request.files['file']
        filename = f'./tmp/{f.filename}'
        f.save(filename)
        return make_response(jsonify(classify_image(filename)))


class Nlp(Resource):

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("nlp_desc", required=True)
        args = parser.parse_args()
        secret = args.get("nlp_desc", "")
        return {'nlp': make_response(jsonify(nlp_make_prediction(model_nlp, vectorizer_nlp, labels_nlp, secret)))}


api.add_resource(Health, '/api/', '/api/health')
api.add_resource(Image, '/api/image')
api.add_resource(Nlp, '/api/Nlp')

if __name__ == '__main__':
    app.run()
