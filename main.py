"""IMPORTS"""
import json
import os
import sys

sys.path.append("monk_v1-master/")
sys.path.append("monk_v1-master/monk/")
from monk.gluon_prototype import prototype
from elastic import connect_elasticsearch, get_last_scan_date, get_houses
from dotenv import load_dotenv
import shutil
from logger_arqisoft import BaseMLLogger, make_formatter
import requests
import concurrent.futures
import numpy as np
from keras.applications.vgg16 import VGG16
from PIL import Image
import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import pandas as pd

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


def classify_image_worker(img_to_save, description_nlp):
    name = img_to_save['path']
    img_url = "https://api.ez-assets.com/api/images/" + name
    img_data = requests.get(img_url, stream=True, timeout=5).content
    if not os.path.exists(f'{os.environ["EZA_ML_TEMP_IMAGES_PATH"]}/{name}'):
        with open(f'{os.environ["EZA_ML_TEMP_IMAGES_PATH"]}/{name}', 'wb') as handler:
            handler.write(img_data)
    first_img_class = classify_image(f'{os.environ["EZA_ML_TEMP_IMAGES_PATH"]}/{name}')
    second_img_class = make_prediction(f'{os.environ["EZA_ML_TEMP_IMAGES_PATH"]}/{name}')
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
    logger.info(f'{img_url} classified as {classes}')
    return classes, is_rehab


es = connect_elasticsearch()
last_scan_date = get_last_scan_date(es)
data = get_houses(es, last_scan_date)

sid = data["_scroll_id"]
scroll_size = data['hits']['total']['value']
all_properties = scroll_size
logger.info(f"All properties {scroll_size}")
houses_all = data['hits']['hits']
all_houses_size = len(houses_all)
rehabs = 0
while scroll_size > 0:
    if not os.path.exists(os.getenv('EZA_ML_TEMP_IMAGES_PATH')):
        os.makedirs(os.environ['EZA_ML_TEMP_IMAGES_PATH'])
    for ind, house in enumerate(houses_all):
        try:
            house = house['_source']
            alias = house.get('alias')
            imgs = house.get('images')
            description = house.get("description")
            all_images = len(house['images'])
            for index, img in enumerate(imgs):
                try:
                    img_classes = classify_image_worker(img, description)
                    house['images'][index]['classes'] = img_classes[0]
                    if img_classes[1]:
                        rehabs += 1
                except Exception as e:
                    logger.error('{}'.format(e))

            house_class = nlp_make_prediction(model_nlp, vectorizer_nlp, labels_nlp, description)
            if house_class == 'rehab' and rehab_no_rehab(rehabs, all_images):
                house['classes'] = ['rehab']
            else:
                house['classes'] = [house_class]
            rehabs = 0
            logger.info(f'house classified as {house["classes"]}')
            body = {
                "doc": house
            }
            es.update(index='aggregated-properties', id=house['id'], body=body)
            logger.info(f"{alias} - {house['id']}: {ind + 1}/{all_houses_size}")
        except:
            pass
    logger.info("scrolling...")
    page = es.scroll(scroll_id=sid, scroll=os.environ['EZA_ML_PAGE_SCROLLING'])  # page scrolling
    sid = page['_scroll_id']
    scroll_size = len(page['hits']['hits'])
    houses_all = page['hits']['hits']

    if os.path.exists(os.environ['EZA_ML_TEMP_PATH']):
        shutil.rmtree(os.environ['EZA_ML_TEMP_PATH'])

data = get_houses(es, last_scan_date)

scroll_size = data['hits']['total']['value']
if scroll_size > 2:
    logger.info(f"properties left, try to rerun project to decrease unclassified property numbers {scroll_size}")
