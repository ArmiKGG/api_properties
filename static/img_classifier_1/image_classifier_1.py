import numpy as np
import pickle
from keras.applications.vgg16 import VGG16
from PIL import Image

SIZE = 256
VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))
le = pickle.load(open('encoder.pkl', 'rb'))
model = pickle.load(open('../img_classifier_2/model_v1.pkl', 'rb'))
for layer in VGG_model.layers:
    layer.trainable = False


def classify_image(img_path):
    img = Image.open(img_path)
    img = img.resize((SIZE, SIZE), Image.ANTIALIAS)
    input_img = np.expand_dims(img, axis=0)
    input_img_feature = VGG_model.predict(input_img)
    input_img_features = input_img_feature.reshape(input_img_feature.shape[0], -1)
    prediction = model.predict(input_img_features)[0]
    prediction = le.inverse_transform([prediction])
    return prediction[0]


classify_image('../test.jpg')