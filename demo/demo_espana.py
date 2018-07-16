# coding: utf-8

# In[16]:


import sys
import argparse
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

import numpy as np
from skimage import color, exposure, transform

from skimage import io
import os
import pandas as pd
from keras.models import load_model

from sklearn.ensemble import RandomForestClassifier
import pickle

# In[17]:


#code_path = "/home/david/Escritorio/TFG/MemoriaTFG/Presentacion/demo_espana/"
code_path="/home/drobert/tfg/traffic_sign_machine_learning/demo/"
model_path = "/home/drobert/tfg/traffic_sign_machine_learning/cnn6l/"

os.chdir(model_path)

modelname = "mini_vgg_48_epochs20_test_acc_97.66%_09-07-2018.h5"
#modelname = "rf_100trees_22depth_2fold_0.956val_acc"

# Para random forest
#model = pickle.load(open(modelname, 'rb'))

# Para otros modelos(cnn6l, nn, resnet50, xception, vgg16..)
model = load_model(modelname)


os.chdir(code_path)
# Para RF
# test_accuracy = loaded_model.predict()


NUM_CLASSES = 43
IMG_SIZE = 48  # 48
target_size = (48, 48)


# Funcion para preprocesar las imagenes
def preprocess_img(img):
    # normalizacion del histograma en el canal 'v'
    hsv = color.rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)
    # recorte del cuadrado central
    min_side = min(img.shape[:-1])
    centre = img.shape[0] // 2, img.shape[1] // 2
    img = img[centre[0] - min_side // 2:centre[0] + min_side // 2,
          centre[1] - min_side // 2:centre[1] + min_side // 2, :]

    # reescalado de imagenes a tamaño standard
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE), mode='constant')

    return img


def predict(model, img, top_n=3):
    x = img
    x = preprocess_img(x)

    # Para RF
    #x = x.reshape((-1, 48 * 48 * 3)).astype(np.float32)

    #Para modelos cnn necesitamos una entrada con 4 dimensiones
    #siendo la primera el número de imágenes
    x = np.expand_dims(x, axis=0)

    # Nos devuelve predicciones del modelo
    #preds = model.predict(x)

    #Para RF
    #proba = model.predict_proba(x)

    #Para modelos cnn
    proba = model.predict(x)

    # Toma las etiquetas codificadas que devuelve la funcion model.predict y devuelve etiquetas que se pueden entender.
    return proba #, preds


# img = Image.open('/home/david/Escritorio/TFG/MemoriaTFG/Presentacion/demo_espana/senales_img/img6.jpg')
img = Image.open(code_path+'senales_img/img01.png')


img = preprocess_img(img)
# imshow(np.asarray(img))

#proba, _ = predict(model, img)
proba = predict(model, img)


# Los ultimos indices de ind son los que más probabilidad tienen

ind = proba.argsort()
length = len(ind[0])
length = length - 1

print(ind)

# Cogemos los 5 ultimos indices (más probabilidad)
top5_index = []
for x in range(0, 5):
    top5_index.append(ind[0][length])
    length = length - 1

# Indices con más peso de mayor a menor
print(top5_index)

# Pillamos las 5 probabilidades más altas
top5_proba = []
for idx in top5_index:
    top5_proba.append(proba[0][idx])

# Probabilidades máximas
print(top5_proba)


def id_class2sign_name(id_class):
    SignNames = pd.read_csv('signnames.csv')
    id_class2sign_name = {}
    for class_id, signname in zip(list(SignNames['ClassId']), list(SignNames['SignName'])):
        id_class2sign_name.update({class_id: signname})

    sing_name = id_class2sign_name[id_class]

    return sing_name


#imshow(np.asarray(img))


sign_name_list = []
for c in range(0, 5):
    idx = top5_index[c]
    sign = id_class2sign_name(top5_index[c])
    prob = top5_proba[c]
    #print(str(idx) + ' ' + sign + '  --->  ' + str(prob * 100) + '%')
    print( str(idx) +' ' + sign + ": %.2f%%" % (prob * 100))








