import numpy as np
from skimage import color, exposure, transform

from skimage import io
import os
import glob

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.model_selection import StratifiedKFold
import datetime
import logging

#Corleone
code_path="/home/drobert/tfg/traffic_sign_machine_learning/rf/"
dataset_path='/home/drobert/tfg/'

NUM_CLASSES = 43
IMG_SIZE = 48
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

os.chdir(code_path)
#modelname = "rf_500trees_2fold_0.968val_acc"
modelname = "rf_300trees_12depth_3fold_0.902val_acc"

loaded_model = pickle.load(open(modelname, 'rb'))

#os.chdir('/home/david/Escritorio/TFG/Pruebas/GTSRB')
os.chdir(dataset_path+'GTSRB')#En corleone

# Cargamos el archivo csv con los datos de test y vemos que contienen los 10 primeros
test = pd.read_csv('GT-final_test.csv', sep=';')
#test.head(10)

# In[61]:

print('Cargando imagenes de test ...')
logging.info('Cargando imagenes de test ...')
# Cargamos el dataset de test
#os.chdir('/home/david/Escritorio/TFG/Pruebas/GTSRB/Final_Test/Images/')
os.chdir(dataset_path+'GTSRB/Final_Test/Images/')#en corleone

X_test = []
y_test = []
i = 0
for file_name, class_id in zip(list(test['Filename']), list(test['ClassId'])):
    # img_path = os.path.join('GTSRB/Final_Test/Images/', file_name)
    img_path = os.path.join(os.getcwd(), file_name)
    X_test.append(preprocess_img(io.imread(img_path)))
    y_test.append(class_id)

X_test = np.array(X_test)
y_test = np.array(y_test)

# Cambiamos los formatos de entrada de las imagenes para que sea una matriz bidimensional
X_test = X_test.reshape((-1, 48 * 48 * 3)).astype(np.float32)



result = loaded_model.score(X_test, y_test)
print("Resultado final del modelo en test: %.2f%% " % (result*100))
logging.info("Resultado final del modelo en test: %.2f%% " % (result*100))