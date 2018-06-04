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
from keras.models import load_model
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

#Corleone
code_path="/home/drobert/tfg/traffic_sign_machine_learning/nn/"
dataset_path='/home/drobert/tfg/'

modelo="nn_3H_32"

#fichero_log = ('/home/drobert/tfg/traffic_sign_machine_learning/cnn6l/cnn6l.log')
fichero_log = (code_path +modelo+'.log')



print('Archivo Log en ', fichero_log)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    filename = fichero_log,
                    filemode = 'w',)# w for new log each time/a for over write


print ("[STATUS] --------"+modelo+"- start time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
logging.info(" ---------"+modelo+"- start time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))



NUM_CLASSES = 43
IMG_SIZE = 32 #48


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

modelname = "nn_4H_32_epochs20_test_acc_88.55%_04-06-2018.h5"

#Para random forest
#loaded_model = pickle.load(open(modelname, 'rb'))

#Para otros modelos(cnn6l, nn, resnet50, xception, vgg16..)
loaded_model = load_model(modelname)


os.chdir(dataset_path+'GTSRB')#En corleone

# Cargamos el archivo csv con los datos de test y vemos que contienen los 10 primeros
test = pd.read_csv('GT-final_test.csv', sep=';')

print('Cargando imagenes de test ...')
logging.info('Cargando imagenes de test ...')
# Cargamos el dataset de test
#os.chdir('/home/david/Escritorio/TFG/Pruebas/GTSRB/Final_Test/Images/')
os.chdir(dataset_path+'GTSRB/Final_Test/Images/')#en corleone

X_test = []
y_test = []
i = 0
for file_name, class_id in zip(list(test['Filename']), list(test['ClassId'])):
    img_path = os.path.join(os.getcwd(), file_name)
    X_test.append(preprocess_img(io.imread(img_path)))
    y_test.append(class_id)

X_test = np.array(X_test)
y_test = np.array(y_test)

#Vamos a dividir el conjunto de test en 5

kf = KFold(n_splits=5,random_state=13) # Define the split - into 2 folds
kf.get_n_splits(X_test) # returns the number of splitting iterations in the cross-validator

print(kf)

fold =1
accuracy_list = []

for train_index, test_index in kf.split(X_test):
    _, X_test_fold = X_test[train_index], X_test[test_index]
    _, y_test_fold = y_test[train_index], y_test[test_index]



    # Cambiamos los formatos de entrada de las imagenes para que sea una matriz bidimensional
    X_test_fold = X_test_fold.reshape((-1, IMG_SIZE * IMG_SIZE * 3)).astype(np.float32)

    # --Para resto de modelos-- Los targets tienen que estar en formato one target
    y_test_fold_one_target = np.eye(NUM_CLASSES, dtype='uint8')[y_test_fold]

    # Para RF
    # test_accuracy = loaded_model.score(X_test, y_test)

    # Para resto de modelos
    test_accuracy = loaded_model.evaluate(X_test_fold, y_test_fold_one_target, verbose=1)
    accuracy_list.append(test_accuracy[1] * 100)

    print(str(fold)+"Resultado final del modelo en test: %.2f%% " % (test_accuracy[1] * 100))
    logging.info(str(fold)+"Resultado final del modelo en test: %.2f%% " % (test_accuracy[1] * 100))

    fold += 1


precision_media = (np.mean(accuracy_list))
desviacion_standar = (np.std(accuracy_list))

print("mean_accuarcy de " +modelo+": %.2f%% (+/- %.2f%%)" % (np.mean(accuracy_list), np.std(accuracy_list)))
logging.info("mean_accuarcy de " +modelo+": %.2f%% (+/- %.2f%%)" % (np.mean(accuracy_list), np.std(accuracy_list)))

