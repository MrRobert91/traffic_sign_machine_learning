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
code_path="/home/drobert/tfg/traffic_sign_machine_learning/cnn6l/"
dataset_path='/home/drobert/tfg/'

modelo="cnn_skip_conect_v1_32"

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

modelname = "cnn_multi_scale_epochs30_test_acc_94.83%_24-05-2018.h5"

#Para random forest
#loaded_model = pickle.load(open(modelname, 'rb'))

#Para otros modelos(cnn6l, nn, resnet50, xception, vgg16..)
loaded_model = load_model(modelname)


# diccionario las keys son clases del dataset aleman y los valores del belga
ger2bel_dic = { '00009': '00031','00011': '00017','00012': '00061','00013': '00019',
                '00014': '00021','00015': '00028','00016': '00025','00017': '00022',
                '00018': '00013','00019': '00003','00020': '00004','00021': '00005',
                '00022': '00000','00023': '00002','00024': '00016','00025': '00010',
                '00028': '00007','00029': '00008','00035': '00034','00038': '00035',
                '00040': '00037'}

# diccionario las keys son clases del dataset belga y los valores del aleman
bel2ger_dic = { '00031': '00009' ,'00017': '00011' ,'00061': '00012' ,'00019': '00013' ,
                '00021': '00014' ,'00028': '00015' ,'00025': '00016' ,'00022': '00017' ,
                '00013': '00018' ,'00003': '00019' ,'00004': '00020' ,'00005': '00021' ,
                '00000': '00022' ,'00002': '00023' ,'00016': '00024' ,'00010': '00025' ,
                '00007': '00028' ,'00008': '00029' ,'00034': '00035' ,'00035': '00038' ,
                '00037': '00040', }


def belgium2german(img_path):
    belgium_class = str(img_path.split('/')[-2])
    if belgium_class in bel2ger_dic:
        german_class = bel2ger_dic[belgium_class]
        return german_class
    else:
        return None

def get_class(img_path):
    return int(img_path.split('/')[-2])

#Cargamos las imagenes del dataset de test belga
os.chdir(dataset_path) #direccion local Jupyter Notebooks/pycharm
root_dir = 'belgium_test/'


imgs = []
labels = []

ruta_actual = os.getcwd()
print(ruta_actual)

all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))

print(os.path.join(root_dir, '*/*.ppm'))
print(len(all_img_paths))

np.random.shuffle(all_img_paths)

for img_path in all_img_paths:
    img = preprocess_img(io.imread(img_path))
    label_g = belgium2german(img_path)
    if label_g is not None:
        imgs.append(img)
        labels.append(label_g)

X_test = np.array(imgs, dtype='float32')
y_test = np.asarray(labels)

print(X_test.shape)
print(y_test.shape)

#Vamos a dividir el conjunto de test en 5

kf = KFold(n_splits=5,random_state=13) # Define the split - into 2 folds
kf.get_n_splits(X_test) # returns the number of splitting iterations in the cross-validator

print(kf)

fold =1
accuracy_list = []

for train_index, test_index in kf.split(X_test):
    _, X_test_fold = X_test[train_index], X_test[test_index]
    _, y_test_fold = y_test[train_index], y_test[test_index]


    # --Para Random Forest-- Cambiamos los formatos de entrada de las imagenes para que sea una matriz bidimensional
    #X_test = X_test.reshape((-1, 48 * 48 * 3)).astype(np.float32)

    # --Para resto de modelos-- Los targets tienen que estar en formato one target
    y_test_fold_one_target = np.eye(NUM_CLASSES, dtype='uint8')[y_test_fold]

    # Para RF
    # test_accuracy = loaded_model.score(X_test, y_test)

    # Para resto de modelos
    test_accuracy = loaded_model.evaluate(X_test_fold, y_test_fold_one_target, verbose=1)
    accuracy_list.append(test_accuracy[1] * 100)

    print(str(fold)+" Resultado final del modelo en test: %.2f%% " % (test_accuracy[1] * 100))
    logging.info(str(fold)+" Resultado final del modelo en test: %.2f%% " % (test_accuracy[1] * 100))

    fold += 1


precision_media = (np.mean(accuracy_list))
desviacion_standar = (np.std(accuracy_list))

print("mean_accuarcy de" +modelo+": %.2f%% (+/- %.2f%%)" % (np.mean(accuracy_list), np.std(accuracy_list)))
logging.info("mean_accuarcy de " +modelo+": %.2f%% (+/- %.2f%%)" % (np.mean(accuracy_list), np.std(accuracy_list)))

