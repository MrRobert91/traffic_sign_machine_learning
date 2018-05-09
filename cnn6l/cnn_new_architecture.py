# coding: utf-8

# Version sistematizada:
# 1- preprocesado
# 2- divsión del dataset en entrenamiento y validación
# 3- validacion cruzada estratificada(10 fold)
# 4- media y desviación de las matrices para cada fold
# 5- quedarse con el módelo más próximo al modelo promedio.
# 6- guardar los resultados y los hiperparametros. (diccionario, csv, ...)

# #Random forest para clasificar señales de tráfico
#
# Vamos a utilizar una red neuronal convolucional para clasificar
# imagenes de señales de tráfico del
# dataset [GTSRB](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)

# In[1]:


import numpy as np
from skimage import color, exposure, transform

from skimage import io
import os
import glob

import pandas as pd
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pickle
from sklearn.model_selection import StratifiedKFold

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K
from keras import layers
from keras.models import Model

from keras.layers import Input, Dense
K.set_image_data_format('channels_last')

from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import os, logging
from keras import metrics
from keras.models import load_model
import datetime
from skimage import color
from skimage import io
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils.np_utils import to_categorical
import json
from sklearn.model_selection import train_test_split

# load the user configs
with open('conf.json') as f:
	config = json.load(f)

# config variables
model_name 		= config["model"]
weights 		= config["weights"]
include_top 	= config["include_top"]
train_path 		= config["train_path"]
test_path       = config["test_path"]
features_path 	= config["features_path"]
labels_path 	= config["labels_path"]
test_size 		= config["test_size"]
results 		= config["results"]
model_path 		= config["model_path"]
seed      		= config["seed"]
classifier_path = config["classifier_path"]
log_path		= config["log_path"]


#local
#code_path= "/home/david/PycharmProjects/traffic_sign_machine_learning/cnn6l/"
#dataset_path="/home/david/Escritorio/TFG/Pruebas"

#Corleone
code_path="/home/drobert/tfg/traffic_sign_machine_learning/cnn6l/"
dataset_path='/home/drobert/tfg/'

#fichero_log = ('/home/drobert/tfg/traffic_sign_machine_learning/cnn6l/cnn6l.log')
fichero_log = (code_path +'cnn_new_arch.log')


print('Archivo Log en ', fichero_log)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    filename = fichero_log,
                    filemode = 'a',)# w for new log each time


print ("[STATUS] --------cnn new architecture - start time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
logging.info(" ---------cnn new architecture - start time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))


def cnn_model_res_multi_v2():
    input_tensor = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='4d_input')

    #1ª Etapa
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu') (input_tensor)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.5)(x)
    x_flatten_1 = layers.Flatten()(x)

    #2ª Etapa
    x_principal = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x_principal = layers.MaxPooling2D(pool_size=(2, 2))(x_principal)
    x_principal = layers.Dropout(0.5)(x_principal)
    x_flatten_2 = layers.Flatten()(x_principal)

    #3ª Etapa
    x_principal = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x_principal)
    x_principal = layers.MaxPooling2D(pool_size=(2, 2))(x_principal)
    x_principal = layers.Dropout(0.5)(x_principal)
    x_flatten_3 = layers.Flatten()(x_principal)

    # Etapa de concatenacion
    concatenated = layers.concatenate([x_flatten_3, x_flatten_1, x_flatten_2],axis=-1)#probar tambien con add
    concatenated = layers.Dense(512, activation='relu')(concatenated)
    concatenated = layers.Dropout(0.5)(concatenated)

    output_tensor = layers.Dense(NUM_CLASSES, activation='softmax')(concatenated)
    model = Model(input_tensor, output_tensor)
    return model
#Modelo: original
def cnn_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(IMG_SIZE, IMG_SIZE, 3),
                     activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))#antes 0.2

    model.add(Conv2D(64, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))#antes 0.2

    model.add(Conv2D(128, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))#antes 0.2

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model

#Modelo: red neuronal con 6 capas convolucionales
def cnn_model_v2():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',
                     #input_shape=(IMG_SIZE, IMG_SIZE, 3), #imagenes a color
                     input_shape=(IMG_SIZE, IMG_SIZE, 3),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, (3, 3), padding='same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))#antes 0.2


    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model


NUM_CLASSES = 43
IMG_SIZE = 48 # Como se sugiere en el paper de LeCun

batch_size = 32 #16
epochs = 30 #30 o 50
lr = 0.01

def lr_schedule(epoch):
    return lr * (0.1 ** int(epoch / 10))

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
          centre[1] - min_side // 2:centre[1] + min_side // 2,
          :]

    # reescalado de imagenes a tamaño standard
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE), mode='constant')

    return img


def get_class(img_path):
    return int(img_path.split('/')[-2])


os.chdir(dataset_path) #direccion local Jupyter Notebooks/pycharm
root_dir = 'GTSRB/Final_Training/Images/'


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
    label = get_class(img_path)
    imgs.append(img)
    labels.append(label)

X = np.array(imgs, dtype='float32')
Y = np.asarray(labels)

print(X.shape)
print(Y.shape)

logging.info(X.shape)
logging.info(Y.shape)

X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.25, random_state=42)


train_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=preprocess_img)#

test_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=preprocess_img)

'''
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(32, 32),
    color_mode='rgb',#or 'greyscale'
    batch_size=32,
    class_mode='categorical')
'''
train_generator = train_datagen.flow(
    X_train,
    y_train,
    batch_size=32)

'''
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(32, 32),
    color_mode='rgb',#or 'greyscale'
    batch_size=32,
    class_mode='categorical')
'''

val_generator = test_datagen.flow(
    X_val,
    y_val,
    batch_size=32)


model = cnn_model_v2()

sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(
    loss='categorical_crossentropy',
    optimizer=sgd,
    metrics=[metrics.categorical_accuracy])

history = model.fit_generator(
    train_generator,
    #steps_per_epoch=100,
    steps_per_epoch=(31367 / batch_size), #39209*0,8 = 31367 train_data length
    epochs=30,
    verbose=1,
    validation_data=val_generator,
    validation_steps=50,
    callbacks=[LearningRateScheduler(lr_schedule)])






os.chdir(dataset_path+'/GTSRB')#En local

# Cargamos el archivo csv con los datos de test y vemos que contienen los 10 primeros
test = pd.read_csv('GT-final_test.csv', sep=';')

# Cargamos el dataset de test
os.chdir(dataset_path+'/GTSRB/Final_Test/Images/')#en local
#os.chdir('/home/drobert/tfg/GTSRB/Final_Test/Images/')#en corleone

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


#Los targets tienen que estar en formato one target
y_test_one_target = np.eye(NUM_CLASSES, dtype='uint8')[y_test]


test_accuracy = model.evaluate(X_test, y_test_one_target, verbose=1)

today_date = datetime.date.today().strftime("%d-%m-%Y")
model_filename= ("cnn_new_architecture_epochs%s_test_acc_%.2f%%_%s.h5" % (epochs,test_accuracy[1] * 100, today_date))

print("Accuracy en test : %s: %.2f%%" % (model.metrics_names[1], test_accuracy[1] * 100))
logging.info("Accuracy en test : %s: %.2f%%" % (model.metrics_names[1], test_accuracy[1] * 100))


model.save(model_filename)

loaded_model = load_model(model_filename)# No funciona con custom metrics

loaded_model_test_accuracy = loaded_model.evaluate(X_test, y_test_one_target, verbose=1)
print("Loaded_model accuracy en test : %s: %.2f%%" % (loaded_model.metrics_names[1], loaded_model_test_accuracy[1] * 100))


print("Fin de la prueba con CNN new architecture")
logging.info("-----------Fin de la prueba con CNN new architecture-----------")
logging.info("program ended on - " + str(datetime.datetime.now))
