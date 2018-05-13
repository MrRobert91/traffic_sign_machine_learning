# filter warnings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# keras imports
from keras.applications.vgg16 import VGG16, preprocess_input

from keras.preprocessing import image
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Input

import numpy as np
from skimage import color, exposure, transform

from skimage import io


from keras import backend as K
K.set_image_data_format('channels_last')

from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import os, logging
from keras import metrics
from keras.models import load_model
import datetime
# other imports
from sklearn.preprocessing import LabelEncoder
import numpy as np
import glob
#import cv2
import h5py
import os
import json
import datetime
import time
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pickle
import logging

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import RMSprop
from keras import metrics
from keras.optimizers import SGD

from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.utils.np_utils import to_categorical

from keras.callbacks import TensorBoard

# load the user configs
with open('conf.json') as f:
	config = json.load(f)

# config variables
model_name 		= config["model"]
weights 		= config["weights"]
include_top 	= config["include_top"]
train_path 		= config["train_path"]
features_path 	= config["features_path"]
labels_path 	= config["labels_path"]
test_size 		= config["test_size"]
results 		= config["results"]
model_path 		= config["model_path"]
seed      		= config["seed"]
classifier_path = config["classifier_path"]
log_path		= config["log_path"]


#Corleone
code_path="/home/drobert/tfg/traffic_sign_machine_learning/vgg16/"
dataset_path='/home/drobert/tfg/'

fichero_log = (log_path)
fichero_log_tb = (code_path +'tb_vgg16_feature_extraction.log')

print('Archivo Log en ', fichero_log)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    filename = fichero_log,
                    filemode = 'a',)
# start time
print ("[STATUS]-------vgg16 systematized with nn - start time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
logging.info(" --------vgg16 systematized with nn - start time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
start = time.time()





# create the pretrained models
# check for pretrained weight usage or not
# check for top layers to be included or not
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
          centre[1] - min_side // 2:centre[1] + min_side // 2,
          :]

    # reescalado de imagenes a tamaño standard
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE), mode='constant')

    return img
#cargar imagenes

def get_class(img_path):
    return int(img_path.split('/')[-2])


os.chdir(dataset_path) #direccion local Jupyter Notebooks/pycharm
root_dir = 'GTSRB/Final_Training/Images/'

#os.chdir('/home/drobert/tfg/')#direccion en corleone
#root_dir = 'GTSRB/Final_Training/Images/'

top_model_weights_path = "/home/drobert/tfg/traffic_sign_machine_learning/vgg16/output/vgg16_top_classifier.h5"

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

X_train, X_val, y_train, y_val = train_test_split(
    X ,Y,
    test_size=0.20,
    random_state=42)

y_train_one_hot = to_categorical(y_train, NUM_CLASSES)
y_val_one_hot = to_categorical(y_val, NUM_CLASSES)

batch_size = 32
epochs = 20 #ponemos 5 para que sea mas rapido, normalmente 30
lr = 0.01

def lr_schedule(epoch):
    return lr * (0.1 ** int(epoch / 10))

tensorboard = TensorBoard(log_dir=fichero_log_tb, histogram_freq = 1, write_graph =True)

# creating the final model
#model_final = Model(input = model.input, output = predictions)
#-------------Extract features----------------------

# build the VGG16 network
base_model = VGG16(include_top=False, weights='imagenet')


bottleneck_features_train = base_model.predict(X_train)

np.save(open('bottleneck_features_train.npy', 'wb'),
            bottleneck_features_train)

bottleneck_features_validation = base_model.predict(X_val)

np.save(open('bottleneck_features_validation.npy', 'wb'),
            bottleneck_features_validation)

#--------------Train Top model---------------------

train_data = np.load(open('bottleneck_features_train.npy','rb'))
train_labels = y_train_one_hot
validation_data = np.load(open('bottleneck_features_validation.npy','rb'))
validation_labels = y_val_one_hot


top_model = Sequential()
top_model.add(Flatten(input_shape=train_data.shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(43, activation='softmax'))


#model.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['accuracy'])

# compile the model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
top_model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=[metrics.categorical_accuracy])


top_model.fit(bottleneck_features_train, train_labels,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(validation_data,validation_labels),
              verbose=1,
              callbacks=[LearningRateScheduler(lr_schedule),tensorboard]
              )


top_model.save_weights(top_model_weights_path)

#---------------------------------
os.chdir(dataset_path+'/GTSRB')#En local
#os.chdir('/home/drobert/tfg/GTSRB')#En corleone

# Cargamos el archivo csv con los datos de test y vemos que contienen los 10 primeros
test = pd.read_csv('GT-final_test.csv', sep=';')

#Cargamos el dataset de test
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

test_accuracy = top_model.evaluate(X_test, y_test_one_target, verbose=1)

#Guardar best_model en un pickle


today_date = datetime.date.today().strftime("%d-%m-%Y")

model_filename= ("finetuningVGG16s%s_test_acc_%.2f%%_%s.h5" % (epochs,test_accuracy[1] * 100, today_date))

#pickle.dump(best_model, open((code_path + str(best_model_filename)), 'wb'))

#guardar con h5 no funciona por tener un metodo custom de accuracy
top_model.save(model_filename)

print("Accuracy en test : %s: %.2f%%" % (top_model.metrics_names[1], test_accuracy[1] * 100))

logging.info("Accuracy en test : %s: %.2f%%" % (top_model.metrics_names[1], test_accuracy[1] * 100))



# end time
end = time.time()
print ("[STATUS] end time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
logging.info("[STATUS] end time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))






