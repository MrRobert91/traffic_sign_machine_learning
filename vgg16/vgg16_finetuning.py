# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation

from sklearn.metrics import log_loss

import numpy as np
from skimage import color, exposure, transform

from skimage import io
import os
import glob

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pickle
from sklearn.model_selection import StratifiedKFold
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.models import Model
from keras import backend as K
K.set_image_data_format('channels_last')

from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import os, logging
from keras import metrics
from keras.models import load_model
import datetime
import json
from sklearn.model_selection import train_test_split
from keras import metrics
from PIL import Image

logging.info("program started on - " + str(datetime.datetime.now))

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
#top_model_path  = config["top_model_path"]

#Corleone
code_path="/home/drobert/tfg/traffic_sign_machine_learning/vgg16/"
dataset_path='/home/drobert/tfg/'
top_model_path  = config["top_model_path"]

#local
#code_path= "/home/david/PycharmProjects/traffic_sign_machine_learning/vgg16/"
#dataset_path="/home/david/Escritorio/TFG/Pruebas/"
#top_model_path =code_path+"output/vgg16_top_classifier.h5"

img_rows, img_cols = 48, 48 #80,80 #100,100#224, 224 # 48, 48 Resolution of inputs
channel = 3
num_classes = 43
batch_size = 16
epochs = 20
IMG_SIZE = 48
lr = 0.01

fichero_log = (code_path +'vgg16.log')

print('Archivo Log en ', fichero_log)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    filename = fichero_log,
                    filemode = 'a',)# w for new log each time


print ("[STATUS] --------vgg16 finetuning - systematized - start time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
logging.info(" ---------vgg16 finetuning - start time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))




def preprocess_img(img):
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    '''
    img = image.img_to_array(img)
    im_pil = Image.fromarray(img)
    im_pil = im_pil.resize((img_rows, img_cols))
    img = preprocess_input(im_pil)
    '''


    return img

def preprocess_img_old(img):
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

#os.chdir('/home/drobert/tfg/')#direccion en corleone
#root_dir = 'GTSRB/Final_Training/Images/'


imgs = []
labels = []

ruta_actual = os.getcwd()
print(ruta_actual)

all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))

print(os.path.join(root_dir, '*/*.ppm'))
print(len(all_img_paths))

np.random.shuffle(all_img_paths)

for img_path in all_img_paths:
    #img = preprocess_img(io.imread(img_path))
    img = image.load_img(img_path, target_size=(IMG_SIZE,IMG_SIZE))
    preprocess_img(img)#nuevo, probando...
    label = get_class(img_path)
    imgs.append(img)
    labels.append(label)

X = np.array(imgs, dtype='float32')
Y = np.asarray(labels)

print(X.shape)
print(Y.shape)

logging.info(X.shape)
logging.info(Y.shape)


# Vamos a hacer cross validation con nuestro conjunt de test.
# En concreto vamos a hacer un Kfold con 10 splits estratificado,
# de tal manera que cada conjunto tenga aproximadamente el mismo porcentaje
# de muestras de cada clase que el conjunto de entrenamiento.

training_history_list = []
val_accuracy_list = []

confusion_matrix_list = []
clf_list = []
filename_clf_list = []


def lr_schedule(epoch):
    return lr * (0.1 ** int(epoch / 10))


X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2)

# Make one hot targets
y_train_one_hot = np.eye(num_classes, dtype='uint8')[y_train]
y_val_one_hot = np.eye(num_classes, dtype='uint8')[y_val]



# Load our model
#-------------------------------------------

top_model_weights_path = top_model_path

# build the VGG16 network
#model = VGG16(weights='imagenet', include_top=False)


# create the base pre-trained model
base_model = VGG16(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 43 classes
predictions = Dense(43, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

print("Entrenendo top model...")
logging.info("Entrenendo top model...")

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=[metrics.categorical_accuracy])

# train the model on the new data for a few epochs
model.fit(X_train, y_train_one_hot,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        verbose=1,
        validation_data=(X_val, y_val_one_hot),
        )

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 15 layers and unfreeze the rest:
for layer in model.layers[:15]:
   layer.trainable = False
for layer in model.layers[15:]:
   layer.trainable = True

print("Entrenando model ensamblado (top+base) ...")
logging.info("Entrenando ensamblado (top+base) ...")

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.00001, momentum=0.9),
              loss='categorical_crossentropy',metrics=[metrics.categorical_accuracy])

# we train our model again (this time fine-tuning the top vgg16 conv block
# alongside the top Dense layers
model.fit(X_train, y_train_one_hot,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        verbose=1,
        validation_data=(X_val, y_val_one_hot),
        )

# Make predictions
predictions_valid = model.predict(X_val, batch_size=batch_size, verbose=1)
# Cross-entropy loss score
score = log_loss(y_val, predictions_valid)

val_accuracy = model.evaluate(X_val, y_val_one_hot, verbose=1)

print('val accuracy: '+ str(val_accuracy))
logging.info('val accuracy: '+ str(val_accuracy))

#print("%s: %.2f%%" % (model.metrics_names[1], val_accuracy[1] * 100))
#logging.info("%s: %.2f%%" % (model.metrics_names[1], val_accuracy[1] * 100))

#clf_list.append(model)  # lista de cada uno de los los clasificadores

#print('lista de accuracys de los modelos: '+str(val_accuracy_list))
#logging.info('lista de accuracys de los modelos: '+str(val_accuracy_list))

#precision_media = (np.mean(val_accuracy_list))
#desviacion_standar = (np.std(val_accuracy_list))

#print("mean_accuarcy: %.2f%% (+/- %.2f%%)" % (np.mean(val_accuracy_list), np.std(val_accuracy_list)))
#logging.info("mean_accuarcy: %.2f%% (+/- %.2f%%)" % (np.mean(val_accuracy_list), np.std(val_accuracy_list)))


# ---------TEST--------
print("Cargando imagenes de Test...")
logging.info("Cargando imagenes de Test...")


ruta_actual = os.getcwd()
#print(ruta_actual)
#print(os.listdir(ruta_actual))
os.chdir(dataset_path+'GTSRB')#En local
#os.chdir('/home/drobert/tfg/GTSRB')#En corleone

# Cargamos el archivo csv con los datos de test y vemos que contienen los 10 primeros
test = pd.read_csv('GT-final_test.csv', sep=';')
#test.head(10)

# In[61]:

# Cargamos el dataset de test
os.chdir(dataset_path+'GTSRB/Final_Test/Images/')#en local
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
y_test_one_target = np.eye(num_classes, dtype='uint8')[y_test]


#print("precision media: "+str(precision_media))
#logging.info("precision media: "+str(precision_media))

#model_indx = modelo_medio_indx(precision_media, val_accuracy_list)

#print("indice del modelo medio: "+str(model_indx))
#logging.info("indice del modelo medio: "+str(model_indx))

# cargamos el modelo medio de disco
os.chdir(code_path)
#best_model =clf_list[0]

test_accuracy = model.evaluate(X_test, y_test_one_target, verbose=1)

print('test accuracy: '+str(test_accuracy))
logging.info('test accuracy: '+str(test_accuracy))

#print("Accuracy en test : %s: %.2f%%" % (model.metrics_names[1], test_accuracy[1] * 100))
#logging.info("Accuracy en test : %s: %.2f%%" % (model.metrics_names[1], test_accuracy[1] * 100))


#Guardar best_model en un pickle


today_date = datetime.date.today().strftime("%d-%m-%Y")

best_model_filename= ("finetuningVGG16_epochs%s_test_acc_%.2f%%_%s.h5" % (epochs,test_accuracy[1] * 100, today_date))

#pickle.dump(best_model, open((code_path + str(best_model_filename)), 'wb'))

#guardar con h5 no funciona por tener un metodo custom de accuracy
model.save(best_model_filename)


#Comprobamos que el modelo cargado tiene la misma precision

#loaded_model = pickle.load(open(best_model_filename, 'rb'))
loaded_model = load_model(best_model_filename)# No funciona con custom metrics

loaded_model_test_accuracy = loaded_model.evaluate(X_test, y_test_one_target, verbose=1)


print('test accuracy: '+str(loaded_model_test_accuracy))
logging.info('test accuracy: '+str(loaded_model_test_accuracy))

#print("Loaded_model accuracy en test : %s: %.2f%%" % (loaded_model.metrics_names[1], loaded_model_test_accuracy[1] * 100))
#https://github.com/keras-team/keras/issues/3911
#La solucion propuesta arriba tampoco funciona

#loaded_model = load_model('best_model_filename', custom_objects={'get_categorical_accuracy_keras': get_categorical_accuracy_keras})
#loaded_model_test_accuracy = loaded_model.evaluate(X_test, y_test_one_target, verbose=1)

# Una técnica muy útil para visualizar el rendimiento de nuestro algoritmo es
# la matriz de confusión. y la mostramos de varia formas. Solo mostramos
# la matriz de confusion del modelo medio.

#Para generar la matriz de confusión necesitamos los targets en formato lista
#No en one hot encoding.


y_pred = loaded_model.predict(X_test)
#pasamos a one hot encoding para que tenga la misma estructura que y_pred
#No funciona así, tendran que ser los 2 vectores unidimensionales
#y_test_one_hot = to_categorical(y_test, NUM_CLASSES)

#pasamos y_pred que esta en one hot encoding a un vector plano
y_pred_no_one_hot= np.argmax(y_pred, axis=1, out=None)

print("shape de y_test , y_pred_no_one_hot :")

print(y_test.shape)
print(y_pred_no_one_hot.shape)

#cm = pd.DataFrame(confusion_matrix(y_test, y_pred_no_one_hot))
#logging.info("matriz de confusión del modelo medio: ")
#logging.info(cm)


print("Fin de la prueba vgg16 finetuning ")
logging.info("-----------Fin de la prueba vgg16 finetuning-----------")
logging.info("program ended on - " + str(datetime.datetime.now))