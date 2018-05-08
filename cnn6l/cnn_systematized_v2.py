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

#local
#code_path= "/home/david/PycharmProjects/traffic_sign_machine_learning/cnn6l/"
#dataset_path="/home/david/Escritorio/TFG/Pruebas"

#Corleone
code_path="/home/drobert/tfg/traffic_sign_machine_learning/cnn6l/"
dataset_path='/home/drobert/tfg/'

#fichero_log = ('/home/drobert/tfg/traffic_sign_machine_learning/cnn6l/cnn6l.log')
fichero_log = (code_path +'cnn_v2.log')


print('Archivo Log en ', fichero_log)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    filename = fichero_log,
                    filemode = 'a',)# w for new log each time


print ("[STATUS] --------cnn_v2  [0,1] norm systematized - start time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
logging.info(" ---------cnn_v2 [0,1] Norm normystematized - start time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))


#Modelo: red neuronal con 6 capas convolucionales
def cnn_model():
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
IMG_SIZE = 32 # Como se sugiere en el paper de LeCunn

batch_size = 32 #16
epochs = 20 #30 o 50
lr = 0.01

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

    #Imagen en escala de grises, en escala [0,1]
    #img = color.rgb2gray(img)

    # Scale features to be in [0, 1]
    #img = (img / 255.).astype(np.float32)

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


#Data Augmentation:

# this is the augmentation configuration we will use for training
datagen = ImageDataGenerator(
        #rescale=1./255,
        rotation_range=25,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.1,
        fill_mode='nearest')
        #preprocessing_function=preprocess_img(img))


# Vamos a hacer cross validation con nuestro conjunt de test.
# En concreto vamos a hacer un Kfold con 10 splits estratificado,
# de tal manera que cada conjunto tenga aproximadamente el mismo porcentaje
# de muestras de cada clase que el conjunto de entrenamiento.

training_history_list = []
val_accuracy_list = []

confusion_matrix_list = []
clf_list = []
filename_clf_list = []

fold = 1

skf = StratifiedKFold(n_splits=3)  # numero de 'trozos' en los que dividimos el dataset de entrenamiento
print(skf)
logging.info(skf)
#cnn_classifier = cnn_model()


def lr_schedule(epoch):
    return lr * (0.1 ** int(epoch / 10))


print('justo antes del for del skf ---')
for train_index, test_index in skf.split(X, Y):
    # conjuntos de train y test(validacion) para cada fold
    x_train, x_test = X[train_index], X[test_index]
    y_train_no_one_hot, y_test_no_one_hot = Y[train_index], Y[test_index]

    # Make one hot targets
    #y_train = np.eye(NUM_CLASSES, dtype='uint8')[y_train_no_one_hot]
    #y_test = np.eye(NUM_CLASSES, dtype='uint8')[y_test_no_one_hot]

    #one hot encodig con to_categorical
    y_train = to_categorical(y_train_no_one_hot, NUM_CLASSES)
    y_test = to_categorical(y_test_no_one_hot, NUM_CLASSES)

    cnn_classifier = cnn_model()

    # vamos a entrenar nuestro modelo con SGD + momentum
    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    cnn_classifier.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           # metrics=['accuracy'])
                           metrics=[metrics.categorical_accuracy])

    #Only required if featurewise_center or featurewise_std_normalization or zca_whitening.
    #datagen.fit(x_train)
    print("Hasta el fit generator en el fold "+ str(fold))
    print("len x_train: "+str(len(x_train)))

    # fits the model on batches with real-time data augmentation:
    hist = cnn_classifier.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=(len(x_train) / batch_size), epochs=epochs,
                        verbose=1, validation_data=(x_test, y_test),
                        callbacks=[LearningRateScheduler(lr_schedule)])


    print("tamaños de x_train e y_train")
    print(x_train.shape)
    print(y_train.shape)


    #ruta para local
    filepath = code_path+"cnn6l_v2-fold"+str(fold)+"-epochs"+str(epochs)+".h5"

    #Guardar training / validation loss/accuracy en cada epoch
    training_history_list.append(hist.history)


    val_accuracy = cnn_classifier.evaluate(x_test, y_test, verbose=1)

    print("%s: %.2f%%" % (cnn_classifier.metrics_names[1], val_accuracy[1] * 100))
    logging.info("%s: %.2f%%" % (cnn_classifier.metrics_names[1], val_accuracy[1] * 100))

    val_accuracy_list.append(val_accuracy[1] * 100)


    #y_pred = cnn_classifier.predict_classes(x_test)
    #test_accuracy = np.sum(y_pred == y_test) / np.size(y_pred)

    print("loss y val accuracy del fold "+str(fold)+" :"+str(val_accuracy))
    logging.info("loss y val accuracy del fold "+str(fold)+" :"+str(val_accuracy))

    clf_list.append(cnn_classifier)  # lista de cada uno de los los clasificadores

    fold = fold + 1

print('lista de accuracys de los modelos: '+str(val_accuracy_list))
logging.info('lista de accuracys de los modelos: '+str(val_accuracy_list))

precision_media = (np.mean(val_accuracy_list))
desviacion_standar = (np.std(val_accuracy_list))


print("mean_accuarcy: %.2f%% (+/- %.2f%%)" % (np.mean(val_accuracy_list), np.std(val_accuracy_list)))
logging.info("mean_accuarcy: %.2f%% (+/- %.2f%%)" % (np.mean(val_accuracy_list), np.std(val_accuracy_list)))


ruta_actual = os.getcwd()
#print(ruta_actual)
#print(os.listdir(ruta_actual))
os.chdir(dataset_path+'/GTSRB')


# Cargamos el archivo csv con los datos de test y vemos que contienen los 10 primeros
test = pd.read_csv('GT-final_test.csv', sep=';')
#test.head(10)

# In[61]:

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




# Función para encontrar el modelo que está mas proximo a la media
def modelo_medio_indx(final, numeros):
    def el_menor(numeros):
        menor = numeros[0]
        retorno = 0
        for x in range(len(numeros)):
            if numeros[x] < menor:
                menor = numeros[x]
                retorno = x
        return retorno

    diferencia = []
    for x in range(len(numeros)):
        diferencia.append(abs(final - numeros[x]))
    # devuelve el indice del modelo más próximo a la media
    return numeros.index(numeros[el_menor(diferencia)])



print("precision media: "+str(precision_media))
logging.info("precision media: "+str(precision_media))

model_indx = modelo_medio_indx(precision_media, val_accuracy_list)

print("indice del modelo medio: "+str(model_indx))
logging.info("indice del modelo medio: "+str(model_indx))

# cargamos el modelo medio de disco
os.chdir(code_path)
best_model =clf_list[model_indx]

test_accuracy = best_model.evaluate(X_test, y_test_one_target, verbose=1)

#NO RULA LO DE ABAJO
# this is the augmentation configuration we will use for testing:
# only rescaling
#test_datagen = ImageDataGenerator(rescale=1./255)
#test_accuracy = best_model.evaluate_generator(test_datagen.flow(x_train, y_train))

#Guardar best_model en un pickle


today_date = datetime.date.today().strftime("%d-%m-%Y")

best_model_filename= ("cnn6l_v2_epochs%s_test_acc_%.2f%%_%s.h5" % (epochs,test_accuracy[1] * 100, today_date))

#pickle.dump(best_model, open((code_path + str(best_model_filename)), 'wb'))

#guardar con h5 no funciona por tener un metodo custom de accuracy
best_model.save(best_model_filename)

print("Accuracy en test : %s: %.2f%%" % (best_model.metrics_names[1], test_accuracy[1] * 100))

logging.info("Accuracy en test : %s: %.2f%%" % (best_model.metrics_names[1], test_accuracy[1] * 100))


#Comprobamos que el modelo cargado tiene la misma precision

#loaded_model = pickle.load(open(best_model_filename, 'rb'))
loaded_model = load_model(best_model_filename)# No funciona con custom metrics

loaded_model_test_accuracy = loaded_model.evaluate(X_test, y_test_one_target, verbose=1)
print("Loaded_model accuracy en test : %s: %.2f%%" % (loaded_model.metrics_names[1], loaded_model_test_accuracy[1] * 100))
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



cm = pd.DataFrame(confusion_matrix(y_test, y_pred_no_one_hot))

#logging.info("matriz de confusión del modelo medio: ")
#logging.info(cm)


print("Fin de la prueba con CNN v2 con data augmentation")
logging.info("-----------Fin de la prueba con CNN v2 con data augmentation-----------")
logging.info("program ended on - " + str(datetime.datetime.now))
