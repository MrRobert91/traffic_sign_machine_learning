# coding: utf-8

# Version sistematizada:
# 1- preprocesado
# 2- divsión del dataset en entrenamiento y validación
# 3- validacion cruzada estratificada(10 fold)
# 4- media y desviación de las matrices para cada fold
# 5- quedarse con el módelo más próximo al modelo promedio.
# 6- guardar los resultados y los hiperparametros. (diccionario, csv, ...)



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
K.set_image_data_format('channels_last')

# cargar las variables de configuracion
with open('conf.json') as f:
	config = json.load(f)

# variables de configuracion
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


fichero_log = (code_path +'cnn_v2.log')


print('Archivo Log en ', fichero_log)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    filename = fichero_log,
                    filemode = 'a',)# w para un nuevo log cada vez


print ("[STATUS] --------cnn_v2 multi scale  systematized - start time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
logging.info(" ---------cnn_v2 multi scale systematized - start time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))


#Modelo: red neuronal con 6 capas convolucionales
#4º modelo en tfg con API Funcional
def mini_vgg():
    input_tensor = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='4d_input')

    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(input_tensor)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output_tensor = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(input_tensor, output_tensor)

    return model


NUM_CLASSES = 43
IMG_SIZE = 48 # Como se sugiere en el paper de LeCun

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

# Configuaracion de las imagenes generadas con data augmentation
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
# En concreto vamos a hacer un Kfold con 3 splits estratificado,
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



def lr_schedule(epoch):
    return lr * (0.1 ** int(epoch / 10))


print('justo antes del for del skf ---')
for train_index, test_index in skf.split(X, Y):
    # conjuntos de train y test(validacion) para cada fold
    x_train, x_test = X[train_index], X[test_index]
    y_train_no_one_hot, y_test_no_one_hot = Y[train_index], Y[test_index]

    #one hot encodig con to_categorical
    y_train = to_categorical(y_train_no_one_hot, NUM_CLASSES)
    y_test = to_categorical(y_test_no_one_hot, NUM_CLASSES)

    cnn_classifier = mini_vgg()

    # vamos a entrenar nuestro modelo con SGD + momentum
    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    cnn_classifier.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           # metrics=['accuracy'])
                           metrics=[metrics.categorical_accuracy])

    #Solo hace falta hacer datagen.fit si se utiliza  featurewise_center
    # o featurewise_std_normalization o zca_whitening.
    datagen.fit(x_train)
    print("Hasta el fit generator en el fold "+ str(fold))
    print("len x_train: "+str(len(x_train)))

    # entrena el modelo en batches con data augmentation en tiempo real:
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
os.chdir(dataset_path+'/GTSRB')


# Cargamos el archivo csv con los datos de test y vemos que contienen los 10 primeros
test = pd.read_csv('GT-final_test.csv', sep=';')


# Cargamos el dataset de test
os.chdir(dataset_path+'/GTSRB/Final_Test/Images/')

X_test = []
y_test = []
i = 0

for file_name, class_id in zip(list(test['Filename']), list(test['ClassId'])):
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

#Guardar best_model

today_date = datetime.date.today().strftime("%d-%m-%Y")

best_model_filename= ("cnn6l_v2_Multi_scale_2_epochs%s_test_acc_%.2f%%_%s.h5" % (epochs,test_accuracy[1] * 100, today_date))

#pickle.dump(best_model, open((code_path + str(best_model_filename)), 'wb'))

#guardar con h5 no funciona en local por tener un metodo custom de accuracy
best_model.save(best_model_filename)

print("Accuracy en test : %s: %.2f%%" % (best_model.metrics_names[1], test_accuracy[1] * 100))

logging.info("Accuracy en test : %s: %.2f%%" % (best_model.metrics_names[1], test_accuracy[1] * 100))


#Comprobamos que el modelo cargado tiene la misma precision

#loaded_model = pickle.load(open(best_model_filename, 'rb'))
loaded_model = load_model(best_model_filename)# No funciona con custom metrics en local

loaded_model_test_accuracy = loaded_model.evaluate(X_test, y_test_one_target, verbose=1)
print("Loaded_model accuracy en test : %s: %.2f%%" % (loaded_model.metrics_names[1], loaded_model_test_accuracy[1] * 100))
#https://github.com/keras-team/keras/issues/3911
#La solucion propuesta arriba tampoco funciona

#loaded_model = load_model('best_model_filename', custom_objects={'get_categorical_accuracy_keras': get_categorical_accuracy_keras})
#loaded_model_test_accuracy = loaded_model.evaluate(X_test, y_test_one_target, verbose=1)




y_pred = loaded_model.predict(X_test)

#pasamos y_pred que esta en one hot encoding a un vector plano
y_pred_no_one_hot= np.argmax(y_pred, axis=1, out=None)

print("shape de y_test , y_pred_no_one_hot :")

print(y_test.shape)
print(y_pred_no_one_hot.shape)




print("Fin de la prueba con CNN con data augmentation")
logging.info("-----------Fin de la prueba con CNN con data augmentation-----------")
logging.info("program ended on - " + str(datetime.datetime.now))
