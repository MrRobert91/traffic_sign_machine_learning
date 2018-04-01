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
# Vamos a utilizar un random forest para clasificar imagenes de señales de tráfico del dataset [GTSRB](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)

# In[1]:


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

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K
K.set_image_data_format('channels_last')

from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import os, logging

#fichero_log = ('/home/drobert/tfg/traffic_sign_machine_learning/cnn6l/cnn6l.log')
fichero_log = ('/home/david/PycharmProjects/traffic_sign_machine_learning/cnn6l/cnn6l.log')


print('Archivo Log en ', fichero_log)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    filename = fichero_log,
                    filemode = 'a',)# w for new log each time


logging.info('Clasificación de señales de tráfico con cnn de 6 capas')


#Modelo: red neuronal con 6 capas convolucionales
def cnn_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(IMG_SIZE, IMG_SIZE, 3),
                     activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model


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

def get_class(img_path):
    return int(img_path.split('/')[-2])


os.chdir('/home/david/Escritorio/TFG/Pruebas') #direccion local Jupyter Notebooks/pycharm
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
# In[4]:

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

#
def lr_schedule(epoch):
    return lr * (0.1 ** int(epoch / 10))

#Me daba un error.
#https://stackoverflow.com/questions/46305252/valueerror-dimension-1-must-be-in-the-range-0-2-in-keras
def get_categorical_accuracy_keras(y_true, y_pred):
    return K.mean(K.equal(K.argmax(y_true, axis=1), K.argmax(y_pred, axis=1)))

batch_size = 32
epochs = 2 #ponemos 5 para que sea mas rapido, normalmente 30
lr = 0.01

for train_index, test_index in skf.split(X, Y):
    # conjuntos de train y test(validacion) para cada fold
    x_train, x_test = X[train_index], X[test_index]
    y_train_no_one_hot, y_test_no_one_hot = Y[train_index], Y[test_index]

    # Make one hot targets
    y_train = np.eye(NUM_CLASSES, dtype='uint8')[y_train_no_one_hot]
    y_test = np.eye(NUM_CLASSES, dtype='uint8')[y_test_no_one_hot]

    #one hot encodin con to_categorical
    #dummy_y = np_utils.to_categorical(y_train_no_one_hot, NUM_CLASSES)
    #dummy_y = np_utils.to_categorical(y_test_no_one_hot, NUM_CLASSES)



    cnn_classifier = cnn_model()

    # vamos a entrenar nuestro modelo con SGD + momentum
    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    cnn_classifier.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=[get_categorical_accuracy_keras])

    print("tamaños de x_train e y_train")
    print(x_train.shape)
    print(y_train.shape)

    #ruta para local
    filepath = "/home/david/PycharmProjects/traffic_sign_machine_learning/cnn6l/cnn6l-fold"+str(fold)+"-epochs"+str(epochs)+".h5"
    #ruta para corleone
    #filepath = "/home/drobert/tfg/traffic_sign_machine_learning/cnn6l/cnn6l-fold"+str(fold)+"-epochs"+str(epochs)+".h5"
    hist = cnn_classifier.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              verbose=1,
              callbacks=[LearningRateScheduler(lr_schedule),
                         ModelCheckpoint(filepath, save_best_only=True)]
              )


    #Guardar training / validation loss/accuracy en cada epoch
    training_history_list.append(hist.history)
    print("history:")
    print(hist.history)
    logging.info("history:")
    logging.info(hist.history)


    val_accuracy = cnn_classifier.evaluate(x_test, y_test, verbose=1)

    print("%s: %.2f%%" % (cnn_classifier.metrics_names[1], val_accuracy[1] * 100))
    logging.info("%s: %.2f%%" % (cnn_classifier.metrics_names[1], val_accuracy[1] * 100))

    val_accuracy_list.append(val_accuracy[1] * 100)


    #y_pred = cnn_classifier.predict_classes(x_test)
    #test_accuracy = np.sum(y_pred == y_test) / np.size(y_pred)


    print("loss y val accuracy del fold "+str(fold)+" :"+str(val_accuracy))
    logging.info("loss y val accuracy del fold "+str(fold)+" :"+str(val_accuracy))

    #Para generar la matriz de confusión necesitamos los targets en formato lista
    #No en one hot encoding.
    #cm = pd.DataFrame(confusion_matrix(y_test_no_one_hot, y_pred))
    #confusion_matrix_list.append(cm)

    clf_list.append(cnn_classifier)  # lista de cada uno de los los clasificadores

    #NO hacemos un pickle porque ya lo guardaos en formato h5

    #filename = 'cnn6l_' + str(fold) + 'fold_'+"{0:.3f}".format(test_accuracy)+'val_acc'
    #filename_clf_list.append(filename)

    #pickle.dump(cnn_classifier, open(('/home/drobert/tfg/cnn6l' + str(filename)), 'wb'))

    fold = fold + 1

print('lista de accuracys de los modelos: '+str(val_accuracy_list))
logging.info('lista de accuracys de los modelos: '+str(val_accuracy_list))

precision_media = (np.mean(val_accuracy_list))
desviacion_standar = (np.std(val_accuracy_list))


print("mean_accuarcy: %.2f%% (+/- %.2f%%)" % (np.mean(val_accuracy_list), np.std(val_accuracy_list)))
logging.info("mean_accuarcy: %.2f%% (+/- %.2f%%)" % (np.mean(val_accuracy_list), np.std(val_accuracy_list)))

#print("mean_accuarcy: " + str(precision_media) + " std: " + str(desviacion_standar))
#logging.info("mean_accuarcy: " + str(precision_media) + " std: " + str(desviacion_standar))

ruta_actual = os.getcwd()
print(ruta_actual)
print(os.listdir(ruta_actual))
os.chdir('/home/david/Escritorio/TFG/Pruebas/GTSRB')#En local
#os.chdir('/home/drobert/tfg/GTSRB')#En corleone

# Cargamos el archivo csv con los datos de test y vemos que contienen los 10 primeros
test = pd.read_csv('GT-final_test.csv', sep=';')
#test.head(10)

# In[61]:

# Cargamos el dataset de test
os.chdir('/home/david/Escritorio/TFG/Pruebas/GTSRB/Final_Test/Images/')#en local
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


print(X_test.shape)
print(y_test.shape)


#Los targets tienen que estar en formato one target
y_test_one_target = np.eye(NUM_CLASSES, dtype='uint8')[y_test]

# Cambiamos los formatos de entrada de las imagenes para que sea una matriz bidimensional
#X_test = X_test.reshape((-1, 48 * 48 * 3)).astype(np.float32)

#print(X_test.shape)
#print(y_test.shape)

# Es una medida que es la media de las clasificaciones correctas.
# https://stackoverflow.com/questions/31438476/parameter-oob-score-in-scikit-learn-equals-accuracy-or-error

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
    return numeros.index(numeros[el_menor(diferencia)])  # devuelve el indice del más próximo
    # return numeros[index(el_menor(diferencia))]


print("precision media: "+str(precision_media))
logging.info("precision media: "+str(precision_media))

model_indx = modelo_medio_indx(precision_media, val_accuracy_list)

print("indice del modelo medio: "+str(model_indx))
logging.info("indice del modelo medio: "+str(model_indx))
# cargamos el modelo medio de disco


os.chdir('/home/david/PycharmProjects/traffic_sign_machine_learning/cnn6l')
#os.chdir('/home/drobert/tfg/traffic_sign_machine_learning/cnn6l')
#modelname = filename_clf_list[model_indx]
best_model =clf_list[model_indx]

#TO-DO guardar best_model en un pickle

test_accuracy = best_model.evaluate(X_test, y_test_one_target, verbose=1)

print("Accuracy en test : %s: %.2f%%" % (best_model.metrics_names[1], test_accuracy[1] * 100))
logging.info("Accuracy en test : %s: %.2f%%" % (best_model.metrics_names[1], test_accuracy[1] * 100))

#loaded_model = pickle.load(open(modelname, 'rb'))
#result = loaded_model.score(X_test, y_test)
#print("Resultado final del modelo medio: "+str(result))
#logging.info("Resultado final del modelo medio: "+str(result))

# Una técnica muy útil para visualizar el rendimiento de nuestro algoritmo es la matriz de confusión. y la mostramos de varia formas. Solo mostramos la matriz de confusion del modelo medio.

# In[82]:
#cm = confusion_matrix_list[2]

print("Fin de la prueba con CNN6l")
#logging.info("matriz de confusión: ")
#logging.info(cm)
logging.info("Fin de la prueba con CNN de 6 capas convolucionales")