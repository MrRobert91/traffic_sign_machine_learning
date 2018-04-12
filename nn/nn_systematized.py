
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
from keras.utils.np_utils import to_categorical


#local
#code_path= "/home/david/PycharmProjects/traffic_sign_machine_learning/nn/"
#dataset_path="/home/david/Escritorio/TFG/Pruebas"

#Corleone
code_path="/home/drobert/tfg/traffic_sign_machine_learning/nn/"
dataset_path='/home/drobert/tfg/'

#fichero_log = ('/home/drobert/tfg/traffic_sign_machine_learning/nn/nn.log')
fichero_log = (code_path +'nn.log')

NUM_CLASSES = 43
IMG_SIZE = 48

print('Archivo Log en ', fichero_log)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    filename = fichero_log,
                    filemode = 'a',)# w for new log each time


logging.info("-----------Inicio de la prueba con NN -----------")
logging.info("program started on - " + str(datetime.datetime.now))
logging.info('Clasificación de señales de tráfico con nn')

dim_data = (IMG_SIZE*IMG_SIZE*3)

def baseline_nn_model():
    # create model
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(dim_data,)))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model

#Modelo utilizando reguralizacion (dropout)
def reg_nn_model():
    model_reg = Sequential()
    model_reg.add(Dense(1024, activation='relu', input_shape=(dim_data,)))
    model_reg.add(Dropout(0.5))
    model_reg.add(Dense(1024, activation='relu'))
    model_reg.add(Dropout(0.5))
    model_reg.add(Dense(512, activation='relu'))
    model_reg.add(Dropout(0.5))
    model_reg.add(Dense(NUM_CLASSES, activation='softmax'))
    return model_reg

#Modelo utilizando reguralizacion (dropout)
def reg_nn_model_2():
    model_reg = Sequential()
    model_reg.add(Dense(1024, activation='relu', input_shape=(dim_data,)))
    model_reg.add(Dropout(0.5))
    model_reg.add(Dense(512, activation='relu'))
    model_reg.add(Dropout(0.5))
    model_reg.add(Dense(NUM_CLASSES, activation='softmax'))
    return model_reg

#Modelo utilizando reguralizacion (dropout)
def reg_nn_model_3():
    model_reg = Sequential()
    model_reg.add(Dense(512, activation='relu', input_shape=(dim_data,)))
    model_reg.add(Dropout(0.5))
    model_reg.add(Dense(512, activation='relu'))
    model_reg.add(Dropout(0.5))
    model_reg.add(Dense(NUM_CLASSES, activation='softmax'))
    return model_reg



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

# Tenemos que cambiar los formatos de entrada para la capa densamente conexa.
# No hay convoluciones aquí
X = X.reshape((-1, 48 * 48 * 3)).astype(np.float32)

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


def lr_schedule(epoch):
    return lr * (0.1 ** int(epoch / 10))

#Me daba un error.
#https://stackoverflow.com/questions/46305252/valueerror-dimension-1-must-be-in-the-range-0-2-in-keras
def get_categorical_accuracy_keras(y_true, y_pred):
    return K.mean(K.equal(K.argmax(y_true, axis=1), K.argmax(y_pred, axis=1)))

batch_size = 32
epochs = 10
lr = 0.01

for train_index, test_index in skf.split(X, Y):
    # conjuntos de train y test(validacion) para cada fold
    x_train, x_test = X[train_index], X[test_index]
    y_train_no_one_hot, y_test_no_one_hot = Y[train_index], Y[test_index]

    # Make one hot targets
    y_train = np.eye(NUM_CLASSES, dtype='uint8')[y_train_no_one_hot]
    y_test = np.eye(NUM_CLASSES, dtype='uint8')[y_test_no_one_hot]

    #one hot encodig con to_categorical
    #dummy_y = np_utils.to_categorical(y_train_no_one_hot, NUM_CLASSES)
    #dummy_y = np_utils.to_categorical(y_test_no_one_hot, NUM_CLASSES)



    nn_classifier = baseline_nn_model()

    # vamos a entrenar nuestro modelo con SGD + momentum
    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    nn_classifier.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  #metrics=['accuracy'])
                  metrics=[metrics.categorical_accuracy])
                  #metrics=[get_categorical_accuracy_keras])#unico que funciona

    print("tamaños de x_train e y_train")
    print(x_train.shape)
    print(y_train.shape)

    #ruta para local
    #filepath = code_path+"nn-fold"+str(fold)+"-epochs"+str(epochs)+".h5"
    #ruta para corleone
    #filepath = "/home/drobert/tfg/traffic_sign_machine_learning/nn/nn-fold"+str(fold)+"-epochs"+str(epochs)+".h5"


    hist = nn_classifier.fit(x_train, y_train,
                             batch_size=batch_size,
                             epochs=epochs,
                             validation_split=0.2,
                             verbose=1,
                             callbacks=[LearningRateScheduler(lr_schedule)]
                             )


    #Guardar training / validation loss/accuracy en cada epoch
    training_history_list.append(hist.history)
    #print("history:")
    #print(hist.history)
    #logging.info("history:")
    #logging.info(hist.history)


    val_accuracy = nn_classifier.evaluate(x_test, y_test, verbose=1)

    print("%s: %.2f%%" % (nn_classifier.metrics_names[1], val_accuracy[1] * 100))
    logging.info("%s: %.2f%%" % (nn_classifier.metrics_names[1], val_accuracy[1] * 100))

    val_accuracy_list.append(val_accuracy[1] * 100)


    #y_pred = nn_classifier.predict_classes(x_test)
    #test_accuracy = np.sum(y_pred == y_test) / np.size(y_pred)


    print("loss y val accuracy del fold "+str(fold)+" :"+str(val_accuracy))
    logging.info("loss y val accuracy del fold "+str(fold)+" :"+str(val_accuracy))



    clf_list.append(nn_classifier)  # lista de cada uno de los los clasificadores

    #NO hacemos un pickle porque ya lo guardaos en formato h5

    fold = fold + 1

print('lista de accuracys de los modelos: '+str(val_accuracy_list))
logging.info('lista de accuracys de los modelos: '+str(val_accuracy_list))

precision_media = (np.mean(val_accuracy_list))
desviacion_standar = (np.std(val_accuracy_list))


print("mean_accuarcy: %.2f%% (+/- %.2f%%)" % (np.mean(val_accuracy_list), np.std(val_accuracy_list)))
logging.info("mean_accuarcy: %.2f%% (+/- %.2f%%)" % (np.mean(val_accuracy_list), np.std(val_accuracy_list)))

# Cargamos el archivo csv con los datos de test
os.chdir(dataset_path+'/GTSRB')#En local
test = pd.read_csv('GT-final_test.csv', sep=';')

# Cargamos el dataset de test
os.chdir(dataset_path+'/GTSRB/Final_Test/Images/')


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

#Guardar best_model en un pickle


today_date = datetime.date.today().strftime("%d-%m-%Y")

best_model_filename= ("nn_epochs%s_test_acc_%.2f%%_%s.h5" % (epochs,test_accuracy[1] * 100, today_date))

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


print("---------- Fin de la prueba con NN ----------")
logging.info("-----------Fin de la prueba con NN -----------")
logging.info("program ended on - " + str(datetime.datetime.now))