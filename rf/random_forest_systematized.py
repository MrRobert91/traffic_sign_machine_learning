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
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.model_selection import StratifiedKFold
import datetime
import logging




#local
#code_path= "/home/david/PycharmProjects/traffic_sign_machine_learning/rf/"
#dataset_path="/home/david/Escritorio/TFG/Pruebas"

#Corleone
code_path="/home/drobert/tfg/traffic_sign_machine_learning/rf/"
dataset_path='/home/drobert/tfg/'

# Fichero de log
fichero_log = (code_path +'rf.log')

print('Archivo Log en ', fichero_log)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    filename = fichero_log,
                    filemode = 'a',)

logging.info("program started on - " + str(datetime.datetime.now))




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


# A ver si no peta sin el bucle que cuenta por donde va

# In[2]:


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
# Make one hot targets
# Y = np.eye(NUM_CLASSES, dtype='uint8')[labels]

# Para el random forest no necesitamos one hot encoding
Y = np.asarray(labels)

print(X.shape)
print(Y.shape)

# In[4]:


# Tenemos que cambiar los formatos de entrada
X = X.reshape((-1, 48 * 48 * 3)).astype(np.float32)
print(X.shape)

# Vamos a hacer cross validation con nuestro conjunt de test. En concreto vamos a hacer un Kfold con 10 splits estratificado, de tal manera que cada conjunto tenga aproximadamente el mismo porcentaje de muestras de cada clase que el conjunto de entrenamiento.

# In[ ]:





#test_scores_list = []
test_accuracy_list = []
confusion_matrix_list = []
clf_list = []
filename_clf_list = []

n_trees = 300
depth = 3
fold = 1
splits = 3

skf = StratifiedKFold(n_splits=splits)  # numero de 'trozos' en los que dividimos el dataset de entrenamiento
print(skf)

for train_index, test_index in skf.split(X, Y):
    # conjuntos de train y test(validacion) para cada fold
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    # El clasificador es un random forest, probamos con 35 arboles para que no tarde mucho
    rf_classifier = RandomForestClassifier(n_estimators=n_trees, max_depth=depth, oob_score=True)
    rf_classifier.fit(x_train, y_train)

    #test_scores_list.append(rf_classifier.score(x_test, y_test))  # lista de los scores  obtenidas por los random forest

    pred_y = rf_classifier.predict(x_test)
    test_accuracy = accuracy_score(y_test, pred_y)  # igual que scores

    test_accuracy_list.append(test_accuracy)  # lista de las precisiones obtenidas por los random forest

    #Son relevantes estas cm? no es mejor guardar solo la de test?
    #cm = pd.DataFrame(confusion_matrix(y_test, pred_y))
    #confusion_matrix_list.append(cm)

    clf_list.append(rf_classifier)  # lista de cada uno de los los clasificadores

    # Persistimos los modelos con pickle
    # save the model to disk

    #filename = 'rf_' + str(n_trees) + 'trees_' + str(fold) + 'fold_' + str(test_accuracy)+'val_acc'
    filename = 'rf_' + str(n_trees) + 'trees_' + str(depth) + 'depth_' + str(fold) + 'fold_' + "{0:.3f}".format(test_accuracy) + 'val_acc'
    filename_clf_list.append(filename)

    #pickle.dump(rf_classifier, open((code_path + str(filename)), 'wb'))

    fold = fold + 1

#  test_scores_list con 35 arboles 94% aprox:
#  [0.93603672532517213, 0.9406273909716909, 0.94008722932129463]


precision_media = (np.mean(test_accuracy_list))
desviacion_standar = (np.std(test_accuracy_list))


print("mean_accuarcy: " + str(precision_media) + " std: " + str(desviacion_standar))
logging.info(("mean_accuarcy: " + str(precision_media) + " std: " + str(desviacion_standar)))

# El accuracy medio con 500 arboles y 10 fold es de 0.975184290242 con un desviacion standar de 0.00221154307211
#
# El accuracy medio con 35 arboles y 3 fold es de 0.938917115206 con un desviacion standar de 0.00204864639076


# Función para encontrar el modelo que está más próximo a la media
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


print("precision media en training : "+str(precision_media))

logging.info("precision media en training: "+str(precision_media))

model_indx = modelo_medio_indx(precision_media, test_accuracy_list)

print(model_indx)
logging.info("indice del modelo medio: "+str(model_indx))

os.chdir(code_path)

# cargamos el modelo medio de disco
#modelname = filename_clf_list[model_indx]
#loaded_model = pickle.load(open(modelname, 'rb'))
#result = loaded_model.score(X_test, y_test)

#--------------------------
#Otra forma sin guardar todods los modelos de entrenamiento
#filename_bestmodel = 'rf_' + str(n_trees) + 'trees_' + str(splits) + 'fold_' + "{0:.3f}".format(test_accuracy) + 'val_acc'
modelname = filename_clf_list[model_indx]
bestmodel = clf_list[model_indx]

##Cargamos los datos de test

ruta_actual = os.getcwd()
print(ruta_actual)
print(os.listdir(ruta_actual))
#os.chdir('/home/david/Escritorio/TFG/Pruebas/GTSRB')
os.chdir(dataset_path+'GTSRB')#En corleone

# Cargamos el archivo csv con los datos de test y vemos que contienen los 10 primeros
test = pd.read_csv('GT-final_test.csv', sep=';')
#test.head(10)

#Guardamos el modelo medio
pickle.dump(bestmodel, open((code_path + str(modelname)), 'wb'))


#Probamos a liberar memoria
import gc
gc.collect()


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
#------------------------------------------------




#Evaluamos el modelo en test
#test_accuracy = bestmodel.score(X_test, y_test)
test_accuracy = bestmodel.accuracy_score(X_test, y_test)


print("Accuracy en test : %s: %.2f%%" % (bestmodel.metrics_names[1], test_accuracy[1] * 100))
logging.info("Accuracy en test : %s: %.2f%%" % (bestmodel.metrics_names[1], test_accuracy[1] * 100))


print("Resultado final del modelo en test: %.2f%% " % (test_accuracy))
logging.info("Resultado final del modelo en test: %.2f%% " % (test_accuracy))

#Cargamos el modelo
loaded_model = pickle.load(open(code_path + str(modelname), 'rb'))

result_loaded_model = loaded_model.score(X_test, y_test)

print("Resultado final del modelo en test cargado: %.2f%% " % (result_loaded_model))
logging.info("Resultado final del modelo en test cargado: %.2f%% " % (result_loaded_model))

# Una técnica muy útil para visualizar el rendimiento de nuestro algoritmo es
# la matriz de confusión.
# Solo mostramos la matriz de confusion del modelo medio.
X_test_shape = X_test.shape
y_test_shape = y_test.shape

print('X_test_shape: '+str(X_test_shape))
print('y_test_shape: '+str(y_test_shape))

#y_pred = bestmodel.predict(X_test)
#cm = pd.DataFrame(confusion_matrix(y_test, pred_y))

#cm = confusion_matrix_list[model_indx]

#print(cm)

print("-----------Fin de la prueba con RF -----------")
logging.info("-----------Fin de la prueba con RF -----------")
logging.info("program ended on - " + str(datetime.datetime.now))