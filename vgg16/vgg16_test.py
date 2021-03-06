# test script to preform prediction on test images inside
# GTSRB/Final_Test/images/
#   -- 00000.ppm
#   -- 00001.ppm
#   ...

# organize imports
from __future__ import print_function

# keras imports
from keras.applications.vgg16 import VGG16, preprocess_input
#from keras.applications.vgg19 import VGG19, preprocess_input
#from keras.applications.xception import Xception, preprocess_input
#from keras.applications.resnet50 import ResNet50, preprocess_input
#from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
#from keras.applications.mobilenet import MobileNet, preprocess_input
#from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Input

# other imports
from sklearn.linear_model import LogisticRegression
import numpy as np
import os
import json
import pickle
#import cv2

import datetime
import time
import logging
import pandas as pd
from skimage import io
from keras.models import load_model
from sklearn.model_selection import KFold

# load the user configs
with open('conf.json') as f:
    config = json.load(f)

# config variables
model_name = config["model"]
weights = config["weights"]
include_top = config["include_top"]
train_path = config["train_path"]
test_path = config["test_path"]
features_path = config["features_path"]
labels_path = config["labels_path"]
test_size = config["test_size"]
results = config["results"]
model_path = config["model_path"]
seed = config["seed"]
classifier_path = config["classifier_path"]
log_path		= config["log_path"]


fichero_log = (log_path)

print('Archivo Log en ', fichero_log)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    filename = fichero_log,
                    filemode = 'a',)
# start time
print ("[STATUS] ---------vgg16 test - start time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
logging.info("------------ vgg16 test - start time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
start = time.time()

# load the trained logistic regression classifier
print("[INFO] loading the classifier...")
print(classifier_path)

classifier = pickle.load(open(classifier_path, 'rb'))

# pretrained models needed to perform feature extraction on test data too!
base_model = VGG16(weights=weights)
model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)
image_size = (224, 224)

modelo="vgg16_fe"
NUM_CLASSES = 43

# get all the train labels
#train_labels = os.listdir(train_path)


#print('nombres de las etiquetas(labels): ')
#for label in train_labels:
#    print(label)

# get all the test images paths
#test_images = os.listdir(test_path)


os.chdir('/home/drobert/tfg/GTSRB')

# Cargamos el archivo csv con los datos de test
test = pd.read_csv('GT-final_test.csv', sep=';')
#print("test.head(10) :")
#test.head(10)


# Cargamos el dataset de test
#os.chdir('/home/david/Escritorio/TFG/Pruebas/GTSRB/Final_Test/Images/')
os.chdir(test_path)#en corleone

print("looping through each image in the test data...")
logging.info("looping through each image in the test data...")

# variables to hold features and labels
features = []
labels = []

i = 0
for file_name, class_id in zip(list(test['Filename']), list(test['ClassId'])):
    # img_path = os.path.join('GTSRB/Final_Test/Images/', file_name)
    img_path = os.path.join(os.getcwd(), file_name)
    #--recorremos todas las imagenes de test y extraemos la features--
    img = image.load_img(img_path, target_size=image_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feature = model.predict(x)
    flat = feature.flatten()
    #flat = np.expand_dims(flat, axis=0)#Para predecir las imagenes una a una
    features.append(flat)
    #----
    labels.append(class_id)

X_test = np.array(features)
y_test = np.array(labels)

print("[INFO] test data   : {}".format(X_test.shape))
print("[INFO] test labels : {}".format(y_test.shape))
logging.info("[INFO] test data   : {}".format(X_test.shape))
logging.info("[INFO] test labels : {}".format(y_test.shape))
#print("Tiene que ser algo como: (31367, 2048) y las labels : (7842,) ")


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
    #test_accuracy = loaded_model.evaluate(X_test_fold, y_test_fold_one_target, verbose=1)

    #Para Feature extraction
    test_accuracy = classifier.score(X_test, y_test)
    accuracy_list.append(test_accuracy * 100)

    print(str(fold)+" Resultado final del modelo en test: %.2f%% " % (test_accuracy * 100))
    logging.info(str(fold)+" Resultado final del modelo en test: %.2f%% " % (test_accuracy * 100))

    fold += 1



precision_media = (np.mean(accuracy_list))
desviacion_standar = (np.std(accuracy_list))

print("mean_accuarcy de" +modelo+": %.2f%% (+/- %.2f%%)" % (np.mean(accuracy_list), np.std(accuracy_list)))
logging.info("mean_accuarcy de " +modelo+": %.2f%% (+/- %.2f%%)" % (np.mean(accuracy_list), np.std(accuracy_list)))



'''

#result on the logistic regression classifier

result = classifier.score(X_test, y_test)

print("test_accuracy final del modelo en test: %.2f%% " % (result*100))
logging.info("test_accuracy final del modelo en test: %.2f%% " % (result *100))


today_date = datetime.date.today().strftime("%d-%m-%Y")
best_model_filename= (model+"_test_acc_%.2f%%_%s.h5" % (result * 100, today_date))

#pickle.dump(best_model, open((code_path + str(best_model_filename)), 'wb'))

#guardar con h5 no funciona por tener un metodo custom de accuracy
classifier.save(best_model_filename)

print("Accuracy en test : %.2f%%" % (result * 100))

logging.info("Accuracy en test : %.2f%%" % ( result * 100))


#Comprobamos que el modelo cargado tiene la misma precision

#loaded_model = pickle.load(open(best_model_filename, 'rb'))
loaded_model = load_model(best_model_filename)# No funciona con custom metrics

loaded_model_test_accuracy = loaded_model.score(X_test, y_test)
print("Loaded_model accuracy en test : %s: %.2f%%" % (loaded_model.metrics_names[1], loaded_model_test_accuracy[1] * 100))
'''

end = time.time()
print ("[STATUS] end time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
logging.info("[STATUS] end time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))

