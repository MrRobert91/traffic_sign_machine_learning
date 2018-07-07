# filter warnings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# keras imports

from keras.applications.xception import Xception, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input

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

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pickle
import logging

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

fichero_log = (log_path)

print('Archivo Log en ', fichero_log)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    filename = fichero_log,
                    filemode = 'a',)
# start time
print ("[STATUS] --------xception systematized - start time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
logging.info(" -------xception systematized - start time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
start = time.time()

# create the pretrained models
# check for pretrained weight usage or not
# check for top layers to be included or not

base_model = Xception(weights=weights)
model = Model(input=base_model.input, output=base_model.get_layer('avg_pool').output)
image_size = (299, 299)

print ("[INFO] successfully loaded base model and model...")
logging.info((" successfully loaded base model and model..."))

# path to training dataset
train_labels = os.listdir(train_path)

# codificar las labels
print ("[INFO] encoding labels...")
logging.info("encoding labels...")
le = LabelEncoder()
le.fit([tl for tl in train_labels])

# variables para guardar las features y labels
features = []
labels = []


# Vamos a medir cuanto tarda en recorrer todas las imagenes
init_loop = datetime.datetime.now().replace(microsecond=0)

# recorremos toda la carpeta
count = 1
for i, label in enumerate(train_labels):
	cur_path = train_path + "/" + label
	count = 1
	for image_path in glob.glob(cur_path + "/*.ppm"):
	    img = image.load_img(image_path, target_size=image_size)
	    x = image.img_to_array(img)
	    x = np.expand_dims(x, axis=0)
	    x = preprocess_input(x) # escala los pixeles entre -1 and 1,sample-wise.
	    feature = model.predict(x)
	    flat = feature.flatten()
	    features.append(flat)
	    labels.append(label)

	    count += 1
	print("[INFO] completed label - " + label)
	logging.info((" completed label - " + label))

end_loop = datetime.datetime.now().replace(microsecond=0)
print("Tarda %s en recorrer todas las imagenes" % (init_loop - end_loop))
logging.info("Tarda %s en recorrer todas las imagenes" % (init_loop - end_loop))

# codificamos las labels usando LabelEncoder
le = LabelEncoder()
le_labels = le.fit_transform(labels)

# obtenemos las dimensiones de las labels de entrenamiento
print ("[STATUS] training labels: {}".format(le_labels))
print ("[STATUS] training labels shape: {}".format(le_labels.shape))

logging.info((" training labels: {}".format(le_labels)))
logging.info(" training labels shape: {}".format(le_labels.shape))


# save features and labels
h5f_data = h5py.File(features_path, 'w')
h5f_data.create_dataset('dataset_1', data=np.array(features))

h5f_label = h5py.File(labels_path, 'w')
h5f_label.create_dataset('dataset_1', data=np.array(le_labels))

h5f_data.close()
h5f_label.close()

# guardamos el modelo y los pesos
model_json = model.to_json()
with open(model_path + str(test_size) + ".json", "w") as json_file:
	json_file.write(model_json)

# guardamos los pesos
model.save_weights(model_path + str(test_size) + ".h5")
print("[STATUS] saved model and weights to disk..")
logging.info("saved model and weights to disk..")

print ("[STATUS] features and labels saved..")
logging.info(" features and labels saved..")



print("--------------- Feature extraction completed. --------------------")
logging.info("--------------- Feature extraction completed. --------------------")

print("--------------- Start training. --------------------")
logging.info("--------------- Start training. --------------------")

# importamos las features y labels
h5f_data  = h5py.File(features_path, 'r')
h5f_label = h5py.File(labels_path, 'r')

features_string = h5f_data['dataset_1']
labels_string   = h5f_label['dataset_1']

features = np.array(features_string)
labels   = np.array(labels_string)

h5f_data.close()
h5f_label.close()



print("[INFO] training started...")
logging.info("[INFO] training started...")
# dividimos  en train y test
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(features),
                                                                  np.array(labels),
                                                                  test_size=test_size,
                                                                  random_state=seed)



# utilizamos  logistic regression como modelo
print("[INFO] creating model...")
logging.info("[INFO] creating model...")

model = LogisticRegression(random_state=seed)
model.fit(trainData, trainLabels)


preds = model.predict(testData)

# escribir classification_report
f.write("{}\n".format(classification_report(testLabels, preds)))
f.close()

#Guardamos el modelo
print ("[INFO] saving model...")
logging.info(" saving model...")
pickle.dump(model, open(classifier_path, 'wb'))



# get the list of training lables
#labels = sorted(list(os.listdir(train_path)))


# end time
end = time.time()
print ("[STATUS] end time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
logging.info("[STATUS] end time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))






