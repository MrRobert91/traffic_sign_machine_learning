from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import os, logging
import datetime
from skimage import color, exposure, transform, io
import numpy as np
import glob
from keras.utils.np_utils import to_categorical
from keras.callbacks import LearningRateScheduler, ModelCheckpoint


#Corleone
code_path="/home/drobert/tfg/traffic_sign_machine_learning/cnn6l/"
dataset_path='/home/drobert/tfg/'

fichero_log = (code_path +'cnn6l.log')

print('Archivo Log en ', fichero_log)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    filename = fichero_log,
                    filemode = 'a',)# w for new log each time

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

def get_class(img_path):
    return int(img_path.split('/')[-2])

def lr_schedule(epoch):
    return lr * (0.1 ** int(epoch / 10))


batch_size = 32
epochs = 20 #ponemos 5 para que sea mas rapido, normalmente 30
lr = 0.01

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
    img = preprocess_img(io.imread(img_path))
    label = get_class(img_path)
    imgs.append(img)
    labels.append(label)

X = np.array(imgs, dtype='float32')
Y = np.asarray(labels)

x_train= X
y_train = to_categorical(Y, NUM_CLASSES)



#-------------------------------------------------------
# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

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

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# train the model on the new data for a few epochs
#model.fit_generator(...)
model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              verbose=1,
              callbacks=[LearningRateScheduler(lr_schedule)]
              )

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
#model.fit_generator(...)
model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              verbose=1,
              callbacks=[LearningRateScheduler(lr_schedule),
                         ModelCheckpoint(filepath, save_best_only=True)]

              )