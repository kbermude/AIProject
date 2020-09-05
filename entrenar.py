import sys
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K 
from data.clasificador import randomizar
import matplotlib.pyplot as plt
K.clear_session()
randomizar()
data_entrenamiento='./data/entrenamiento'
data_validacion='./data/validacion'

#parametros
epocas=50
altura, longitud= 100,100 
batch_size=32 
pasos=1000 
pasos_validacion=200
filtrosCon1=32
filtrosCon2=32
filtrosCon3=64
filtrosCon4=64
tamano_filtro=(5,5)
tamano_filtro2=(3,3)
tamano_pool=(2,2) 
clases=5 

##preprocesamiento de imagenes

entrenamiento_datagen= ImageDataGenerator(
    rescale=1./255,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip = True,
    rotation_range = 90
)

validacion_datagen= ImageDataGenerator(
    rescale=1./255
)
imagen_entrenamiento=entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical'
)

imagen_validacion=validacion_datagen.flow_from_directory(
    data_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical'
)

cnn=Sequential() 

cnn.add(Convolution2D(filtrosCon1, tamano_filtro, padding='same', input_shape=(altura, longitud,3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Convolution2D(filtrosCon2, tamano_filtro, padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Convolution2D(filtrosCon3, tamano_filtro2, padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Convolution2D(filtrosCon4, tamano_filtro2, padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Flatten())

cnn.add(Dense(256, activation='relu'))

cnn.add(Dropout(0.5)) 

cnn.add(Dense(clases,activation='softmax'))

cnn.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='./modelo/best.hdf5', verbose=1, save_best_only=True)
history=cnn.fit_generator(imagen_entrenamiento, epochs=epocas, validation_data=imagen_validacion,callbacks=[checkpointer])

cnn.summary()
directorio= './modelo/' 
if not os.path.exists(directorio):
    os.mkdir(directorio)

cnn.save('./modelo/modelo.h5')
cnn.save_weights('./modelo/pesos.h5')


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'r', label='Precisión de entrenamiento')
plt.plot(epochs, val_acc, 'b', label='Precisión de validación')
plt.title('Precisión de entrenamiento y validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.savefig('Precision.png')
plt.figure()
plt.plot(epochs, acc, 'r',  label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('Accuracy.png')
plt.figure()
plt.plot(epochs, loss, 'r', label='Pérdida de entrenamiento')
plt.plot(epochs, val_loss, 'b', label='Pérdida de validación')
plt.title('Pérdida de entrenamiento y validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.savefig('Perdida.png')
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Loss.png')
