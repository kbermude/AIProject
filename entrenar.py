import sys
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K #para matar sesiones de keras


K.clear_session()

data_entrenamiento='./data/entrenamiento'
data_validacion='./data/validacion'

#parametros
epocas=100
altura, longuitud= 100,100 #redimensionar imagen
batch_size=32 #numero de imagenes a procesar por pasos
pasos=800 #numero de veces que se va a procesar la info por epoca
pasos_validacion=200 #al final de las epocas se correra con el set de datos de la val para ver que aprende el algoritmo
filtrosCon1=32 
filtrosCon2=64
filtrosCon3=32
tamano_filtro=(3,3)
tamano_pool=(2,2) #tamano de filtro de maxpooling
clases=5 #clases de la basura
lr=0.0005 #learning rate

##preprocesamiento de imagenes

entrenamiento_datagen= ImageDataGenerator(
    rescale=1./255,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True
)
#rescale= reecala la imagen 
#shear_range =va a inclinar la imagen para que aprenda que no siempre alguna clase tendra esa direccion
#zoom_range= toma la imagen y por factor aleatorio a algunas le hace zoom, para que aprenda que no siempre la clase saldra completa
#horizontal_flip= toma la imagen y la invierte, para que aprenda direccionamiento
validacion_datagen= ImageDataGenerator(
    rescale=1./255
)
imagen_entrenamiento=entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=(altura, longuitud),
    batch_size=batch_size,
    class_mode='categorical'
)
#entra al directorio, a cada una de las carpetas y las imagenes
#las va a procesar segun la definicion y categorical es la etiqueta que le damos
imagen_validacion=validacion_datagen.flow_from_directory(
    data_validacion,
    target_size=(altura, longuitud),
    batch_size=batch_size,
    class_mode='categorical'
)

cnn=Sequential() #le decimos que son varias capas secuenciales

cnn.add(Convolution2D(filtrosCon1, tamano_filtro, padding='same', input_shape=(altura, longuitud,3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Convolution2D(filtrosCon2, tamano_filtro, padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Convolution2D(filtrosCon3, tamano_filtro, padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Flatten())

cnn.add(Dense(256, activation='relu')) #anade una capa donde cada neurona esta conectada a las neuronas de la capa anterior

cnn.add(Dropout(0.5)) #a la capa densa durante el entrenamiento se le apagara 50% de las neuronas cada paso, para evitar overfitting
#prendiendo caminos alternos para clasificar la data

cnn.add(Dense(clases,activation='softmax'))
#con esta ultima capa lo que hacemos es ver los valores porcentuales de ver a que categoria pertenece

cnn.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])
#durante el entrenamiento su fucion de perdida sera categorical_crossentropy con una optimizacion lr y la metrica es para saber que tan bien esta aprendiendo la red

cnn.fit_generator(imagen_entrenamiento, steps_per_epoch=int(pasos/batch_size), epochs=epocas, validation_data=imagen_validacion, validation_steps=int(pasos_validacion/batch_size))
#va a correr 1000 pasos en las epocas y luego de cada correra 300 pasos de las de validacion

directorio= './modelo/' #directorio donde va a quedar guardado el modelo
if not os.path.exists(directorio):
    os.mkdir(directorio)

cnn.save('./modelo/modelo.h5')
cnn.save_weights('./modelo/pesos.h5')