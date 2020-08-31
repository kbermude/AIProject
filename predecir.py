import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

altura, longuitud= 100,100
modelo='./modelo/modelo.h5'
pesos='./modelo/pesos.h5'
cnn=load_model(modelo)
cnn.load_weights(pesos)
types=["cardboard","glass","metal","paper","plastic"]

def predict(imagen):
    x=load_img(imagen,target_size=(longuitud,altura))
    x=img_to_array(x)
    x=np.expand_dims(x,axis=0)
    arreglo=cnn.predict(x)
    resultado=arreglo[0]
    respuesta=np.argmax(resultado)
    return types[respuesta]


l=['cardboard','metal','metal','metal','glass','paper','cardboard']  

def comprobar():
    for i in range(7):
        var="prueba"+str(i)+".jpg"
        a=predict(var)
        print(a,l[i],a==l[i])

