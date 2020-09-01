import os
import shutil
import random as rnd
import errno

path1 = os.path.abspath(os.path.dirname(__file__))
types=["cardboard/","glass/","metal/","paper/","plastic/"]

def randomizar():
    #z=0
    #y=0
    #w=0
    for i in types:
        img=set()
        path = os.path.join(path1, "../original/"+i)
        pathEntr= os.path.join(path1, "./entrenamiento/"+i)
        pathVal=os.path.join(path1, "./validacion/"+i)
        
        limpiar(pathEntr)
        limpiar(pathVal)
        with os.scandir(path) as ficheros:
            ficheros = [fichero.name for fichero in ficheros if fichero.is_file()]
        a=int(round(len(ficheros)*0.8,0)) #80% - entrenamiento 
        #print(len(ficheros))
        b=len(ficheros)-a #20 validacion
        #print(len(ficheros),a,b)
        #z+=a
        #y+=b
        #w+=a+b
        while len(img)<b:
            x=rnd.randint(0,len(ficheros)-1)
            if ficheros[x] not in img:
                img.add(ficheros[x])
        for j in range(len(ficheros)):
            if ficheros[j] not in img:
                shutil.copy(path+ficheros[j], pathEntr)
            else:
                shutil.copy(path+ficheros[j], pathVal)
    #print(z,y,w)

def limpiar(carpeta):
    shutil.rmtree(carpeta)
    try:
        os.mkdir(carpeta)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
      

#randomizar()