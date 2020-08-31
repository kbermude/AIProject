import os
import shutil
import random as rnd
import errno

path1 = os.path.abspath(os.path.dirname(__file__))
support= "C:/Users/ferva/Desktop/AIProject-master/data"
types=["cardboard/","glass/","metal/","paper/","plastic/"]
def randomizar():
    for i in types:
        limpiar(i)
        img=[]
        path = os.path.join(path1, "../original/"+i)
        pathEntr= "/entrenamiento/"+i
        pathVal="/validacion/"+i
        with os.scandir(path) as ficheros:
            ficheros = [fichero.name for fichero in ficheros if fichero.is_file()]
        a=int(round(len(ficheros)*0.8,0)) #80% - entrenamiento 
        b=len(ficheros)-a #20 validacion
        #print(len(ficheros),a,b)
        while len(img)!=b:
            x=rnd.randint(1,len(ficheros)-1)
            if ficheros[x] not in img:
                img.append(ficheros[x])
        
        for j in range(len(ficheros)):
            if ficheros[j] not in img:
                shutil.copy(path+ficheros[j], support+pathEntr)
            else:
                shutil.copy(path+ficheros[j], support+pathVal)

def limpiar(carpeta):
    l=['/entrenamiento/','/validacion/']
    for i in l:
        directorio=support+i+carpeta
        shutil.rmtree(directorio)
        try:
            os.mkdir(directorio)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

      

    