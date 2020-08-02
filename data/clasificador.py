import os
import shutil

path1 = os.path.abspath(os.path.dirname(__file__))
support= "c:/Users/eduar/git/AIProject/data"
types=["cardboard/","glass/","metal/","paper/","plastic/"]
for i in types:
    path = os.path.join(path1, "../original/"+i)
    pathEntr= "/entrenamiento/"+i
    pathVal="/validacion/"+i
    with os.scandir(path) as ficheros:
        ficheros = [fichero.name for fichero in ficheros if fichero.is_file()]
    a=int(round(len(ficheros)*0.8,0)) #80% - entrenamiento 
    #b=len(ficheros)-a #20 validacion
    #print(len(ficheros),a,b)
    for j in range(len(ficheros)):
        if a>0:
            shutil.copy(path+ficheros[j], support+pathEntr)
            a+=-1
        else:
            shutil.copy(path+ficheros[j], support+pathVal)
        

    