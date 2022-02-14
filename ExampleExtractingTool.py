import random
from os import listdir
from os.path import join, isfile
import pandas as pd
import ExtractingTool as ET
import cv2


def getallnamesfromdirectory(path, extension='', numfiles=-1):
    names = []
    dir_p = path
    filenames = [f for f in listdir(dir_p) if isfile(join(dir_p, f))]
    totalimagestoload = numfiles if numfiles != -1 else len(filenames)

    for f in filenames:
        if extension != '':
            if f[-4:] == extension:
                if (totalimagestoload != 0):
                    totalimagestoload -= 1
                    names.append(f)
                else: break
        else:
            if (totalimagestoload != 0):
                totalimagestoload -= 1
                names.append(f)
            else:
                break
    return names


pathImages = 'png/'
pathJsons = "jsons/"
listaNomes = getallnamesfromdirectory(pathImages)
random.shuffle(listaNomes)

#listaNomes = ['bar593.png']

for z in range(len(listaNomes)):
    print('IMAGEM ATUAL: ', z, listaNomes[z])
    imageRes =  cv2.imread(join(pathImages, listaNomes[z]))
    dictExtracted = ET.extract(imageRes)

    #print(dict)
    filename = listaNomes[z]
    df = pd.DataFrame(dictExtracted)
    df.to_json(join(pathJsons, filename + '.json'))

#if __name__ == '__main__':