from os import listdir
from os.path import join, isfile

import numpy as np
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
pathCropped = 'segment2/'
listaNomes = getallnamesfromdirectory(pathImages)
for z in range(len(listaNomes)):
    print('IMAGEM ATUAL: ', z, listaNomes[z])
    imageRes = cv2.imread(join(pathImages, listaNomes[z]))
    #imageRes = cv2.imread('png/bar1446.png')
    dict = ET.extract(imageRes)
    #print(dict['Barras'])
    #print("TAMANHO: ", len(dict['Barras']))
    if(len(dict['Barras']) > 0):
        h, w = imageRes.shape[:2]
        y0 = 0
        y = h
        x0 = 0
        x = dict['Barras'][0][0]
        imageLeft = imageRes[y0:y, x0:x]

        y0 = 0
        y = h
        x0 = dict['Barras'][len(dict['Barras']) - 1][0] + dict['Barras'][0][2]
        i = 1
        while(x0 > w):
            x0 = dict['Barras'][len(dict['Barras']) - i][0] + dict['Barras'][0][2]
            i = i + 1

        x = w
        imageRight = imageRes[y0:y, x0:x]
        print("lefter", x0)

        #cv2.imshow("LeftCrop", imageLeft)
        #cv2.imshow("RightCrop", imageRight)
        #cv2.waitKey(0)

        pathLeft = "{}{}L.png".format(pathCropped, listaNomes[z][:-4])
        pathRight = "{}{}R.png".format(pathCropped, listaNomes[z][:-4])

        hL, wL = imageLeft.shape[:2]
        hR, wR = imageRight.shape[:2]

        imgeRightBW = cv2.cvtColor(imageRight, cv2.COLOR_BGR2GRAY)
        imgeLeftBW = cv2.cvtColor(imageLeft, cv2.COLOR_BGR2GRAY)

        #imageRightBin = cv2.threshold(imgeRightBW, 0, 255, cv2.THRESH_OTSU)[1]
        #imageLeftBin =  cv2.threshold(imgeLeftBW, 0, 255, cv2.THRESH_OTSU)[1]

        imageLeftBin = imageLeft
        imageRightBin = imageRight


        imageRightBinGrad = cv2.Sobel(imageRightBin,cv2.CV_64F,1,0,ksize=5)
        abs_imageRightBinGrad = np.absolute(imageRightBinGrad)
        imageRightBinGrad = np.uint8(abs_imageRightBinGrad)


        imageLeftBinGrad = cv2.Sobel(imageLeftBin, cv2.CV_64F, 1, 0, ksize=5)
        abs_imageLeftBinGrad = np.absolute(imageLeftBinGrad)
        imageLeftBinGrad = np.uint8(abs_imageLeftBinGrad)


        #cv2.imshow("GRADIENTE DIREITA", imageRightBinGrad)
        #cv2.imshow("GRADIENTE ESQUERDA", imageLeftBinGrad)
        #cv2.waitKey(0)

        rightcount = np.count_nonzero(imageRightBinGrad) / (hR * wR)

        leftcount = np.count_nonzero(imageLeftBinGrad) / (hL * wL)

        if(rightcount >= leftcount):
            cv2.imwrite(pathRight, imageRight)
            print("IMAGEM DIREITA {}".format(rightcount))
            #cv2.imshow("RightBin", imageRightBin)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()


        else:
            cv2.imwrite(pathLeft, imageLeft)
            print("IMAGEM ESQUERDA {}".format(leftcount))
            #cv2.imshow("LeftBin", imageLeftBin)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()


        # cv2.imshow("LeftBin", imageLeftBin)
        # cv2.imshow("RightBin", imageRightBin)
        # cv2.waitKey(0)
        # #print(pathLeft)
        # #print(pathRight)
        # cv2.waitKey(0)
        # cv2.imwrite(pathLeft, imageLeft)
        # cv2.imwrite(pathRight, imageRight)

