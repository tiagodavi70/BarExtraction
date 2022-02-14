# Needed imports
# -*- coding: utf-8 -*-
from sklearn.cluster import KMeans
import cv2
import numpy as np
from pytesseract import Output
import pytesseract as ocr
import pandas as pd
import math
from matplotlib import pyplot as plt
from operator import itemgetter
import json
from os import listdir
from os.path import isfile
from os.path import join
import sys
from random import randint

# ----------------FUNTIONS------------------ #
def applycontrast(image, alfa, beta):
    img = image.copy()
    return cv2.convertScaleAbs(img * alfa + beta)


def sharpen(image, k=3):
    img = image.copy().astype(np.int16)
    media = cv2.blur(img, (k, k))
    residual = img - media.astype(np.int16)
    shp = img + residual
    return cv2.convertScaleAbs(shp)


def drawocrrects(img, args, color):
    for i in range(args.shape[0]):
        espaco = 4
        x, y, w, h = args.values[i]
        cv2.rectangle(img, (x - espaco, y - espaco), (x+w+espaco, y+h+espaco), (98, 133, 98), 3)


# view module
def layoutsimage(rows=1, cols=1):
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4,rows*6), dpi=200)
    if rows != 1 or cols != 1:
        for ax in axes:
            ax.clear()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    else:
        axes.get_xaxis().set_visible(False)
        axes.get_yaxis().set_visible(False)

    return axes

def preprocess(params):
    ddepth = cv2.CV_16S

    # contrast = applycontrast(img_org, alfa=1.5, beta=10)

    contrast = applycontrast(img_ocr, alfa=1.2, beta=5)
    sharp = sharpen(contrast, k=5)

    img_gray = cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY)

    kernel_size = 3
    lpl_depth = cv2.Laplacian(img_gray, ddepth, kernel_size)
    img_laplace = cv2.convertScaleAbs(lpl_depth)

    img_laplace = np.where(img_laplace[:, :] == 0, 255, 0)
    # img_laplace = cv.convertScaleAbs(img_laplace)

    thresh = np.where(img_gray[:, :] != 255, 0, 255)
    if (params['isDilate']):
        img_laplace = cv2.dilate(img_laplace.astype(np.uint8), (3, 3))
        thresh = cv2.dilate(thresh.astype(np.uint8), (3, 3))

    thresh = thresh.astype(np.int32)
    img_laplace = img_laplace.astype(np.int32)

    return contrast, sharp, img_gray, img_laplace, thresh, cv2.addWeighted(thresh, .7, img_laplace, .3, 0)

def applyocr(params):

    data = ocr.image_to_data(params['img'], output_type=Output.DICT)
    df = pd.DataFrame(data)
    # print(df)
    isRecog = df['text'].str.strip().str.len() != 0
    isOnlyNumber = df['text'].str.strip().str.isdigit()
    isWord = np.bitwise_not(isOnlyNumber)

    return df[isRecog & isWord],  df[isRecog & isOnlyNumber]

def displayimages(listimg, shape=''):
    totalimgs = len(listimg)
    axes = layoutsimage(1, totalimgs)
    if totalimgs != 1:
        for i in range(totalimgs):
            axes[i].set_title(listimg[i]['title'])
            axes[i].imshow(listimg[i]['img'] if len(listimg[i]['img'].shape) == 2
                           else cv2.cvtColor(listimg[i]['img'],cv2.COLOR_BGR2RGB), cmap='gray')
    else:
        axes.set_title(listimg[0]['title'])
        axes.imshow(listimg[0]['img'] if len(listimg[0]['img'].shape) == 2
                    else cv2.cvtColor(listimg[0]['img'],cv2.COLOR_BGR2RGB), cmap='gray')
    plt.show()

def getaxis(numbers):
    tvc = numbers['top'].value_counts()
    #print('tvc', tvc)
    top_values = np.array(tvc[tvc != 1].index[:].values)
    nohori = pd.DataFrame()
    if len(top_values) != 0:
        for i in range(top_values.shape[0]):
            v = top_values[i]
            #print(numbers['top'])
            condition_max = numbers['top'] <= v + 10
            condition_min = numbers['top'] >= v - 10
            nohori = nohori.append(numbers[np.bitwise_and(condition_max.values, condition_min.values)])
        nohori = numbers[~numbers.top.isin(nohori.top)]
    else:
        nohori = numbers.copy()

    #print('no horizontal', nohori)
    lvc = nohori['left'].value_counts()
    left_values = np.array(lvc[lvc != 1].index[:].values)
    #print(left_values, len(left_values))
    lefts = pd.DataFrame()
    if len(left_values) > 0:
        for i in range(left_values.shape[0]):
            l = left_values[i]
            condition_max = numbers['left'] <= l + 10
            condition_min = numbers['left'] >= l - 10
            lefts = lefts.append(numbers[np.bitwise_and(condition_max.values, condition_min.values)])
        return lefts
    else:
        return nohori

def getnearestnumbers(numbers, xAxis):
    v1,v2,h,p = 0,0,sys.maxsize,0
    d1, d2 = 0,0
    #print(numbers)

    if(numbers.shape[0] != 0):

        for i in range(0, numbers.shape[0]-1):
            df1 = numbers[['text', 'top']].iloc[i]
            df2 = numbers[['text', 'top']].iloc[i+1]

            f_v1 = int(df1['text'])
            f_v2 = int(df2['text'])
            f_h = abs(df2['top'] - df1['top']) # if df2['top'] > df1['top'] else df1['top'] - df2['top']

            if f_h < h:
                h = f_h
                v1 = f_v1
                v2 = f_v2
                d1 = df1['top']
                d2 = df2['top']

            # ideally we remove the row

        if v1 == 0 and v2 != 0:
            h = abs(d2 - xAxis)
            p = 'partial'

        if v1 != 0 and v2 == 0:
            h = abs(d1 - xAxis)
            p = 'partial'

        if (v1 != 0 and v2 != 0):
            p = 'total'

        if(v1 == 0 and v2 == 0):
            p = 'fault'

    else:
        p = 'fault'

        #print(v1, v2)
    #p = numbers[['top']].iloc[0].values
    #print(p)
    return {'v1': v1, 'v2': v2, 'h': h, 'p': p}

def getaxis(numbers):
    tvc = numbers['top'].value_counts()
    #print('tvc', tvc)
    top_values = np.array(tvc[tvc != 1].index[:].values)
    nohori = pd.DataFrame()
    if len(top_values) != 0:
        for i in range(top_values.shape[0]):
            v = top_values[i]
            #print(numbers['top'])
            condition_max = numbers['top'] <= v + 10
            condition_min = numbers['top'] >= v - 10
            nohori = nohori.append(numbers[np.bitwise_and(condition_max.values, condition_min.values)])
        nohori = numbers[~numbers.top.isin(nohori.top)]
    else:
        nohori = numbers.copy()

    #print('no horizontal', nohori)
    lvc = nohori['left'].value_counts()
    left_values = np.array(lvc[lvc != 1].index[:].values)
    #print(left_values, len(left_values))
    lefts = pd.DataFrame()
    if len(left_values) > 0:
        for i in range(left_values.shape[0]):
            l = left_values[i]
            condition_max = numbers['left'] <= l + 10
            condition_min = numbers['left'] >= l - 10
            lefts = lefts.append(numbers[np.bitwise_and(condition_max.values, condition_min.values)])
        return lefts
    else:
        return nohori

def update(parametro):

    #cv2.imshow('Encontrados', imageRes)
    Scale = parametro[0]/10
    DilateErodeKernelSize = parametro[1]
    DilateIterations = parametro[2]
    ErodeIterations = parametro[3]
    GaussianFilterKernel = parametro[4]
    AddWeighted1a = parametro[5]/10
    AddWeighted2a = -(parametro[6]/10)


    image = imageRes
    imge = cv2.resize(image, None, fx=Scale, fy=Scale, interpolation=cv2.INTER_NEAREST)
    imge = cv2.cvtColor(imge, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((DilateErodeKernelSize, DilateErodeKernelSize), np.uint8)
    imge = cv2.dilate(imge, kernel, iterations=DilateIterations)
    imge = cv2.erode(imge, kernel, iterations=ErodeIterations)

    gaussian_1 = cv2.GaussianBlur(imge, (GaussianFilterKernel, GaussianFilterKernel), 10.0)
    imge = cv2.addWeighted(imge, AddWeighted1a, gaussian_1, AddWeighted2a, 0, imge)

    imge = cv2.threshold(imge, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # OCR Functions
    img_labeled = imge.copy()
    isDilate = False
    v1, v2 = 0, 0
    run = 0
    image_labeled = 0
    dict = 0
    while ((v1 == 0 and v2 == 0) or isDilate) and run != 3:
        run += 1
        if run == 2 and isDilate:
            continue

        # print isDilate
        contrast, sharp, img_gray, img_laplace, thresh, w_sum = preprocess(
            {'alfa': 1.5, 'beta': 10, 'isDilate': isDilate})
        words, numbers = applyocr({'img': imge})

        # drawocrrects(img_labeled, words[['left', 'top', 'width', 'height']], (255, 0, 0))

        drawocrrects(img_labeled, numbers[['left', 'top', 'width', 'height']], (0, 255, 0))

    lateralaxis = getaxis(numbers)
   # print("For index: ", index, "Found: ", lateralaxis.shape[0])
    #print(lateralaxis.to_string())
    #print("MODIFICADO")
    #print(lateralaxis.drop_duplicates().to_string())
    #print(lateralaxis)
    #cv2.imshow("Encontrados", img_labeled)
    #cv2.waitKey(0)

    lateralaxis = getaxis(numbers)

    return lateralaxis.drop_duplicates().shape[0]

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

def generateRandomParams():
    param = np.zeros([7], np.uint8)
    param[0] = randint(10, 15)
    param[1] = 1
    param[2] = randint(1, 11)
    param[3] = randint(1, 11)
    impar = randint(1, 11)
    if (impar % 2) == 0:
        impar = impar + 1

    param[4] = impar
    param[5] = randint(5, 21)
    param[6] = randint(4, 11)

    return param

# --------------END FUNTIONS---------------- #

debug = False
ocrTotal = 0

#imageRes = cv2.imread("TESTE/bar59.png")
pathImages = 'segment/'

listaNomes = getallnamesfromdirectory(pathImages)

ocrTotal = 0
melhores = []

results = np.zeros([len(listaNomes), 3], np.int16)
for z in range(len(listaNomes)):
    print("IMAGEM ", z, " - File: ", listaNomes[z])
    usefull = False
    imageRes = cv2.imread(join(pathImages, listaNomes[z]))
    img_ocr = imageRes.copy()
    img_labeled = imageRes.copy()

    atual = []
    hits = 0

    for melhor in melhores:
        hits = update(melhor)
        if hits >= 2:
            atual = melhor
            print("ANTIGO")
            break


    #print(Params)

    startcount = 0
    while hits < 2 and startcount < 100:
        atual = generateRandomParams()
        hits = update(atual)
        if hits >= 2:
            melhores.append(atual)
        startcount = startcount + 1

    print("FUNCIONOU COM: ", atual)




for melhor in melhores:
    print(melhor)

print(len(melhores))