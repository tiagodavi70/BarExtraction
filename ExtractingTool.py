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
import firevent
import json
import imutils

#ocr.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'
#TESTECOMMIT

DEBUG = False
DEBUGADVANCED = False

# ------- Helpíng Functions ------ #

def filterImage(sourceimg):
    scale = 2
    image = cv2.bilateralFilter(sourceimg, 3, 100, 100)
    imge = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    imgResized = imge
    imge = cv2.cvtColor(imge, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((2, 2), np.uint8)
    imge = cv2.erode(imge, kernel, iterations=1)
    imge = cv2.dilate(imge, kernel, iterations=1)


    imge = cv2.GaussianBlur(imge, (3, 3), 10.0)
    #cv2.imshow("FILTERED", imge)
    #cv2.waitKey(0)

    return imge, imgResized

def findBarMasks(image):

    # Create mask, mask list and test index vector
    h, w = image.shape[:2]
    testIndex = np.zeros((h, w), np.bool) # If the value is true we don't need to analyze starting in that point anymore

    maskList = [] # In here we store the masks we find
    # mask = np.zeros((h+2, w+2), np.uint8)


    # Iterate through image and only test for blobs on pixels that are not True on test vector
    for l in range(0, len(testIndex)):
        for c in range(0, len(testIndex[0])):
            if not testIndex[l][c]:  # Only if that pixel isn't yet on

                # Reinitialise the mask so it will always check all the pixels
                mask = np.zeros((h + 2, w + 2), np.uint8)

                # FloodFill low and high threshold values
                lo = 20
                hi = 20
                # Run floodFill centered in the current (c, l) pixel
                cv2.floodFill(image, mask, (c, l), (255, 255, 255), (lo,) * 3, (hi,) * 3, 4 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE)

                # Mask is bigger than actual image, adjust it so the contour match with the shape in the real image
                cutMask = mask[1:len(mask), 1:len(mask[0])]

                # Add each new element found as a binary image in the maskList array
                maskList.append(cutMask)

                if(DEBUGADVANCED):
                    cv2.imshow("mascara", cutMask)
                    cv2.waitKey(0)

                # See where the found mask is equal to white and set they as true on the test vector so we won't iterate through them
                a = np.where(mask == 255)

                # Set to true the value of all the pixels that floodfill found connected
                for x in range(0, len(a[0])):
                    col = a[0][x]
                    lin = a[1][x]

                    # Remember that each pixel (col, lin) on the image corresponde to a (col + 1, lin + 1) pixel in the image
                    testIndex[col - 1][lin - 1] = True

    #print("Lenght of masks found on image {}".format(len(maskList)))
    return maskList


def findMasksContours(maskList, image):
    # Iterate through masks list and save only the ones with the area at least 80% of the area of it's bounding box

    # Initialize a new array that will be used to store the contours on each mask
    maskContours = []
    smallContours = []
    barsimgbefore = image.copy()

    # get image dimensions
    h1, w1 = image.shape[:2]
    #Iterate through masks
    for masks in maskList:

        #Get the contours
        contours, hierarchy = cv2.findContours(masks, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #Calculate the area of the image
        AreaImg = h1 * w1
        #print("Lenght of contours vector for this mask {}".format(len(contours)))

        #Iterate through the contours we found
        for q in range(len(contours)):
            #Get the contour area
            areaCont = cv2.contourArea(contours[q])
            #get the contour bouding box
            x, y, w, h = cv2.boundingRect(contours[q])
            #Calculare the bounding box area
            areaBox = w * h

            escala = 2
            # sale when implementOCR

            # If the area of the contour is at leats 80% of the area of it's bouding box
            # The area of the bouding box is at least 50% smaller than the total area of the image
            # And if the contour isn't inside another contour (hierarchy [0, k, 3] = -1; k being the index of the contour)
            if areaCont >= (areaBox * 0.8) and areaBox < AreaImg * 0.5 and hierarchy[0, q, 3] == -1 :
                #Add that contour to the list of bars

                maskContours.append(contours[q])

                if(DEBUGADVANCED):
                    cv2.drawContours(barsimgbefore, contours, q, (0, 255, 0), 1)
                    cv2.imshow("Contorno atual", barsimgbefore)
                    cv2.waitKey(0)
                    print("Hierarchy of this contour: {}".format(hierarchy[0, q, 3]))

    #print("Lenght of contours vector is {}".format(len(maskContours)))
    return maskContours

def inferOrientation(contornos):

    # False if graph is horizontal, true if graph is vertical
    orientation = None
    widthlist = []  # List to store the bars widhts
    heightlist = [] # Lists to store the bars heights

    for contornins in contornos:
        x, y, w, h = cv2.boundingRect(contornins) #Get each bar dimensions
        # Add widht and height to the lists
        widthlist.append(w) #Add to list
        heightlist.append(h) #add to list

    #print(widthlist)
    #print(heightlist)
    # Create two image-like vectors to store the widhts ans heights
    a = np.zeros((1, len(widthlist)), np.uint8)
    b = np.zeros((1, len(heightlist)), np.uint8)

    #Storing the actual heights and widhts in the image-like vectors
    for x in range(len(widthlist)):
        a[0][x] = widthlist[x]

    for x in range(len(heightlist)):
        b[0][x] = heightlist[x]

    #Calculating the histogram of the heights and widhts
    hist1 = cv2.calcHist([a], [0], None, [100], [0, 256])
    hist2 = cv2.calcHist([b], [0], None, [100], [0, 256])
    # hist = cv2.calcHist(heightlist, [0], None, [256], [0, 256])

    # Plot the histograms
    if(DEBUG):
        plt.figure()
        plt.title("Histograma")
        plt.xlabel("Largura e Altura da Barra")
        plt.ylabel("Nº de Barras")
        plt.plot(hist1)
        plt.plot(hist2)
        plt.xlim([0, 100])
        plt.show()
        cv2.waitKey(0)

    # print(hist1)
    # print(hist2)

    bigger1 = 0
    bigger2 = 0

    # Looking for the biggest variation in each histogram
    for indexes in hist1:
        if indexes[0] > bigger1:
            bigger1 = indexes[0]

    for indexes in hist2:
        if indexes[0] > bigger2:
            bigger2 = indexes[0]

    #print("Maior da largura {}".format(bigger1))
    #print("Maior da altura {}".format(bigger2))


    # Using the information above to infer the graph orientation (Is the orientation with the bigger variation)
    if bigger1 > bigger2:
        if(DEBUG):
            print("GRAFICO VERTICAL")
        orientation = True
    else:
        if (DEBUG):
            print("GRAFICO HORIZONTAL")
        orientation = False

    return orientation

def getXAxis(mascaras, orientacao, imger):

    h1, w1 = imger.shape[:2]

    topList = [] # This list will store all the position of the top of the bars
    bottomList = [] # This list will store all the position of the bottom of the bars
    # Bottom and top here is used in a poetic way, if the chart i horizontal they correspond to the
    # left and right of

    bigger1 = 0  # The biggest value in histTop will be stored here
    bigger2 = 0  # The biggest value in histBottom will be stored here
    index1 = 0  # The index of the biggest in histTop (THIS IS WHERE OUR X AXIS MAY BE)
    index2 = 0  # The index of the biggest in histBottom (THIS IS WHERE OUR X AXIS MAY BE)

    #If the chart is vertical
    if orientacao:

        for mascara in mascaras:
            x, y, w, h = cv2.boundingRect(mascara)  # Get each bar dimensions
            # Add widht and height to the lists
            topList.append(y)
            bottomList.append(y + h)

        #Create image-like vectors
        topImage = np.zeros((1, len(topList)), np.uint16)
        bottomImage = np.zeros((1, len(bottomList)), np.uint16)


        #Add the values in the image-like vectors
        for k in range(len(topList)):
            topImage[0][k] = topList[k]

        for o in range(len(bottomList)):
            bottomImage[0][o] = bottomList[o]

        # Calculating the histogram of the heights and widhts
        histTop = cv2.calcHist([topImage], [0], None, [h1], [0, h1])
        histBottom = cv2.calcHist([bottomImage], [0], None, [h1], [0, h1])
        # hist = cv2.calcHist(heightlist, [0], None, [256], [0, 256])

        # Plot the histograms
        if(DEBUG):
            plt.figure()
            plt.title("Histograma")
            plt.xlabel("Posição do Topo e Base da Barra")
            plt.ylabel("N de Barras")
            plt.plot(histTop)
            plt.plot(histBottom)
            plt.xlim([0, h1])
            plt.show()
            cv2.waitKey(0)


        #Find the mode and where the mode is of each
        for h in range(len(histTop)):
            if histTop[h][0] > bigger1:
                bigger1 = histTop[h][0]
                index1 = h

        for h in range(len(histBottom)):
            if histBottom[h][0] > bigger2:
                bigger2 = histBottom[h][0]
                index2 = h


    # The bigger mode is where the X axis is and it's index stores the value of where the X axis is
    if(bigger1 > bigger2):
        if(DEBUG):
            print("EIXO X EM {}".format(index1))
        return index1

    else:
        if (DEBUG):
            print("EIXO X EM {}".format(index2))
        return index2

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
        #x = x//2
        #y = y//2
        #w = w//2
        #h = h//2
        cv2.rectangle(img, (x - espaco, y - espaco), (x+w+espaco, y+h+espaco), (98, 133, 98), 2)


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


def applyocr(params):

    data = ocr.image_to_data(params['img'], output_type=Output.DICT)
    df = pd.DataFrame(data)
    #print(df)
    if(DEBUG):
        print("DATAFRAME OCR ORIGINAL:")
        print(df.to_string())


    isRecog = df['text'].str.strip().str.len() != 0
    isOnlyNumber = df['text'].str.replace('.', '').str.isdigit()
    isWord = np.bitwise_not(isOnlyNumber)

    return df[isRecog & isWord],  df[isRecog & isOnlyNumber]

# ESSAS FUNÇÕES AQUI SAO PARA ACHAR As LABELS E TITUlOS
def OCR2(params):

    data = ocr.image_to_data(params['img'], output_type=Output.DICT)
    df = pd.DataFrame(data)

    df = df.drop(df[df.conf.astype(int) < 40].index)
    df = df.drop(df[np.bitwise_not(df.text.str.isalpha())].index)
    isRecog = df['text'].str.strip().str.len() != 0
    isOnlyNumber = df['text'].str.replace('.', '').str.isdigit()

    isWord = np.bitwise_not(isOnlyNumber)

    return df[isRecog & isWord]

def getTitles(image, barrasSorted):
    # TOP

    higher = [barrasSorted[0][1], 0]
    for u in range(len(barrasSorted)):
        if (barrasSorted[u][1] < higher[0]):
            higher[0] = barrasSorted[u][1]
            higher[1] = u

    cropTOP = image[:higher[0], barrasSorted[0][0]:barrasSorted[-1][0] + barrasSorted[-1][2]]

    #LEFT
    cropLEFT = image[higher[0]:barrasSorted[0][1] + barrasSorted[0][3], 0:barrasSorted[0][0]]

    #BOTTOM
    cropBOTTOM = image[barrasSorted[0][1] + barrasSorted[0][3]: , barrasSorted[0][0]:barrasSorted[-1][0] + barrasSorted[-1][2]]

    #RIGHT
    cropRIGHT = image[higher[0]:barrasSorted[0][1] + barrasSorted[0][3], barrasSorted[-1][0] + barrasSorted[-1][2]: ]


    cropLEFT = imutils.rotate_bound(cropLEFT, 90)
    cropRIGHT = imutils.rotate_bound(cropRIGHT, -90)


    dfTOP = runOCR2(cropTOP)

    dfRIGHT = runOCR2(cropRIGHT)
    dfLEFT = runOCR2(cropLEFT)
    dfBOTTOM = runOCR2(cropBOTTOM)


    listaLeft = dfLEFT['text'].tolist()
    listaRight = dfRIGHT['text'].tolist()

    listaSides = listaLeft + listaRight
    print(listaSides)

    try:
        labelY = max(listaSides, key=len)
    except:
        labelY = 'NaN'

    Title = 'NaN'
    top = False
    listaTOP = dfTOP['text'].tolist()
    if(len(listaTOP) > 0):
        top = True
        Title = ''
        for i in range(len(listaTOP)):
            Title = Title + ' ' + listaTOP[i]

    listaBOTTOM = dfBOTTOM['text'].tolist()

    labelX = 'NaN'

    if(top):

        try:
            labelX = max(listaBOTTOM, key=len)
        except:
            labelX = 'NaN'

    else:
        if(len(listaBOTTOM) == 3):
            labelX = listaBOTTOM[0]
            Title = listaBOTTOM[1] + ' ' + listaBOTTOM[2]

    print("TITULO:", Title)
    print("Label X:", labelX)
    print("Label Y:", labelY)

    return Title, labelX, labelY

    # cv2.imshow("ORIGINAL", image)
    # cv2.waitKey(0)

    # cv2.imshow("TOP", cropTOP)
    # cv2.imshow("BOTTOM", cropBOTTOM)
    # cv2.imshow("LEFT", cropLEFT)
    # cv2.imshow("RIGHT", cropRIGHT)
    # cv2.waitKey(0)

def runOCR2(image):

    size = 5


    imagere = cv2.bilateralFilter(image.copy(), 3, 100, 100)
    imge = cv2.resize(imagere, None, fx=size, fy=size, interpolation=cv2.INTER_CUBIC)
    imge = cv2.cvtColor(imge, cv2.COLOR_BGR2GRAY)
    resize = imge.copy()

    kernel = np.ones((1, 1), np.uint8)

    imge = cv2.erode(imge, kernel, iterations=1)
    imge = cv2.dilate(imge, kernel, iterations=1)

    imge = cv2.GaussianBlur(imge, (5, 5), 10.0)

    df = OCR2({'img': imge})
    print(df.to_string())
    espaco = 1
    for i in range(len(df)):
        x, y, w, h, word_num, confidence = df[['left', 'top', 'width', 'height', 'word_num', 'conf']].values[i]
        word_num = int(word_num)
        confidence = int(confidence)

        cv2.rectangle(imge, (x - espaco, y - espaco), (x + w + espaco, y + h + espaco), (98, 133, 98), 2)

    imge = cv2.resize(imge, None, fx=1/size, fy=1/size, interpolation=cv2.INTER_CUBIC)

    # cv2.imshow("TEXTOR", imge)
    # cv2.waitKey(0)

    return df



# ESSAS FUNÇÕES AQUI SAO PARA ACHAR As LABELS E TITUlOS


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

            f_v1 = float(df1['text'])
            f_v2 = float(df2['text'])
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
    return {'v1': v1, 'v2': v2, 'h': h/2, 'p': p}

def calculateScale(numDistance, pixelDistance):

    scaleC = None
    if(pixelDistance > 0):
        scaleC = pixelDistance/numDistance  #NumDistance is the numeric distance as (bigger - smaller)

    return scaleC

def checkDecimalPoint(numbers, imagem):
    thereisapoint = False  #False if no oints were found on the image, true otherwise

    for i in range(len(numbers)):
        padding = 3  #  Padding for the number image segmentation
        size = 10  #  Scale to resize the image
        #print(numbers[['left', 'top', 'width', 'height']].iloc[i].values)
        x, y, w, h = numbers[['left', 'top', 'width', 'height']].values[i]  # bounding box of the current number
        # Casting them to integers, they come as strings
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)

        imgcrop = imagem[y - padding:y + h + padding, x - padding:x + w + padding] #Crop the number in the image
        imgcropresize = cv2.resize(imgcrop, None, fx=size, fy=size, interpolation=cv2.INTER_CUBIC) # Resize to the scale

        imgcropresize = np.where(imgcropresize < 210, 0, imgcropresize) #Binarize

        imgcroprint = imgcropresize.copy()
        imgcroprint = cv2.cvtColor(imgcroprint, cv2.COLOR_GRAY2BGR) #Back to BGR so i can see color when drawing on top
        circles = [None]

        circles = cv2.HoughCircles(imgcropresize, cv2.HOUGH_GRADIENT, 1, 20, param1=30, param2=30, minRadius=0, maxRadius=50) # Look for circles

        #circles = np.uint16(np.around(circles))
        #print("CIRCLES", circles)
        cimg = imgcroprint.copy()

        if circles is not None: #Se tiver algum circulo
            for i in range(len(circles[0])): #Para cada circulo
                #print(circles[0][i][0])
                color = cimg[int(circles[0][i][1]), int(circles[0][i][0])]
                #print("cor", color)
                if(DEBUG):
                    cv2.circle(cimg, (circles[0][i][0], circles[0][i][1]), circles[0][i][2], (0, 255, 0), 2)
                    cv2.imshow('detected circles', cimg)
                    cv2.waitKey(0)

                if(color[0] == 0): #Se ele for preto [0. 0. 0]
                    thereisapoint = True #Marca que achou um circulo

            # for i in circles[0, :]:
            #     # draw the outer circle
            #     print(int(i[0]), int(i[1]))
            #     color = cimg[int(i[0]),int(i[1])]
            #     print("COR", color)
            #     cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            #     #draw the center of the circle
            #
            #     cv2.imshow('detected circles', cimg)
            #     cv2.waitKey(0)

    if(DEBUG):
        if(thereisapoint):
            print("DECIMAL")
        else:
            print("NAO DECIMAL")

    return thereisapoint

def runOCR(imageRes, original, xAxis):
    ponto = 'Not Found'
    size = 2
    image = imageRes
    image = cv2.bilateralFilter(image, 3, 100, 100)
    imge = cv2.resize(image, None, fx=size, fy=size, interpolation=cv2.INTER_CUBIC)
    imge = cv2.cvtColor(imge, cv2.COLOR_BGR2GRAY)
    resize = imge.copy()

    kernel = np.ones((2, 2), np.uint8)

    imge = cv2.erode(imge, kernel, iterations=1)
    imge = cv2.dilate(imge, kernel, iterations=1)

    imge = cv2.GaussianBlur(imge, (3, 3), 10.0)
    #imge = np.where(imge < 200, 0, imge)

    # OCR Functions
    img_labeled = imge.copy()
    v1, v2 = 0, 0
    image_labeled = 0
    dict = 0

    words, numbers = applyocr({'img': imge})

    # drawocrrects(img_labeled, words[['left', 'top', 'width', 'height']], (255, 0, 0))

    if (DEBUGADVANCED):
        drawocrrects(img_labeled, numbers[['left', 'top', 'width', 'height']], (0, 255, 0))
        cv2.imshow("LABELED", img_labeled)
        cv2.waitKey(0)

    lateralaxis = getaxis(numbers)
    # if(numbers.__len__() > 0):
    #     print("SEGUNDO", float(numbers.iloc[1]['text']))
    # print("TAMANHO", numbers.__len__())



    lateralaxis = lateralaxis.drop_duplicates()
    if(DEBUG):
        print("DATAFRAME LATERAL ANTES DE FILTRAR PONTOS")
        print(lateralaxis)


    itensWithDecimalPoints = [] # Store the index of itens that have the 'numbers' filt with decimal points

    if (lateralaxis.__len__() > 0):
        for q in range(lateralaxis.__len__()):

            if(DEBUG):
                print("NUMERO ATUAL", q)


            numero = lateralaxis.iloc[q]['text']

            if '.' in numero:
                itensWithDecimalPoints.append(q)


    if(len(itensWithDecimalPoints) > 0): # IF THERE'S ALREADY AT LEAST ONE FLOATING POINT NUMBER ON THE DATAFRAME
        ponto = "Found by OCR"
        for u in range(lateralaxis.__len__()):
            numero = lateralaxis.iloc[u]['text']
            if '.' not in numero:
                lateralaxis.iat[u, 11] = int(lateralaxis.iat[u, 11]) / 10
            # lateralaxis.iat[itensWithDecimalPoints[u], 11] = '999999'

    else:  #IF THERE'S  NONE THAN WE HAVE TO RUN THE OTHER POINT INFERENCE TOOL
        divideByTen = checkDecimalPoint(numbers, resize)
        if(divideByTen):
            ponto = "Found by Function"
            for u in range(lateralaxis.__len__()):
                lateralaxis.iat[u, 11] = int(lateralaxis.iat[u, 11]) / 10

    if (DEBUG):
        print("DATAFRAME LATERAL DEPOIS DE FILTRAR PONTOS")
        print(lateralaxis)
        print("ITEM COM PONTO", itensWithDecimalPoints)


    dict = getnearestnumbers(lateralaxis, xAxis)
    v1, v2 = dict['v1'], dict['v2']

    if(DEBUG):
        print("DICT DO GENERATE NEAREST", dict.__str__())

        print("DADOS MAGICOS, N1, N2 E DISTANCIA EM PIXELS")
        print(dict['v1'], dict['v2'], dict['h'])


    numDif = abs((dict['v1'] - dict['v2']))
    # print(numDif)
    # print('DICTH', dict['h'])
    # print('DICTP', dict['p'])

    scale = None
    if (dict['p'] != 'fault' and numDif != 0):
        scale = calculateScale(numDif, dict['h'])
        # print('scale', scale)

    if(DEBUG):
        print("NUM DIF: ", numDif)
        print("SCALE: ", scale)

    return scale, dict, ponto

def calulateBars(contornosfinais):
    barras = []
    for p in range(len(contornosfinais)):
        barras.append(cv2.boundingRect(contornosfinais[p]))

    return barras

def calulateZeros(barrasordenadas, xAxis, limpa):
    thresh = 5 #How much in pixels can the distance between two bars vary more than the mode and still be considered normal
    distancias = []
    for i in range((len(barrasordenadas) - 1)):
        distancias.append(barrasordenadas[i+1][0] - barrasordenadas[i][0])

    #print(distancias)

    if(len(distancias) > 0):
        moda = np.min(distancias)
    # TODO: Corrigir moda, cálculo errado(?) e saindo zeros
    #print("MODA", moda)

    indexHole = []
    for j in range(len(distancias)):

        if((distancias[j] > moda + thresh) and moda != 0):
            #print(distancias[j],moda)
            quantos = int(round(distancias[j]/moda)) - 1
            indexHole.append([j, quantos])


   # print("INDEX HOLES: ", indexHole)

    for k in range(len(indexHole)):
         for c in range(indexHole[k][1]):
            cv2.circle(limpa, (barrasordenadas[indexHole[k][0]][0] + ((c + 1) * moda), xAxis), 1, (0, 0, 255), -1)
            barrasordenadas.append((barrasordenadas[indexHole[k][0]][0] + ((c + 1) * moda), xAxis, barrasordenadas[0][2], 0))

    if(DEBUG):
        print("novasBarras", barrasordenadas)
        cv2.imshow("added zeros", limpa)
        cv2.waitKey(0)

    return barrasordenadas

def calculateDataValues(sortedBarras, escala, OCRvalue):
    datavalues = []
    if(OCRvalue == 'total' and escala != None):
        for d in range(len(sortedBarras)):
            datavalues.append((sortedBarras[d][3])/escala)

    elif(OCRvalue == 'fault'):
        for d in range(len(sortedBarras)):
            datavalues.append((sortedBarras[d][3]))

    elif(OCRvalue == 'partial'):
        for d in range(len(sortedBarras)):
            datavalues.append((sortedBarras[d][3]) / escala)

    else:
        for d in range(len(sortedBarras)):
            datavalues.append((sortedBarras[d][3]))
    if(DEBUG):
        print("DATAVALUES:", datavalues)

    return datavalues

def printAXIS(xaxis, imageteste):

    cv2.line(imageteste, (0, xaxis), (446, xaxis), (255, 0, 0), 4)
    cv2.imshow("teste", imageteste)
    cv2.waitKey(0)


def getScaleSegment(imagem, barras):

    returnIMG = None

    if (len(barras) > 0):
        h, w = imagem.shape[:2]
        y0 = 0
        y = h
        x0 = 0
        x = barras[0][0]
        imageLeft = imagem[y0:y, x0:x]

        y0 = 0
        y = h
        x0 = barras[len(barras) - 1][0] + barras[0][2]
        x = w

        i = 1
        while (x0 > w and (len(barras) - i) > 0):
            x0 = barras[len(barras) - i][0] + barras[0][2]
            i = i + 1

        x = w
        imageRight = imagem[y0:y, x0:x]
        #print("lefter", x0)

        # cv2.imshow("LeftCrop", imageLeft)
        # cv2.imshow("RightCrop", imageRight)
        # cv2.waitKey(0)

        hL, wL = imageLeft.shape[:2]
        hR, wR = imageRight.shape[:2]

        imgeRightBW = cv2.cvtColor(imageRight, cv2.COLOR_BGR2GRAY)
        imgeLeftBW = cv2.cvtColor(imageLeft, cv2.COLOR_BGR2GRAY)

        # imageRightBin = cv2.threshold(imgeRightBW, 0, 255, cv2.THRESH_OTSU)[1]
        # imageLeftBin =  cv2.threshold(imgeLeftBW, 0, 255, cv2.THRESH_OTSU)[1]

        imageLeftBin = imageLeft
        imageRightBin = imageRight

        imageRightBinGrad = cv2.Sobel(imageRightBin, cv2.CV_64F, 1, 0, ksize=5)
        abs_imageRightBinGrad = np.absolute(imageRightBinGrad)
        imageRightBinGrad = np.uint8(abs_imageRightBinGrad)

        imageLeftBinGrad = cv2.Sobel(imageLeftBin, cv2.CV_64F, 1, 0, ksize=5)
        abs_imageLeftBinGrad = np.absolute(imageLeftBinGrad)
        imageLeftBinGrad = np.uint8(abs_imageLeftBinGrad)

        # cv2.imshow("GRADIENTE DIREITA", imageRightBinGrad)
        # cv2.imshow("GRADIENTE ESQUERDA", imageLeftBinGrad)
        # cv2.waitKey(0)

        rightcount = np.count_nonzero(imageRightBinGrad) / (hR * wR)

        leftcount = np.count_nonzero(imageLeftBinGrad) / (hL * wL)

        if (rightcount >= leftcount):
            if(DEBUG):
                print("IMAGEM DIREITA {}".format(rightcount))
                cv2.imshow("RightBin", imageRight)
                cv2.waitKey(0)
            returnIMG = imageRight

        else:
            if(DEBUG):
                print("IMAGEM ESQUERDA {}".format(leftcount))
                cv2.imshow("LeftBin", imageLeft)
                cv2.waitKey(0)
            returnIMG = imageLeft

    return returnIMG

def printvalues(barras, escala, img_labeled):
    if(escala != None):
        for g in range(len(barras)):

            x, y, w, h = barras[g]
            datavalue = h/escala

            # = x//2
            #y = y//2
            #w = w//2
            #h = h//2

            font = cv2.FONT_HERSHEY_PLAIN
            #print('AQUI')

            cv2.putText(img_labeled, '{0:.2f}'.format(datavalue), (x, (y - 6)), font, 1.0, (98, 133, 98), 2, cv2.LINE_AA)
            cv2.rectangle(img_labeled,(x, y),(x + w,y + h),(0,255,0),2)
            cv2.imshow("VALOR", img_labeled)
            cv2.waitKey(0)

# ------- Helpíng Functions ------ #

def extract(sourceimg):

    imgFiltered, imgResized = filterImage(sourceimg)  # Filter IMG for OCR and Resize it for the rest
    imgFiltered = imgResized
    maskList = findBarMasks(sourceimg.copy())  # Find the masks of the bars
    maskContours = findMasksContours(maskList, sourceimg)  # Find the bar contours of the bars


    if (DEBUG):  # Draw the contours to see if everything is alright
        barsimg = sourceimg.copy()
        cv2.drawContours(barsimg, maskContours, -1, (0, 255, 0), 1)
        cv2.imshow("Contornos", barsimg)
        cv2.waitKey(0)

    orientation = inferOrientation(maskContours)  # Infer orientation, True if vertical, False if horizontal


    xAxis = getXAxis(maskContours, orientation, sourceimg.copy())  # Find the X Axis
    #printAXIS(xAxis, imgResized.copy())
    #scale, dict = runOCR(imgFiltered, xAxis)  #  Run OCR and return the scale
    #print("Scale", scale)

    barras = calulateBars(maskContours) #  Store the bars as a list of x, y, w, h. Example: The x of the first bar is in barras[0][0]
    sortedBarras = sorted(barras, key=itemgetter(0)) #  Left to right order
    barrasComZeros = calulateZeros(sortedBarras, xAxis, sourceimg.copy()) #  Added the zeros in the middle
    sortedBarras = sorted(barrasComZeros, key=itemgetter(0))  #  Left to right order with the zeros included

    if (DEBUG):
        print("NOVAS BARRAS", sortedBarras)

    #datavalues = calculateDataValues(sortedBarras, scale, dict['p'])
    #print("OCR STATUS:", dict['p'])
    imageForOCR = getScaleSegment(sourceimg.copy(), sortedBarras)
    scale, dict, ponto = runOCR(imageForOCR, sourceimg, xAxis)


    datavalues = calculateDataValues(sortedBarras, scale, dict['p'])



    if (DEBUG):
        print("ESCALA", scale)
        printvalues(sortedBarras, scale, sourceimg.copy())
    #print("DATAVALUES: ", datavalues)

    for g in range(len(sortedBarras)):  #Fix Scale, this will have to be adjusted to the scaling factor once the OCR is implemented correctly
        sortedBarras[g] = tuple(ti//1 for ti in sortedBarras[g])

    title, labelx, labely = getTitles(sourceimg.copy(), sortedBarras)

    return {'Barras': sortedBarras, 'xAxis': xAxis//1, 'OCRStatus': dict['p'], 'ponto': ponto, 'ValorBarras': datavalues,
            "titulo": title, "labelX": labelx, "labelY": labely }
