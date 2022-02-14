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
from vizmodule.VizServer.imagesocket import gen64str, genbytesfrom64, genimgfrombyte

#ocr.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'

DEBUG = False
DEBUGADVANCED = False

#-----------------------COMEÇO FUNÇÕES-----------------------#

#Calculate the scale in pixels/dataUnity
def calculateScale(numDistance, pixelDistance):

    scaleC = None
    if(pixelDistance > 0):
        scaleC = pixelDistance/numDistance  #NumDistance is the numeric distance as (bigger - smaller)

    return scaleC
    #else:
     #   pixelDistance = abs(umElemento - axisX)
      #  scaleC = pixelDistance / numDistance  # NumDistance is the numeric distance as (bigger - smaller)
       # return scaleC


# Calcula a distancia euclidiana entre duas cores
def euclidianDistance(cor1, cor2):


    #Self-explanatory, just euclidian distance formula
    distance = math.sqrt(pow((cor1[0] - cor2[0]), 2) +
                         pow((cor1[1] - cor2[1]), 2) +
                         pow((cor1[2] - cor2[2]), 2))
    return distance


def findOutbarColor(image):
    immg = image.copy()
    immg = cv2.cvtColor(immg, cv2.COLOR_BGR2RGB)

    immg = immg.reshape((immg.shape[0] * immg.shape[1], 3))  # represent as row*column,channel number
    clt = KMeans(n_clusters=8)  # cluster number
    clt.fit(immg)

    hist = find_histogram(clt)
    hist, centers = (list(t) for t in zip(*sorted(zip(hist, clt.cluster_centers_), reverse=True)))

    bar = plot_colors2(hist, centers)

    if DEBUG:
        plt.axis("off")
        plt.imshow(bar)
        plt.show()

    #print(len(clt.cluster_centers_))
    #colorRGB = np.zeros((1, 1, 3), np.uint8)
    #colorRGB[0][0]= clt.cluster_centers_[1]
    #corHSV = cv2.cvtColor(colorRGB, cv2.COLOR_BGR2HSV)
    #print(corHSV)
    #print(colorRGB)


    for k in range(len(clt.cluster_centers_)):
        colorRGB = np.zeros((1, 1, 3), np.uint8)
        colorRGB[0][0] = clt.cluster_centers_[k]
        corHSVim = cv2.cvtColor(colorRGB, cv2.COLOR_BGR2HSV)
        corHSV = corHSVim[0][0]

    return hist, centers

#Finds a histogram of the colors found in the function above, not used anymore either
def find_histogram(clt):
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()
    return hist


#Creates an image with percentages of the found colors in the function above, not used anymore
def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX


    # return the bar chart
    return bar


#Find the connected regions of the image that may be a bar
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


#Here we analyze the Masks and we get only the contours that we are interested in (bars)
def findMasksContours(maskList):
    # Iterate through masks list and save only the ones with the area at least 80% of the area of it's bounding box

    # Initialize a new array that will be used to store the contours on each mask
    maskContours = []

    # get image dimensions
    h1, w1 = image.shape[:2]
    #Iterate through masks
    for masks in maskList:

        #Get tge contours
        im2, contours, hierarchy = cv2.findContours(masks, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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

            # If the area of the contour is at leats 80% of the area of it's bouding box
            # The area of the bouding box is at least 50% smaller than the total area of the image
            # And if the contour isn't inside another contour (hierarchy [0, k, 3] = -1; k being the index of the contour)
            if areaCont >= (areaBox * 0.8) and areaBox < AreaImg * 0.5 and hierarchy[0, q, 3] == -1:
                #Add that contour to the list of bars
                maskContours.append(contours[q])

                if(DEBUGADVANCED):
                    cv2.drawContours(barsimgbefore, contours, q, (0, 255, 0), 1)
                    cv2.imshow("Contorno atual", barsimgbefore)
                    cv2.waitKey(0)
                    print("Hierarchy of this contour: {}".format(hierarchy[0, q, 3]))

    #print("Lenght of contours vector is {}".format(len(maskContours)))
    return maskContours


#Not used anymore, our bars are always found in the above function
def removeBackgroundBars(maskContours):
    recuo = 1
    # Distance for one color to another to consider they're the same
    RGBdistance = 4
    definitiveContours = []

    for contorno in maskContours:
        # Obtain the location and dimention of each contour
        x, y, w, h = cv2.boundingRect(contorno)

        # Extract only the independent bar to calculate it's dominant inner color
        crop_img = clean[y:y + h, x:x + w]

        # Get dominant color usind K-means method
        immg = crop_img.reshape((crop_img.shape[0] * crop_img.shape[1], 3))  # represent as row*column,channel number
        clt = KMeans(n_clusters=1)  # cluster number
        clt.fit(immg)
        corDominante = clt.cluster_centers_[0]

        # The dominand color is here
        # print("Cor dominate da barra {}".format(corDominante))

        outsidePointsThrsh = 10 #  number of acceptable outsite points of the same color of the inside of the bar

        variation = 10 #   Number of pixels to scan after the border of the rectangle + 1
        testPixels = []
        numPixelsIguais = 0

        # Find the variation pixels and store them in a list
        # in the order: upper[i], downer[i], lefter[i], righter[i], upper[i +1] ...

        for j in range(variation):

            upperMiddle = (int(x + w / 2), y - j - 2)  # Top pixel
            downerMiddle = (int(x + w / 2), y + h + j + 2)  # Bottom pixel
            lefterMiddle = (x - j - 2, int(y + h / 2))  # Left pixel
            righterMiddle = (x + w + j + 2, int(y + h / 2))  # Right pixel

            testPixels.insert((4*j), upperMiddle)
            testPixels.insert((4*j) + 1, downerMiddle)
            testPixels.insert((4*j) + 2, lefterMiddle)
            testPixels.insert((4*j) + 3, righterMiddle)



        # print("Size of new things thingy {}".format(len(testPixels)))
        # print(testPixels)

        for b in range(len(testPixels)):
            cv2.circle(imge, testPixels[b], 1, (0, 0, 255), -1)

            cor = limpa[testPixels[b][1], testPixels[b][0]]

            distancia = euclidianDistance(cor, corDominante)

            if(distancia < RGBdistance):
                numPixelsIguais += 1


        # print("Num de pontos iguais fora da barra {}".format(numPixelsIguais))

        if (numPixelsIguais < outsidePointsThrsh):
            definitiveContours.append(contorno)

    # cv2.imshow("Pontos escolhidos", imge)
    # cv2.waitKey(0)
    return definitiveContours


# Figure out the orientation of the bar
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
        topImage = np.zeros((1, len(topList)), np.uint8)
        bottomImage = np.zeros((1, len(bottomList)), np.uint8)


        #Add the values in the image-like vectors
        for h in range(len(topList)):
            topImage[0][h] = topList[h]

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
       # print("EIXO X EM {}".format(index1))
        return index1

    else:
       # print("EIXO X EM {}".format(index2))
        return index2

#Just pirinting the X axis
def printAXIS(xaxis, imageteste):

    cv2.line(imageteste, (0, xaxis), (446, xaxis), (255, 0, 0), 4)
    cv2.imshow("teste", img_labeled)
    cv2.waitKey(0)

def printvalues(contornosese, escala):
    for g in range(len(contornosese)):

        x, y, w, h = cv2.boundingRect(contornosese[g])
        datavalue = h/escala

        font = cv2.FONT_HERSHEY_PLAIN
        print('AQUI')

        cv2.putText(img_labeled, '{0:.1f}'.format(datavalue/10), (x, (y - 6)), font, 1.0, (98, 133, 98), 2, cv2.LINE_AA)
        cv2.drawContours(img_labeled, contornosese, g, (98, 133, 98), 3)
        cv2.imshow("VALOR", img_labeled)
        cv2.waitKey(0)

def calulateBars(contornosfinais):
    barras = []
    for p in range(len(contornosfinais)):
        barras.append(cv2.boundingRect(contornosfinais[p]))

    return barras

def calculateDataValues(sortedBarras, escala, OCRvalue):
    datavalues = []
    if(OCRvalue == 'total' and scale != None):
        for d in range(len(sortedBarras)):
            datavalues.append((sortedBarras[d][3])/escala)

    elif(OCRvalue == 'fault'):
        for d in range(len(sortedBarras)):
            datavalues.append((sortedBarras[d][3]))

    elif(OCRvalue == 'partial'):
        for d in range(len(sortedBarras)):
            datavalues.append((sortedBarras[d][3]) / escala)

    return datavalues

def generateJson(dataValues, sortedBarras, OCRstatus, filename):


    pythonDictionary = {'FileName': filename, 'values': dataValues, 'OCR': OCRstatus}

    dictionaryToJson = json.dumps(pythonDictionary)

    return dictionaryToJson, pythonDictionary

#OCR FUNCTIONS_________________________________________

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


def getallnameimagesfromdirectory(path, extension='', numimages=-1, colorspace=1):
    imgs = []
    dir_p = path
    filenames = [f for f in listdir(dir_p) if isfile(join(dir_p, f))]
    totalimagestoload = numimages if numimages != -1 else len(filenames)

    for f in filenames:
        if extension != '':
            if f[-4:] == extension:
                if (totalimagestoload != 0):
                    totalimagestoload -= 1
                    imgs.append(cv2.imread(dir_p + f, colorspace))
                else: break
        else:
            if (totalimagestoload != 0):
                totalimagestoload -= 1
                imgs.append(cv2.imread(dir_p + f, colorspace))
            else:
                break
    return imgs


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

def pyarrtojsonarr(vec):
    arr = '['
    for arr_i in range(len(vec)):
        arr += str(vec[arr_i])
        if arr_i != len(vec)-1:
            arr += ', '
    return arr + ']'


def augmentImage(imageaugmented, dictinfo):
    hist, centers = findOutbarColor(imageaugmented)

    sortedBarras, values, ocrstatus = dictinfo['Barras'], dictinfo['ValorBarras'], dictinfo['OCRStatus']

    avg_sum = 0
    startcircle = 0
    augmentedstring = '{\n\t\"chartdata\":[\n\t\t'
    for barindex in range(len(sortedBarras)):
        bar = sortedBarras[barindex]
        bardata = pyarrtojsonarr(bar)

        posy, posx = int(bar[0] + bar[2]/2), bar[1]
        medianPoint = '[' + str(posy) + ',' + str(posx) + ']'

        augmentedstring += '{\n\t\t\"bar\":' + bardata + ',\n\t\t'
        augmentedstring += '\"value\":' + str(values[barindex]) + ',\n\t\t'
        augmentedstring += '\"median\":' + medianPoint + '\n\t\t}'
        if barindex != len(sortedBarras) - 1:
            augmentedstring += ','
        # cv2.circle(imageaugmented, (posy, posx), 2, tendencycolor, 4)
        if startcircle != 0:
            s = 0
            # cv2.line(imageaugmented, startcircle, (posy, posx), tendencycolor, 2)
        startcircle = posy, posx
        avg_sum += posx
    avgy = int(avg_sum/len(sortedBarras))

    ctemp = centers[1]
    c = np.zeros((1, 1, 3), dtype=np.float)
    c[:, :] = ctemp

    ccomp = cv2.convertScaleAbs(c - (255, 255, 255))
    chsv = cv2.cvtColor(c.astype(np.uint8), cv2.COLOR_BGR2HSV)
    chsvT1, chsvT2 = chsv.copy(), chsv.copy()

    chsvT1[0, 0, 1], chsvT2[0, 0, 1] = 128, 128

    chsvT1[0, 0, 0] = abs(180 - chsvT1[0, 0, 0] + 60) % 180
    chsvT2[0, 0, 0] = abs(180 - chsvT2[0, 0, 0] - 60) % 180

    augmentedstring += '], \n\t\"average\": ' + str(avgy) + ",\n\t"
    augmentedstring += '\"color\": ' + pyarrtojsonarr(c.astype(np.uint8)[0, 0]) + ',\n\t'
    augmentedstring += '\"colorComp\": ' + pyarrtojsonarr(applycontrast(ccomp, 1.5, 0)[0, 0]) + ',\n\t'
    augmentedstring += '\"colorBright\": ' + pyarrtojsonarr(applycontrast(c, 2.1, 10)[0, 0]) + ',\n\t'
    augmentedstring += '\"colorDark\": ' + pyarrtojsonarr(applycontrast(c, 0.1, -10)[0, 0]) + ',\n\t'
    augmentedstring += '\"colorT1\": ' + pyarrtojsonarr(cv2.cvtColor(chsvT1, cv2.COLOR_HSV2RGB)[0, 0]) + ',\n\t'
    augmentedstring += '\"colorT2\": ' + pyarrtojsonarr(cv2.cvtColor(chsvT2, cv2.COLOR_HSV2RGB)[0, 0]) + ',\n\t'
    augmentedstring += '\"ocr\": ' + '\"' + ocrstatus + '\"' + ',\n\t'
    augmentedstring += '\"dimensions\": ' + pyarrtojsonarr(imageaugmented.shape[:2]) + "\n}"


    # send the color of the bars too!!!!!!!!

    # prefix = filename[:-4]
    # with open(pathaugmented + prefix + '.json', 'w+') as f:
    #     f.write(augmentedstring)

    b64str = firevent.fireAugment("".join(augmentedstring.split()).replace(" ", "").replace("\"", "\"\"\""), gen64str(imageaugmented))
    tempbytes = genbytesfrom64(b64str)
    return genimgfrombyte(tempbytes)
    # cv2.imshow('augmented', tempimg)
    # cv2.waitKey(0)

def calulateZeros(barrasordenadas):
    thresh = 5 #How much in pixels can the distance between two bars vary more than the mode and still be considered normal
    distancias = []
    for i in range((len(barrasordenadas) - 1)):
        distancias.append(barrasordenadas[i+1][0] - barrasordenadas[i][0])

    #print(distancias)

    moda = np.min(distancias)

    #print("MODA", moda)


    indexHole = []
    for j in range(len(distancias)):

        if(distancias[j] > moda + thresh):
            quantos = int(round(distancias[j]/moda)) - 1
            indexHole.append([j, quantos])


   # print("INDEX HOLES: ", indexHole)

    for k in range(len(indexHole)):
         for c in range(indexHole[k][1]):
            #cv2.circle(limpa, (barrasordenadas[indexHole[k][0]][0] + ((c + 1) * moda), xAxis), 1, (0, 0, 255), -1)
            sortedBarras.append((barrasordenadas[indexHole[k][0]][0] + ((c + 1) * moda), xAxis, barrasordenadas[0][2], 0))

    #print("novasBarras", barrasordenadas)
    #cv2.imshow("added zeros", limpa)
    #cv2.waitKey(0)

    return barrasordenadas

# -----------------------FIM FUNÇÕES----------------------- #

if __name__ == '__main__':

    pathJsons = "jsons/"
    pathImages = 'TESTE/'
    pathaugmented = 'augmented/'
    # listaImagens = getallnameimagesfromdirectory("TESTE/")
    listaNomes = getallnamesfromdirectory(pathImages)


    ocrTotal = 0
    for z in range(len(listaNomes))[:1]:
        print('IMAGEM ATUAL: ', z, listaNomes[z])  #Esse aqui fica fora do debug pq mostra qual a imagem atual e é importante saber sempre
        # Load image, from file for now to make things easy
        #imageRes = cv2.imread("SampleGraphs/bar75.png")
        #imageRes = listaImagens[z]
        imageRes = cv2.imread(join(pathImages, listaNomes[z]))
        imageaugmented = imageRes.copy()

        img_ocr = imageRes.copy()

        #Apply bilateral filter
        image = cv2.bilateralFilter(imageRes,3,100,100)

        #Create smne copies because some functions used modify the original image, we need MOOORE
        #imge = image.copy()
        clean = image.copy()
        limpa = image.copy()
        barsimg = image.copy()
        barsimgbefore = image.copy()
        testeimg = image.copy()
        testevalues = image.copy()

        image = imageRes
        imge = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        imge = cv2.cvtColor(imge, cv2.COLOR_BGR2GRAY)

        kernel = np.ones((1, 1), np.uint8)
        imge = cv2.dilate(imge, kernel, iterations=1)
        imge = cv2.erode(imge, kernel, iterations=1)

        gaussian_1 = cv2.GaussianBlur(imge, (1, 1), 10.0)
        imge = cv2.addWeighted(imge, 1.8, gaussian_1, -0.6, 0, imge)

        imge = cv2.threshold(imge, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        #cv2.imshow("Final Result", imge)
        #cv2.waitKey(0)


        #Find the masks in the image
        maskList = findBarMasks(image)

        #Find the contours in the masks
        maskContours = findMasksContours(maskList)  # Aqui volta os contornos das barras <----- AQUIIIII
                                                    # Em formato de contorno do opencv mesmo < ----- AQUIIII



        #Draw only for debug
        if(DEBUG):
            cv2.drawContours(barsimg, maskContours, -1, (0, 255, 0), 1)
            cv2.imshow("Contornos", barsimg)
            cv2.waitKey(0)


        #contornos = removeBackgroundBars(maskContours)
        #Infer the orientation
        orientation = inferOrientation(maskContours) #True if vertical, false if horizontal

        #infer x axis location
        # Printing x axis location
        xAxis = getXAxis(maskContours, orientation, testeimg)
        if(DEBUG):
            print('X Axis Location in pixels: ', xAxis)


        #OCR Functions
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

            #drawocrrects(img_labeled, words[['left', 'top', 'width', 'height']], (255, 0, 0))

            if(DEBUGADVANCED):
                drawocrrects(img_labeled, numbers[['left', 'top', 'width', 'height']], (0, 255, 0))

                displayimages([
                                 {'img': img_ocr, 'title': 'Original'},
                                 {'img': contrast, 'title': 'Contraste'},
                                 {'img': sharp, 'title': 'Destaque de Borda'},
                                 {'img': img_gray, 'title': 'Escala de Cinza'},
                                 {'img': img_laplace, 'title': 'Laplace'},
                                 {'img': thresh, 'title': 'Binarizacao'},
                                 {'img': w_sum, 'title': 'Soma Ponderada'},
                                 {'img': img_labeled, 'title': 'Marcada'}
                               ])


                cv2.imshow("LABELED", img_labeled)
                cv2.waitKey(0)


            p = np.chararray((words.shape[0]), 30)
            n = np.chararray((numbers.shape[0]), 30)

            for i in range(words.shape[0]):
                p[i] = words['text'].values[i].encode('utf-8')

            for i in range(numbers.shape[0]):
                n[i] = numbers['text'].values[i].encode('utf-8')

            lateralaxis = getaxis(numbers)
            dict = getnearestnumbers(lateralaxis, 229)
            v1, v2 = dict['v1'], dict['v2']
            isDilate = v1 == 0 and v2 == 0
            if not isDilate:
                ocrTotal += 1
                # print dict['v1'], dict['v2'], dict['h']
        # cv.imwrite('prints/labels/bar_'+str(nameindex)+'.png', img_labeled)
        # print '------ ' + str(nameindex + 1) + ' ------'
        #print (ocrTotal, '/', 100)

        print(dict['v1'], dict['v2'], dict['h'])

        numDif = abs((dict['v1'] - dict['v2']))
       # print(numDif)
       # print('DICTH', dict['h'])
       # print('DICTP', dict['p'])

        scale = None
        if(dict['p'] != 'fault' and numDif != 0):
            scale = calculateScale(numDif, dict['h'])
            #print('scale', scale)

        xAxis = getXAxis(maskContours, orientation, image)
        #printvalues(maskContours, scale)
        if(DEBUG):
            printAXIS(xAxis, barsimg)

        barras = calulateBars(maskContours) #  Store the bars as a list of x, y, w, h. Example: The x of the first bar is in barras[0][0]

        sortedBarras = sorted(barras, key=itemgetter(0))
        #print("barras",sortedBarras)
        barrasComZeros = calulateZeros(sortedBarras)
        sortedBarras = sorted(barrasComZeros, key=itemgetter(0)) #NOVAS BARRAS ORDENADAS CONTENDO OS ZEROS
        #print("Novas Barras: ", sortedBarras)


        #print('bars', sortedBarras)

        #Calculate the data values of the sorted bars using the calculated scale
        datavalues = calculateDataValues(sortedBarras, scale, dict['p'])
        #print('DATAVALUES', datavalues)

        filename = listaNomes[z]

        #augimg = augmentImage(imageaugmented, sortedBarras, datavalues)

        dictionaryToJson, dict = generateJson(datavalues, sortedBarras, dict['p'], filename)

        #print(dict)
        df = pd.DataFrame(dict)
        #print(df)
        df.to_json(join(pathJsons, filename + '.json'))

        #print(dictionaryToJson)
        #print(listaNomes[z])
        #cv2.imshow("testerder", testevalues)
        #cv2.waitKey(0)