# Needed imports
from sklearn.cluster import KMeans
import cv2
import scipy
import numpy as np
from time import sleep
import math
from matplotlib import pyplot as plt


# Load image, from file for now to make things easy
imageRes = cv2.imread("SampleGraphs/bar4.png")

# Normalize image
#r = 250.0/img.shape[0]
#dim = (int(img.shape[1] * r), 250)
# Height defined, width is proportionally adjusted
#imageRes = cv2.resize(img, dim, interpolation = cv2.INTER_NEAREST)
image = cv2.bilateralFilter(imageRes,3,100,100)
imge = image.copy()
clean = image.copy()
#cv2.imshow("ds", image)
#cv2.waitKey(0)
imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Create mask, mask list and test vector
h, w = image.shape[:2]
testIndex = np.zeros((h, w), np.uint8)
maskList = []
# mask = np.zeros((h+2, w+2), np.uint8)
# print(testIndex)

# Iterate through image and only test for blobs on pixels that are not True on test vector

for l in range(0, len(testIndex)):
    for c in range(0, len(testIndex[0])):
        if testIndex[l][c] != 255:  # Only if that pixel isn't yet on

            #print(c)
            # Reinitialise the mask so it will always check all the pixels
            mask = np.zeros((h+2, w+2), np.uint8)
            # Run floodFill centered in teh current (c, l) pixel
            lo = 180
            hi = 180

            cv2.floodFill(image, mask, (c, l), (255, 255, 255), (lo,)*3, (hi,)*3, 4 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE)
            # Add each new element found as a binary image in the maskList array
            maskList.append(mask)
            #cv2.imshow("mascara",mask)
            #cv2.waitKey(1)

            #See where the found mask is equal to white and set they as true on the test vector so we won't iterate through them
            a = np.where(mask == 255)
            # print(len(a[0]))
            for x in range(0, len(a[0])):
                col = a[0][x]
                lin = a[1][x]


                # Remember that each pixel (col, lin) on the image corresponde to a (col + 1, lin + 1) pixel in the image
                testIndex[col-1][lin-1] = 255

# Iterate through masks list and save only the ones with the area at least 90% of the area of it's bounding box

print(len(maskList))
# Initialize a new array that will be used to store the contours on each mask
newContours = []

h1, w1 = image.shape[:2]
for masks in maskList:

    # cv2.imshow("fundo", masks)
    im2, contours, hierarchy = cv2.findContours(masks, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    AreaImg = h1 * w1
    print("Tamanho do vetor de contornos {}".format(len(contours)))
    # Tentando desenhar os contornos
    newa = clean.copy()
    cv2.drawContours(newa, contours, -1, (0, 255, 0), 1)
    cv2.imshow("new", newa)
    cv2.waitKey(0)


    for q in range(len(contours)):

        areaCont = cv2.contourArea(contours[q])
        x, y, w, h = cv2.boundingRect(contours[q])
        areaBox = w * h

        if(w * h) > 200 and (w * h) < 0.5 * AreaImg and w > 2 and h > 2:
            newContours.append(contours[q])

        #print('Area do Contorno: {}'.format(areaCont))
        #print('Area da BoudingBox: {}'.format(areaBox))

        #if areaCont >= (areaBox * 0.9) and w > 2 and h > 2 and areaBox < AreaImg:
        #if areaCont >= (areaBox * 0.8) and areaBox < AreaImg * 0.5 and w > 2 and h > 2:
        #newContours.append(contours[q])


print("Tamanho do vetor novo {}".format(len(newContours)))


newa1 = clean.copy()
cv2.drawContours(newa1, newContours, -1, (0, 255, 0), 1)
cv2.imshow("new", newa1)
cv2.waitKey(0)


#
#newa1 = clean.copy()
#for k in range(len(newContours)):
#    cv2.drawContours(newa1, newContours, k, (0, 255, 0), 1)
#    cv2.imshow("new", newa1)
#    cv2.waitKey(0)





