# Needed imports
from sklearn.cluster import KMeans
import cv2
import scipy
import numpy as np
from time import sleep
import math
from matplotlib import pyplot as plt


# Load image, from file for now to make things easy
imageRes = cv2.imread("SampleGraphs/bar0.png")

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
testIndex = np.zeros((h, w), np.bool)
maskList = []
# mask = np.zeros((h+2, w+2), np.uint8)
# print(testIndex)

# Iterate through image and only test for blobs on pixels that are not True on test vector

for l in range(0, len(testIndex)):
    for c in range(0, len(testIndex[0])):
        if not testIndex[l][c]:  # Only if that pixel isn't yet on
            #print(c)
            # Reinitialise the mask so it will always check all the pixels
            mask = np.zeros((h+2, w+2), np.uint8)
            # Run floodFill centered in the current (c, l) pixel
            lo = 20
            hi = 20

            cv2.floodFill(image, mask, (c, l), (255, 255, 255), (lo,)*3, (hi,)*3, 8 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE)
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
                testIndex[col-1][lin-1] = True


# Iterate through masks list and save only the ones with the area at least 90% of the area of it's bounding box

print(len(maskList))
# Initialize a new array that will be used to store the contours on each mask
newContours = []

h1, w1 = image.shape[:2]
for masks in maskList:

    # cv2.imshow("fundo", masks)
    im2, contours, hierarchy = cv2.findContours(masks, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    AreaImg = h1 * w1


    for q in range(len(contours)):

        areaCont = cv2.contourArea(contours[q])
        x, y, w, h = cv2.boundingRect(contours[q])
        areaBox = w * h

        #print('Area do Contorno: {}'.format(areaCont))
        #print('Area da BoudingBox: {}'.format(areaBox))

        #if areaCont >= (areaBox * 0.9) and w > 2 and h > 2 and areaBox < AreaImg:
        if areaCont >= (areaBox * 0.7) and areaBox < AreaImg * 0.5 and hierarchy[0, q, 3] == 1:
            newContours.append(contours[q])



    #cv2.drawContours(mask, contours, -1, (0, 255, 0), 1)
    #cv2.imshow("oi", mask)
    #cv2.waitKey(0)


fundo = np.zeros((h1 + 2, w1 + 2), np.uint8)
print('Quantidade de novos contornos: {}'.format(len(newContours)))
#for x in range (len(newContours)):
#    fundo[:] = 255
#    cv2.drawContours(imge, newContours, x, (0, 255, 0), 1)
#    cv2.imshow("oi", imge)
#    cv2.waitKey(0)

cv2.drawContours(imge, newContours, -1, (0, 255, 0), 1)
cv2.imshow("oi", imge)
cv2.waitKey(0)

# Distance that the check pixel will be from the bar edges
recuo = 3
# Distance for one color to another to consider they're the same
RGBdistance = 50
definitiveContours = []

for contorno in newContours:
    # Obtain the location and dimention of each contour
    x, y, w, h = cv2.boundingRect(contorno)

    # Define each of the check pixel location
    upperMiddle = (int(x + w/2), y - recuo)
    downerMiddle = (int(x + w/2), y + h + recuo)
    lefterMiddle = (x - recuo, int(y + h/2))
    righterMiddle = (x + w + recuo, int(y + h/2))

    # Draw the place of the pixel, only for debug
    cv2.circle(imge, upperMiddle, 1, (0, 0, 255), -1)
    cv2.circle(imge, downerMiddle, 1, (0, 0, 255), -1)
    cv2.circle(imge, lefterMiddle, 1, (0, 0, 255), -1)
    cv2.circle(imge, righterMiddle, 1, (0, 0, 255), -1)

    # Find the color of one pixel inside the bar
    crop_img = clean[y:y + h, x:x + w]
    #cv2.imshow("clean", crop_img)
    #cv2.waitKey(0)

    immg = crop_img.reshape((crop_img.shape[0] * crop_img.shape[1], 3))  # represent as row*column,channel number
    clt = KMeans(n_clusters=1)  # cluster number
    clt.fit(immg)

    # The dominand color is here
    print("Cor dominate da barra {}".format(clt.cluster_centers_[0]))


    # Check if each of the check pixels (quite bad grammar, right?) is of the same color of the pixel chosen in last
    # step
    color1 = clean[upperMiddle[1], upperMiddle[0]]
    color2 = clean[downerMiddle[1], downerMiddle[0]]
    color3 = clean[lefterMiddle[1], lefterMiddle[0]]
    color4 = clean[upperMiddle[1], upperMiddle[0]]

    print(color1)
    print(color2)
    print(color3)
    print(color4)

    d1 = math.sqrt(pow((color1[0] - clt.cluster_centers_[0][0]), 2) +
                   pow((color1[1] - clt.cluster_centers_[0][1]), 2) +
                   pow((color1[2] - clt.cluster_centers_[0][2]), 2))
    d2 = math.sqrt(pow((color2[0] - clt.cluster_centers_[0][0]), 2) +
                   pow((color2[1] - clt.cluster_centers_[0][1]), 2) +
                   pow((color2[2] - clt.cluster_centers_[0][2]), 2))
    d3 = math.sqrt(pow((color3[0] - clt.cluster_centers_[0][0]), 2) +
                   pow((color3[1] - clt.cluster_centers_[0][1]), 2) +
                   pow((color3[2] - clt.cluster_centers_[0][2]), 2))
    d4 = math.sqrt(pow((color4[0] - clt.cluster_centers_[0][0]), 2) +
                   pow((color4[1] - clt.cluster_centers_[0][1]), 2) +
                   pow((color4[2] - clt.cluster_centers_[0][2]), 2))

    print("Distancia1 {}".format(d1))
    print("Distancia2 {}".format(d2))
    print("Distancia3 {}".format(d3))
    print("Distancia4 {}".format(d4))

    if(d1 > RGBdistance and d2 > RGBdistance and d3 > RGBdistance and d4 > RGBdistance):
        definitiveContours.append(contorno)

widthlist = []
heightlist = []

for contornins in definitiveContours:
    x, y, w, h = cv2.boundingRect(contornins)
    widthlist.append(w)
    heightlist.append(h)

print(widthlist)
print(heightlist)
a = np.zeros((1, len(widthlist)), np.uint8)
b = np.zeros((1, len(heightlist)), np.uint8)
for x in range(len(widthlist)):
    a[0][x] = widthlist[x]

for x in range(len(heightlist)):
    b[0][x] = heightlist[x]


hist1 = cv2.calcHist([a], [0], None, [100], [0, 256])
hist2 = cv2.calcHist([b], [0], None, [100], [0, 256])
#hist = cv2.calcHist(heightlist, [0], None, [256], [0, 256])

plt.figure()
plt.title("Histograma")
plt.xlabel("Bins")
plt.ylabel("Nº de Barras")
plt.plot(hist1)
plt.plot(hist2)
plt.xlim([0, 100])
plt.show()
cv2.waitKey(0)

#print(hist1)
#print(hist2)


bigger1 = 0
bigger2 = 0

for indexes in hist1:
    if indexes[0] > bigger1:
        bigger1 = indexes[0]

for indexes in hist2:
    if indexes[0] > bigger2:
        bigger2 = indexes[0]

print("Maior da largura {}".format(bigger1))
print("Maior da altura {}".format(bigger2))

if bigger1 > bigger2:
    print("GRAFICO VERTICAL")
else:
    print("GRAFICO HORIZONTAL")





cv2.imshow("imagem", imge)
cv2.waitKey(0)

cv2.drawContours(clean, definitiveContours, -1, (0, 255, 0), 1)
cv2.imshow("new", clean)
cv2.waitKey(0)

