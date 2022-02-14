# Needed imports
import cv2
import numpy as np


# Load image, from file for now to make things easy
img = cv2.imread("SampleGraphs/graph1.png")

# Normalize image
r = 360.0/img.shape[0]
dim = (int(img.shape[1] * r), 360)
# Height of 360p, width is proportionally adjusted
image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

cv2.waitKey(0)
imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Create mask, mask list and test vector
h, w = image.shape[:2]
testIndex = np.zeros((h, w), np.bool)
maskList = []
mask = np.zeros((h+2, w+2), np.uint8)

# Iterate through image and only test for blobs on pixels that are not 255 on test vector
print(len(testIndex[0]))

for l in range(0, len(testIndex)):
    for c in range(0, len(testIndex[0])):
        print(l)
        print(c)




            #cv2.imshow("mascara", mask)
            #cv2.waitKey(1)


for masks in maskList:
    cv2.imshow("Mascaras", masks)
    cv2.waitKey(0)