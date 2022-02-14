import numpy as np
import cv2

image = cv2.imread("TESTE/bar1089.png")
img = cv2.resize(image, None, fx=1.5, fy=2, interpolation=cv2.INTER_NEAREST)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

kernel = np.ones((2, 2), np.uint8)
img = cv2.dilate(img, kernel, iterations=1)
#img = cv2.erode(img, kernel, iterations=1)


gaussian_1 = cv2.GaussianBlur(img, (1,1), 10.0)
img = cv2.addWeighted(img, 1.8, gaussian_1, -0.6, 0, img)


img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
cv2.imshow("Final Result", img)
cv2.waitKey(0)




