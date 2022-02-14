import cv2
import glob
cv_img = []
path = 'Binary'

for img in glob.glob("Binary/*.png"):
    n = cv2.imread(img)
    cv_img.append(n)

for x in range(len(cv_img)):
    gray_image = cv2.cvtColor(cv_img[x], cv2.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite(str(path) + 'bin{}.jpg'.format(x), im_bw)
    cv2.waitKey(0)

