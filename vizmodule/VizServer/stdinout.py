import cv2 as cv
import sys
import numpy as np

if __name__ == '__main__':
    i = 0
    buffer = sys.stdin.readlines()
    buffer = '\n'.join(buffer)
    print(len(buffer))
    image = np.asarray(bytearray(buffer), dtype="uint8")

    img = cv.imdecode(buffer, 1) # encode or decode?
    print('shape ', img.shape)
    cv.imwrite('teststdinoutpython.png', img)
    print('ok')
