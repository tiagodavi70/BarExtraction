import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

def get_ax(rows=1, cols=1, size=4):
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

def resizeAndPad(img, size, padColor=0):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv.INTER_AREA
    else: # stretching image
        interp = cv.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv.BORDER_CONSTANT, value=padColor)

    return scaled_img, [pad_left, pad_right, pad_top, pad_bot]

def genbinimgfrommask(mk):
    binmask = np.where(mk, 255, 0)
    #print(binmask.shape)
    return binmask.astype(np.uint8)


def scalefactor(orgimg, scaled, pad):
    icrop = orgimg[:, pad[0]:orgimg.shape[0] - pad[1]]
    return icrop.shape[0] / scaled.shape[0]


def adjustcoords(coords, scalefactor, pad):
    c = np.zeros(4, dtype=np.int)
    # print(coords[2] - pad, int((coords[3] - pad) * scalefactor))

    c[0], c[1] = coords[0], coords[1]
    c[2], c[3] = coords[2] - pad, (coords[3] - pad)

    return (c * scalefactor).astype(int)


