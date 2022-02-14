import numpy as np
import cv2 as cv

import cnnmodule.maskutils as mk


def getcnt(img):
    _img, contours, _h = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    return contours


def getbbox(cnts):
    nXmin, nYmin = np.amin(cnts[0], axis=0)[0]
    nXmax, nYmax = np.amax(cnts[0], axis=0)[0]
    return (nXmin, nYmin, nXmax, nYmax)


def drawline(_img_, pts):
    img = _img_.copy()
    cv.line(img, pts[0], pts[1], (0, 255, 0), 2)
    return img


def pointstoline(shape, cnt):
    rows, cols = shape[:2]
    [vx, vy, x, y] = cv.fitLine(cnt, cv.DIST_L2, 0, 0.01, 0.01)
    lefty = int((-x * vy / vx) + y)
    righty = int(((cols - x) * vy / vx) + y)
    return righty, lefty
    # cv.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)
    # point to draw (cols-1,righty), (0,lefty)


def getanglefrompoints(pts):
    y = pts[0][0] - pts[0][1]
    x = pts[1][0] - pts[1][1]
    return np.degrees(np.arctan2(x, y))


def centerfromcontours(cn):
    avgpoints = np.average(cn[0], axis=0)
    return int(avgpoints[0][0]), int(avgpoints[0][1])  # retornando o centro dos contornos cX, cY


def getrotatedimage(img, center, angle, w=512, h=512):
    M = cv.getRotationMatrix2D(center, angle / 3.5, 1)
    return cv.warpAffine(img, M, (w, h))


def cntinfo(imgbin):
    cnt = getcnt(imgbin)
    mincol, minrow, maxcol, maxrow = getbbox(cnt)
    cX, cY = centerfromcontours(cnt)
    return cnt, mincol, minrow, maxcol, maxrow, cX, cY


def getmaskedobj(r, image):
    imgbin = mk.genbinimgfrommask(r['masks'])
    cnt, mincol, minrow, maxcol, maxrow, cX, cY = cntinfo(imgbin)

    righty, lefty = pointstoline(imgbin.shape, cnt[0])
    points = [(maxcol, lefty), (mincol, righty)]
    # points = [(mincol, lefty), (maxcol, righty)]
    angle = getanglefrompoints(points)

    rotatedimagebin = getrotatedimage(imgbin, (cX, cY), -angle)
    rotatedimageorig = getrotatedimage(image.astype(np.uint8), (cX, cY), -angle)

    cnt, mincol, minrow, maxcol, maxrow, cX, cY = cntinfo(rotatedimagebin)

    # dnr = rotatedimageorig.copy()
    # cv.rectangle(dnr, (mincol, minrow), (mincol + (maxcol - mincol), minrow + (maxrow - minrow)), (255,255,0), 4)

    # minchart = rotatedimageorig[minrow : minrow + (maxrow - minrow), mincol : mincol + (maxcol - mincol)]

    # imgd = drawline(imgbin, points)

    # chart = cv.convertScaleAbs(chart)
    # get_ax().imshow(rotatedimageorig, cmap='gray')
    return np.asarray([minrow, minrow + (maxrow - minrow), mincol, mincol + (maxcol - mincol)], dtype=np.int64)

def fft(img):
    dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    magnitude_spectrum = 20 * np.log(cv.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    return magnitude_spectrum


def getimagefreq(_img_, thresh=220):
    freqimg = fft(_img_)
    freqimg = cv.convertScaleAbs(freqimg)
    return np.where(freqimg > thresh, 255, 0).astype(np.uint8)


def getanglefromfreq(_img_):
    grey = cv.cvtColor(_img_, cv.COLOR_RGB2GRAY)
    nfreq = getimagefreq(grey)
    nfreq = cv.morphologyEx(nfreq, cv.MORPH_DILATE, np.ones((3, 3), np.uint8))
    colorfreq = cv.cvtColor(nfreq, cv.COLOR_GRAY2RGB)

    cnfreq = getcnt(nfreq)
    maxcnt = max(cnfreq, key=cv.contourArea)
    rightyfreq, leftyfreq = pointstoline(colorfreq.shape, maxcnt)
    mincolfreq, _minrow, maxcolfreq, _maxrow = getbbox([maxcnt])

    pointsfreq = [(maxcolfreq, leftyfreq), (mincolfreq, rightyfreq)]

    # mk.get_ax().imshow(cm.drawline(colorfreq, pointsfreq), cmap='gray')

    return getanglefrompoints(pointsfreq)


def getincolor(img, color=(0, 255, 0)):
    hsvimg = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    hsvimg[:, :, 2] = np.where(hsvimg[:, :, 2] < 220, hsvimg[:, :, 2], 220)
    return cv.cvtColor(hsvimg, cv.COLOR_HSV2RGB)


def cropbyanglepadding(img, pad):
    padprop = abs(int((0.003 * img.shape[1]) * pad / 2))
    return img.copy()[padprop:img.shape[0] - padprop, padprop:img.shape[1] - padprop]
