import ExtractingTool as ET
from BarExtraction import augmentImage
import numpy as np
from vizmodule.VizServer.imagesocket import ImageSocket

#import cnnmodule.maskrcnnloader as maskloader

#cache memory to compare images and retrieve json info
#TODO: module cache

cache = []

def barExtract(docchartimg):
    print('checking cache')
    indexcache = -1
    for i in range(len(cache)):
        if np.array_equal(docchartimg, cache[i]['img']):
            indexcache = i
            break

    if indexcache >= 0:
        print('on cache')
        barinfo = cache[indexcache]['barinfo']
    else:
        print('extracting')
        #chartimg = maskloader.getobjimg(docchartimg)
        barinfo = ET.extract(docchartimg)
        cache.append({'img': docchartimg, 'barinfo': barinfo})

    print('augmenting: ', barinfo)
    return augmentImage(docchartimg, barinfo)


if __name__ == '__main__':
    # teste = cv.imread('dataset_final/bar51.png', 1)
    # showimg = barExtract(teste)
    # cv.imshow('augment', showimg)
    # cv.waitKey(0)

    # FOR SOME FUCKING REASON PYCHARM DO NOT RECOGNIZE MRCNN AS A PACKAGE
    imsock = ImageSocket(barExtract)
    imsock.run()
