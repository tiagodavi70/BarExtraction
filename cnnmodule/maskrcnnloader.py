import matplotlib.pyplot as plt
import time
import numpy as np
import cv2 as cv
import os
import sys

#os.environ['PATH'] += ";C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0\\bin\\"

sys.path.append("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0\\bin\\")
sys.path.append("C:\\Users\\bianc\\AppData\\Local\\Programs\\Python\\Python35\\Lib\\site-packages\\")


import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.config import Config

import cnnmodule.maskutils as mk
import cnnmodule.maskgeom as mg

class ChartsConfig(Config):
    """Configuration for training on the charts dataset.
    Derives from the base Config class and overrides values specific
    to the charts dataset.
    """
    # Give the configuration a recognizable name
    NAME = "charts"

    # Train on 1 GPU and 2 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 2 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1 # + 1

    BACKBONE = "resnet50"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 chart

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (128, 256)  # anchor side in pixels
    RPN_ANCHOR_RATIOS = [.5, 1, 2]

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 20

    MAX_GT_INSTANCES = 1
    DETECTION_MAX_INSTANCES = 1

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

class InferenceConfig(ChartsConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1
    IMAGE_RESIZE_MODE = "square"

def loadmodelandweights():
    # config = ChartsConfig()
    # config.display()

    MODEL_DIR = './cnnmodule/maskmodel'
    inference_config = InferenceConfig()

    model_path = os.path.join(MODEL_DIR, "mask_rcnn_chartTopLayers.h5")
    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=MODEL_DIR)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
    # model_path = model.find_last()[1]

    # Load trained weights (fill in path to trained weights here)
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    return model


def getobjimg(_img_):
    npimage = _img_
    image, padding = mk.resizeAndPad(npimage, (512, 512))

    image = np.asarray(image, dtype="int32")
    model = loadmodelandweights()
    t1 = time.time()
    results = model.detect([image], verbose=1)
    print('Detection time is {}'.format(time.time() - t1))

    r = results[0]
    # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
    #                             ['BG', 'chart'], r['scores'], ax=mk.get_ax())

    coords = mg.getmaskedobj(r, image)
    newcoords = mk.adjustcoords(coords, mk.scalefactor(npimage, image, padding), padding[0])

    chart = npimage[newcoords[0]:newcoords[1], newcoords[2]:newcoords[3]]
    minichart = image[coords[0]:coords[1], coords[2]:coords[3]].astype(np.uint8)

    # get_ax().imshow(minichart, cmap='gray')
    # minichartgray = cv.cvtColor(minichart, cv.COLOR_RGB2GRAY)

    topbotpad = minichart.shape[0] // 5
    leftrigthpad = minichart.shape[1] // 5
    centerminichart = minichart[topbotpad: minichart.shape[0] - topbotpad,
                      leftrigthpad: minichart.shape[1] - leftrigthpad]

    # centerminichart = cv.GaussianBlur(centerminichart, (3,3),0)
    # centerminichart = getincolor(centerminichart)

    anglefreq = mg.getanglefromfreq(centerminichart)
    # print(anglefreq)

    rotatedimagebin = mg.getrotatedimage(chart, (chart.shape[0] // 2, chart.shape[1] // 2), anglefreq, w=chart.shape[1],
                                         h=chart.shape[0])

    return mg.cropbyanglepadding(rotatedimagebin, anglefreq)

if __name__ == '__main__':

    #TODO: Criar uma classe e encapsular os métodos, reaproveitando as variáveis, como classe e talz

    img = cv.imread('../dataset_docscharts/bar52.png', 1)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    objimg = getobjimg(img)
    mk.get_ax().imshow(objimg, cmap='gray')
    plt.show()