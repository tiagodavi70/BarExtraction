import numpy as np
import cv2
from imagesocket import gen64str
import sys
from socketIO_client_nexus import SocketIO, LoggingNamespace
import threading

class CameraStreamSocket:

    def __init__(self, displayStream, sendmsg, op):
        self.displaystream = displayStream
        self.sendmsg = sendmsg
        self.imageoperation = op
        self.stopcamera = False

    def stopcam(*args):
        global stopcamera
        stopcamera = True

    def run(self):
        def camerarun(self):
            socketIO = 0
            if sendMSG:
                socketIO = SocketIO('127.0.0.1', '9090', LoggingNamespace)
                socketIO.on('video-stop', self.stopcam)

            cap = cv2.VideoCapture(0)

            while True:
                # Capture frame-by-frame
                ret, frame = cap.read()

                # Our operations on the frame come here
                p_frame = self.imageoperation(frame)

                # Display the resulting frame
                if displayStream:
                    cv2.imshow('frame', p_frame)

                if sendMSG:
                    socketIO.emit('video-display', {'host': 'camera', 'img': gen64str(p_frame)})
                    socketIO.wait(.0001)

                if (cv2.waitKey(1) & 0xFF == ord('q')) | self.stopcamera:
                    break

            # When everything done, release the capture
            cap.release()
            cv2.destroyAllWindows()
            print('Python: Stream Finished')

        t = threading.Thread(target=camerarun, args=(self,))
        t.start()

if __name__ == '__main__':

    # normal function to define action, must have a ndarray return
    def grayimg(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.flip(gray, 1)

    displayStream = sys.argv[1] == 'True'
    sendMSG = sys.argv[2] == 'True'

    camera = CameraStreamSocket(displayStream, sendMSG, grayimg)
    camera.run()
    print('running')