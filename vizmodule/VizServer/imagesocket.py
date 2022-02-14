from aiohttp import web
import socketio
#from socketIO_client_nexus import SocketIO, LoggingNamespace

import cv2 as cv, cv2
import numpy as np
import base64
import threading
import asyncio

# base64 decode
# https://stackoverflow.com/questions/40928205/python-opencv-image-to-byte-string-for-json-transfer
def gen64str(img):
    _, buffer = cv.imencode('.png', img)
    buffer = base64.b64encode(buffer)
    return str(buffer)[2:-1]

def genbytesfrom64(str):
    return base64.b64decode(str)

def genimgfrombyte(buff):
    buffer = np.asarray(bytearray(buff), dtype=np.uint8)
    return cv.imdecode(buffer, 1)

class ImageSocket:

    def __init__(self, operation, imagesave=False):
        self.sio = socketio.AsyncServer()
        self.app = web.Application()
        self.sio.attach(self.app)

        @self.sio.on('connect')
        def connect(sid, environ):
            print("connect ", sid)

        @self.sio.on('disconnect')
        def disconnect(sid):
            print('disconnect ', sid)

        self.assignEvent('image-save', operation)

    def assignEvent(self, name, operation):
        @self.sio.on(name)
        async def message(sid, data):
            img = genimgfrombyte(data['data'])

            newimg = operation(img)
            dictAns = {'img': gen64str(newimg)}

            await self.sio.emit('ans', dictAns)

    def run(self):
        def runwebapp(self):
            asyncio.set_event_loop(asyncio.new_event_loop())
            web.run_app(self.app, port=9191, shutdown_timeout=9999999999)

        t = threading.Thread(target=runwebapp, args=(self,))
        t.start()


    # async def index(request):
    #     """Serve the client-side application."""
    #     with open('index.html') as f:
    #         return web.Response(text=f.read(), content_type='text/html')
    #app.router.add_static('/static', 'static')
    #app.router.add_get('/', index)

if __name__ == '__main__':
    # normal function to define action, must have: ndarray return or boolean return
    def grayimg(img):
        return cv2.cvtColor(img, cv.COLOR_BGR2GRAY)

    imagesocket = ImageSocket(grayimg, True)
    imagesocket.run()
    print('running imagesocket server on port 9191')
