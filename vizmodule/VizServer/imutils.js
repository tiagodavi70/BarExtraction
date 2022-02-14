require('canvas-5-polyfill');
const { Image } = require('canvas');
const Canvas = require('canvas');
const fs = require('fs');
const _ = require('underscore');


module.exports.saveImage = (buffer, filename) => {
    let img = new Image;
    img.src = buffer;

    const canvas = new Canvas(img.width, img.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, img.width, img.height);
    canvas.pngStream().pipe(fs.createWriteStream(filename));
};

module.exports.sendImage = (buffer, cb) => {
    sendmsgclientsocket('9191', 'image-save', buffer, cb)
};

module.exports.startStream = (display, sendmsg) => {
    let spawn = require('child_process').spawn,
        py    = spawn('python', ['camerastream.py', display, sendmsg]),
        dataString = '';

    py.stdout.on('data', function(data){
        dataString += data.toString();
    });
    py.stdout.on('end', function(){
        console.log('', dataString);
    });

    py.stderr.on('data', (data) => {
      console.error(`child stderr: \n${data}`);
    });
};

function sendmsgclientsocket(port, type, msg, cb){
    const socket = require('socket.io-client')('http://localhost:'+port);
    socket.on('connect', (d) => {

        socket.emit(type, {'data': msg});
        socket.on('ans', cb);
    });
}