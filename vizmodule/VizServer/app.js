const web_server = require('express')();
const socket_server = require('http').Server(web_server);
const fs = require('fs');
const io = require('socket.io')(socket_server);
const path = require('path');
const main_path = __dirname + '/html';
const ChartGenerator = require('./html/src/charts/chartgen');
const im = require('./imutils');

const web_port = 3000;
const socket_port = 9090;

// Serving webpages
web_server.get('*', function (req, res) {
    if (req.url === '/'){
        res.sendFile(path.join(main_path + '/index.html'));
    }
    else {
        res.sendFile(path.join(main_path + req.url));
    }
});

web_server.listen(web_port, function () {
    console.log('Web Server listening on port: ' + web_port);
});

socket_server.listen(socket_port, function(){
    console.log('Socket Server listening on port: ' + socket_port);
});

let clientsocket = 0, stop = false;
io.on('connection', (socket) => {
    console.log('user connected');
    io.emit('broadcast', { msg: 'new user connected'});

    socket.on('image-save', (imagejson) => {
        im.saveImage(imagejson.img, "output_buffer.png");
    });

    socket.on('image-buffer', (imagejson) => {
        let imageraw = imagejson.img;
        im.sendImage(imageraw, (data) => {
            socket.emit('image-display', data);
        });
    });

    socket.on('video-display', (d) =>{
        if (d.host === 'web') {
            if (clientsocket === 0) {
                clientsocket = socket;
                im.startStream('False', 'True');
                console.log('clientsocket saved, starting camera stream....');
            } else {
                stop = true;
                clientsocket.emit('video-display', {img: 'NULL'});
            }
        } else {
            if (!stop && clientsocket !== 0 ) {
                clientsocket.emit('video-display', d);
            }
            else {
                socket.emit('video-stop');
                clientsocket = 0;
                stop = false;
            }
        }
    });
});

// let data = [
//         {"key": "Apples", "value": 9},
//         {"key": "Oranges", "value": 3},
//         {"key": "Grapes", "value": 5},
//         {"key": "Bananas", "value": 7}
// ];
// let chartgen = new ChartGenerator('barchartVertical', data, 'Bar Chart');
// chartgen.generateChart();