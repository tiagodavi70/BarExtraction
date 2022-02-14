
let args = process.argv.slice(2);
let port = args[0];
let type = args[1];
let msg = args[2];

const socket = require('socket.io-client')('http://localhost:'+port);

socket.on('connect',  (d) => {
    socket.emit(type, {'data': msg});
    socket.on('ans', (d) => {
        console.log("msg sent");
        process.exit();
    });
});