document.querySelector('input').addEventListener('change', function() {
    let reader = new FileReader();
    reader.onload = function() {
        // let arrayBuffer = this.result;
        // let array = new Uint8Array(arrayBuffer);
        // let binaryString = String.fromCharCode.apply(null, array);

        socket.emit("image-buffer", {"img": this.result});
    };
    reader.readAsArrayBuffer(this.files[0]);
});

let cambutton = d3.select("#camera");
cambutton.on("click", () => {
    let txt = cambutton.text() === 'Start Camera Stream'? 'Stop Camera Stream' : 'Start Camera Stream';
    cambutton.text(txt);
    socket.emit('video-display', {host:'web'});
});
