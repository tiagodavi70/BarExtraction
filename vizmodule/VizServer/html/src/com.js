const socket = io('http://localhost:9090');

socket.on('image-display', (data) => {
    d3.select('#uploadPreview').attr('src', "data:image/png;base64," + data.img);
});

socket.on('video-display', (d) => {
    if (d.img !== 'NULL')
        d3.select('#uploadPreview').attr('src', "data:image/png;base64," + d.img);
    else
        d3.select('#uploadPreview').attr('src', "");
});