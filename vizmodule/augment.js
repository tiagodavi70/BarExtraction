const fs = require('fs');
const _ = require('underscore');
const canvasModule = require('canvas');
const D3Node = require('d3-node');
require('canvas-5-polyfill');
const { Image } = require('canvas');

function drawCircle(ctx, color, c){
    let tempStyle = ctx.fillStyle;
    let circle = new Path2D();
    circle.moveTo(0, 0);
    circle.arc(c[0], c[1], 6, 0, 2*Math.PI);
    ctx.fillStyle = color;
    ctx.fill(circle);
    ctx.fillStyle = tempStyle;
}

function drawLine(ctx, color, start, end){
    let tempstroke = ctx.strokeStyle;
    let pathLine = new Path2D();
    pathLine.moveTo(start[0], start[1]);
    pathLine.lineTo(end[0], end[1]);
    ctx.lineWidth = 3;
    ctx.strokeStyle = color;
    ctx.stroke(pathLine);
    ctx.strokeStyle = tempstroke;
}

function drawLegend(ctx, offset, avgColor, minColor, maxColor, tendlineColor, circleColor) {
    let startWidth = ctx.canvas.width - offset + 1;
    let widthSep = offset/10;
    let endWidthDraw = startWidth + 3*widthSep - 1;

    let height = ctx.canvas.height;
    let heightSep = (height/15);
    ctx.font = heightSep*.85 + 'px serif';

    let textConst = heightSep + heightSep/3.5;
    let heightConst = 2.25;

    drawLine(ctx, avgColor, [startWidth, heightSep], [endWidthDraw, heightSep] );
    ctx.fillText('Média', endWidthDraw + 10, textConst );

    drawLine(ctx, tendlineColor, [startWidth, heightConst*heightSep], [endWidthDraw, heightConst*heightSep] );
    drawCircle(ctx, circleColor, [startWidth + 1.5*widthSep, heightConst*heightSep]);
    ctx.fillText('Tendência', endWidthDraw + 10, 2*textConst );

    drawStrokeRect(ctx, minColor, [startWidth, 1.5*heightConst*heightSep, endWidthDraw - startWidth, 2*heightSep] );
    ctx.fillText('Máximo e', endWidthDraw + 10, 3*textConst);
    ctx.fillText('Mínimo', endWidthDraw + 10, 4*textConst);

    //drawRect(ctx, avgColor, [startWidth, heightSep], [endWidthDraw, heightSep] );
    //ctx.fillText('Máximo', endWidthDraw + 1, 3*heightSep + 3*heightSep/3 );
}

function convertColor(rgb,a){
    return `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${a})`
}

function drawStrokeRect(ctx, color, rectData){
    let rect = new Path2D();
    rect.moveTo(0, 0);
    rect.rect(rectData[0], rectData[1], rectData[2], rectData[3]);

    ctx.strokeStyle = color;
    ctx.stroke(rect);
}

function drawFillRect(ctx, color, rectData){
    let rect = new Path2D();
    rect.moveTo(0, 0);
    rect.rect(rectData[0], rectData[1], rectData[2], rectData[3]);

    ctx.fillStyle = color;
    ctx.fill(rect);
}

function main() {

    let arguments = process.argv.slice(2);
    const d3n = new D3Node({canvasModule});

    let chartinfo = JSON.parse(arguments[0]); //JSON.parse(fs.readFileSync(augmentedpath + chartname + '.json', "utf8"));
    let drawdata = chartinfo.chartdata;

    const legendOffset = chartinfo.dimensions[1]/4;
    const canvas = d3n.createCanvas(chartinfo.dimensions[1] + legendOffset, chartinfo.dimensions[0]); // padding for legend
    const ctx = canvas.getContext('2d');

    let img = new Image;

    img.onload = () => {

        drawFillRect(ctx, convertColor([250,250,250], 1), [0, 0, chartinfo.dimensions[1] + legendOffset, chartinfo.dimensions[0]]); // min fill

        ctx.drawImage(img, 0, 0, img.width, img.height);

        let avgColor = convertColor([0,0,255], .6);//convertColor(chartinfo.colorComp,0.9);
        let minColor = convertColor([255,0,0], 1);//convertColor(chartinfo.colorBright,0.8);
        let maxColor = convertColor([255,0,0], 1);//convertColor(chartinfo.colorDark,0.8);
        let tendLineColor = convertColor([50,200,50], .4); //convertColor(chartinfo.colorT1, 0.8);
        let circleColor = convertColor([0,200,0], 1);// convertColor(chartinfo.colorT1, 0.8);

        let avg = chartinfo.average;
        let heightSort = _.sortBy(drawdata, d => d.bar[3]);
        let positionSort = _.sortBy(drawdata, d => d.bar[0]);
        let minPos = _.pluck(positionSort, 'bar')[0][0];
        let maxPos = _.pluck(positionSort, 'bar')[drawdata.length - 1][0];
        let barWidth = drawdata[0].bar[2];

        drawStrokeRect(ctx, minColor, _.pluck(heightSort, 'bar')[0]); // min fill
        drawStrokeRect(ctx, maxColor, _.pluck(heightSort, 'bar')[drawdata.length - 1]); // max fill
        drawLine(ctx, avgColor, [minPos - 15, avg], [maxPos + barWidth + 15, avg]); // avgline
        ctx.fillStyle = avgColor;

        //console.log(drawdata);
        let arrvalues = _.pluck(drawdata, 'value');
        let trueavg = arrvalues.reduce( ( p, c ) => p + c, 0 ) / arrvalues.length;
        ctx.fillText(trueavg.toFixed(2), ((maxPos + barWidth + 15) - minPos - 15)/2, avg - 10 ); //avg valor
        // AUMENTAR O TOPO VIADO, TÁ PERDENDO INFORMAÇÃO

        drawLegend(ctx, legendOffset, avgColor, minColor, maxColor, tendLineColor, circleColor);

        let oldCirclePoint = drawdata[0].median;
        let min = 100000, max = -100000;

        function drawOnBars(ctx, tendLineColor, circleColor, startLinePoint, circlePoint, barValue, ocrstat) {
            let oldfillStyle = ctx.fillStyle;

            ctx.fillStyle = ocrstat === 'total'? "#00dd00" : (ocrstat === 'partial' ? "#fff000" :"#ff0000") ;

            ctx.fillText(barValue.toFixed(2), circlePoint[0] - ctx.canvas.width/35,
                circlePoint[1] - ctx.canvas.height / 25); // ctx.font defined on drawLegend, I am hoping it
            canvas.fillStyle = oldfillStyle;

            drawLine(ctx, tendLineColor, startLinePoint, circlePoint);
            drawCircle(ctx, circleColor, circlePoint);
        }

        for (let i = 0; i < drawdata.length; i++) {
            let circlePoint = drawdata[i].median;
            let barValue = drawdata[i].value;
            drawOnBars(ctx, tendLineColor, circleColor, oldCirclePoint, circlePoint, barValue, chartinfo.ocr);
            oldCirclePoint = circlePoint;
        }
        //canvas.pngStream().pipe(fs.createWriteStream(augmentedpath + '00_buffer' + '.png'));
        console.log(canvas.toDataURL());
    };

    process.stdin.resume();
    process.stdin.setEncoding('utf8');

    process.stdin.on('data', function(chunk) {
        if (!_.isUndefined(chunk)) {
            //let buf = Buffer.from(chunk, 'base64');
            //console.log(buf.toString('base64'));
            img.src = "data:image/png;base64," + chunk;//buf.toString();
        }
    });
    //let buffer = Buffer.from(chartimage, 'base64');//.toString('binary');
}
main();
