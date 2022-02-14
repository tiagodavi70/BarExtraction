(function() {

    const isNodeJS = typeof module !== 'undefined' && typeof module.exports !== 'undefined';
    const htmlpre = "<!DOCTYPE html>\n" +
        "<html lang=\"en\">\n" +
        "<head>\n" +
        "    <meta charset=\"UTF-8\">\n" +
        "    <title>" + "Visualization" + " </title>\n" +
        "</head>\n" +
        "<body>";
    const htmlpos = "</body>\n" +
        "</html>";

    this.colors = [ '#a6cee3','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f',
        '#ff7f00','#cab2d6','#1f78b4','#6a3d9a','#ffff99','#b15928',
        '#8dd3c7','#d9d9d9','#ffffb3'];

    if (isNodeJS) {
        this.d3 = require('d3');
        this._ = require('underscore');
        this.vega = require("vega");
    }

    function loadAllData() {

    }

    class ChartGenerator {

        constructor(chartType, dataExternal, selector){
            this.data = dataExternal;
            this.chartType = chartType;
            this.selector = selector;
            this.path = isNodeJS ? "./html/src/charts/vegadata/" : "./src/charts/vegadata/";
            this.outputpath = "./output/";

            if (isNodeJS){
                const fs = require('fs');
                this.render = function(spec) {
                    let view = new vega.View(vega.parse(spec))
                        .renderer('canvas')        // set renderer (canvas or svg)
                        .initialize();
                    Promise.all([view.toSVG(), view.toCanvas()])
                        .then(([svg,canvas]) => {
                            let htmltext = htmlpre + svg + htmlpos;
                            let prefix = this.outputpath + this.selector;
                            if (!fs.existsSync(prefix))
                                fs.mkdirSync(prefix);
                            prefix += '/' + this.chartType;

                            fs.writeFileSync(prefix + '.png', canvas.toBuffer());
                            fs.writeFileSync(prefix + '.svg', svg);
                            fs.writeFileSync(prefix + '.html', htmltext);
                        }).catch((err) => {console.error(err);});
                };
            } else {
                this.render = (spec) => {
                    return new vega.View(vega.parse(spec))
                        .renderer('svg')            // set renderer (canvas or svg)
                        .initialize(this.selector)  // initialize view within parent DOM container
                        .hover()                    // enable hover encode set processing
                        .run();
                };
            }
        };

        generateChart() {
            vega.loader()
                .load(this.path + this.chartType + ".json")
                .then((dataraw) => {
                    let spec = JSON.parse(dataraw);
                    spec.data[0].values = this.data;
                    let view = this.render(spec);
                });
        };

        static debug() {
            return 'debug ';
        }
    }

    if (isNodeJS)
        module.exports = ChartGenerator;
    else
        window.ChartGenerator = ChartGenerator;
})();