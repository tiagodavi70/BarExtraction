from flask import Flask
app = Flask(__name__)
import ExtractingTool as ET

import cv2
import json
import requests
import json

@app.route('/recover/<index>')
def recoverIndex(index):
    img = cv2.imread(f"png/bar{index}.png",1)
    extracted_dict = ET.extract(img)
    et_pair = [(x_val, y_val) for (x_val, y_val) in enumerate(extracted_dict["ValorBarras"])]
    x, y = list(zip(*et_pair))
    x, y = list(map(lambda d: str(d), x)), map(lambda d: str(d), y)
    charttype = "charttype"
    extracted_dict["ValorEixos"] = x
    y_str = ",".join(y)
    address = f'x={",".join(list(x))}&y={y_str}&chart={charttype}' + \
              f'&title={extracted_dict["titulo"]}&xlabel={extracted_dict["labelX"]}&ylabel={extracted_dict["labelY"]}&base64=true'
    # address = f'http://127.0.0.1:3000/chartgen.html?x={",".join(list(x))}&y={",".join(y)}&chart={charttype}' + \
    #           f'&title={extracted_dict["titulo"]}&xlabel={extracted_dict["labelX"]}&ylabel={extracted_dict["labelY"]}&base64=true'
    # r = requests.get(address)
    print(address, json.dumps(extracted_dict))
    return address + ";" + json.dumps(extracted_dict) + ";" + y_str

@app.route('/bypass/<args>')
def bypass(args):
    r = requests.get("http://127.0.0.1:3000/chartgen.html?" + args)
    return r.text

@app.route('/recover/', methods=["POST"])
def recover():
    pass
    # extracted_dict = ET.extract(img)
    # return json.dumps(extracted_dict)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=45000)