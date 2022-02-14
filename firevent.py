import subprocess
#from vizmodule.VizServer.imagesocket import genbytesfrom64

def fireAugment(jsonBars, buffer):
    #print(jsonBars)
    cmd = 'node vizmodule/augment.js ' + jsonBars
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=False)
    out, err = p.communicate(buffer.encode())
    out = out.decode('utf-8')
    result = out.split('\n')
    img64 = ''
    prefix = "data:image/png;base64,"
    for lin in result[:-1]:
        if not lin.startswith('#'):
            #print(lin)
            img64 = lin[len(prefix):]
    #print(img64)
    return img64

def fireChartGeneration(filename):
    cmd = 'node vizmodule/chartGeneration/chartGeneration.js ' + filename
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    out = out.decode('utf-8')
    result = out.split('\n')
    for lin in result[:-1]:
        if not lin.startswith('#'):
            print(lin)