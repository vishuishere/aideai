from flask import Flask, url_for
from flask import request
import pandas as pd
import sys
# import multiprocessing
import monaitrain
import monaitest
import time
# import requests
# import rq
from worker import conn
app = Flask(__name__)
# queue = rq.Queue(connection=conn,default_timeout=3600)

@app.route('/outputurl')
def outputurl():
    media_url = 'http://192.168.1.3:8082/media/liver/output1.png'
    return {"output": media_url}

@app.route("/")
def home():
    return "AIDE DS App"

@app.route('/predict_liver_segment',methods=['POST'])
def predict_liver_segment():
    print("test the 3d image", str(request.args))
    data = request.args.get('data')
    monaitest.testing(data)
    url = 'http://192.168.1.3:8082/'
    media_url = url + 'media/'+data+'/output1.png'
    return {"output": media_url}


if __name__ == "__main__":
    
    # sys.argv.extend(['-a', SOME_VALUE])
    app.run(host='0.0.0.0', port=8082)

