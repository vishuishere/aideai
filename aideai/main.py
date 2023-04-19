from flask import Flask
from flask import request
import pandas as pd
import sys
import multiprocessing
app = Flask(__name__)

import time
import requests
import rq
from worker import conn
queue = rq.Queue(connection=conn,default_timeout=3600)

def predict_liver(uploaded_file_url):
    # run_ds(uploaded_file_url)
    pass


@app.route("/")
def home():
    return "AIDE DS App"

@app.route('/predict_liver_segment',methods=['POST'])
def predict_liver_segment():
    print("test the video with parameters : ", str(request.args))
    # uploaded_file_url = request.args['liver_path']
    # res = predict_liver(uploaded_file_url)
    return {"status": "ok"}


if __name__ == "__main__":
    
    # sys.argv.extend(['-a', SOME_VALUE])
    app.run(host='0.0.0.0', port=8082)

