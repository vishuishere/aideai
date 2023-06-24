from flask import Flask, url_for
from flask import request
import pandas as pd
import sys
import os
import monaitrain
import testing
import monaitest
import time
from werkzeug.utils import secure_filename
import requests
from flask_cors import CORS
import boto3

app = Flask(__name__)
cors = CORS(app)
app.config['API_URL'] = 'http://ai-api.googerit-ai.com/'
app.static_folder = "/home/ubuntu/mystorage"#os.getcwd()#'/home/ubuntu/app_dev_ai/core/aide_ai/aideai'

@app.route('/outputurl')
def outputurl():
    api_url = app.config['API_URL']
    media_url = 'http://ai-api.googerit-ai.com/static/brain/output1.png'
    return {"output": media_url}

@app.route("/")
def home():
    api_url = app.config['API_URL']
    return "AIDE DS App"



@app.route('/predict_result',methods=['POST', 'GET'])
def predict_result():
    print("test the 3d image", str(request.args))
    data = request.args.get('data')
    project_name = request.args.get('project_name')
    sample_name = request.args.get('sample_name')
    url = 'http://ai-api.googerit-ai.com'#app.config['API_URL']
    output = monaitest.get_output_url(url, data, project_name, sample_name)
    if len(output)>0:
        path = 'media/' + project_name + '/' + data + '/' + sample_name + '/output.dcm'
        final_res_path = 'media/' + project_name + '/' + data + '/' + sample_name + '/final_output.png'
        # print(path)
        
        dicom_path = monaitest.get_dicom_file_path(project_name, data, sample_name)
        static_url = url_for('static', filename=path)
        static_url_final = url_for('static', filename=final_res_path)
        return {"output": output, "dicom_data": dicom_path, "final_output": url + static_url_final}
    else:
        return {"output": output, "dicom_data": None, "final_output": None}

@app.route('/upload', methods=['POST'])
def upload_file():
    data = request.form['data']
    project_name = request.form['project_name']
    sample_name = request.form['sample_name']

    uploaded_file = request.files['input_file']

    if uploaded_file:
        path = '/home/ubuntu/mystorage/media/' + project_name + '/' + data + '/' + sample_name + '/input/'
        vol_path = path + 'TestVolumes/'
        seg_path = path + 'TestSegmentation/'
        filename = secure_filename(uploaded_file.filename)
        file_path = os.path.join(vol_path, filename)
        try:
            os.makedirs(path, exist_ok=True)
        except Exception as ex:
            print("Exception: ", ex)
        try:
            os.makedirs(vol_path, exist_ok=True)
        except Exception as ex:
            print("Exception: ", ex)
        try:
            os.makedirs(seg_path, exist_ok=True)
        except Exception as ex:
            print("Exception: ", ex)
        uploaded_file.save(os.path.join(vol_path, filename))
        print("File saved in input folder")
        url = 'http://ai-api.googerit-ai.com'#app.config['API_URL']
        try:
            output = monaitest.testing(url, file_path, data, project_name, sample_name)
            print(output)
            if 'compose' in str(output):
                return 'compose'
            elif len(output) >= 1:
                 return 'success'
            
            return 'failed'
        except Exception as ex:
            print("Exception as ex: ", ex)
            if 'compose' in str(ex).lower() or 'transform' in str(ex).lower():
                print('compose')
                return 'compose'
            print('failed')
            return "failed: " + str(ex)

    return 'No file uploaded.'


if __name__ == "__main__":
    
    # sys.argv.extend(['-a', SOME_VALUE])
    app.run(host='0.0.0.0', port=8095)

