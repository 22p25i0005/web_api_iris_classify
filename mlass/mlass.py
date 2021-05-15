from flask import Flask,request
from flask_cors import CORS,cross_origin
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def helloworld():
    return 'Hello'

@app.route('/area')
@cross_origin()
def area():
    w = float(request.values['w'])
    h = float(request.values['h'])
    return str(w*h)

@app.route('/bmi',methods =['GET'])
@cross_origin()
def bmi():
    w = float(request.values['w'])
    h = float(request.values['h'])/100
    return str(w/(h**2))

@app.route('/test',methods =['GET'])
@cross_origin()
def test():
    w = float(request.values['w'])
    h = float(request.values['h'])/100
    return str(w/(h**2))

@app.route('/iris',methods=['POST','GET'])
@cross_origin()
def predict_species():
    model = joblib.load('iris.model')
    req = request.values['param']
    inputs = np.array(req.split(','),dtype=np.float32).reshape(1,-1)
    predict_target = model.predict(inputs)
    if predict_target == 0:
        return 'Sentosa'
    elif predict_target == 1:
        return 'Vassicolour'
    else:
        return 'Virginica'

if __name__ == '__main__':
    port = int(os.environ.get('PORT',5000))
    app.run(host='0.0.0.0',port=port)
