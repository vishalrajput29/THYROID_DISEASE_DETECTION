import pickle 
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
pred_model=pickle.load(open('model.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    
    def output(output):
        if output==1:
            print("negative")
        elif output ==2:
            print("compensated_hypothyroid")
        elif output==3:
            print("primary_hypothyroid")
        else:
            print("secondary_hypothyroid")
            
    output=pred_model.predict(final_input)[0]
    return render_template("index.html",prediction_text="The type of thyroid is {}".format(output))




if __name__=="__main__":
    app.run(host="0.0.0.0",port=8000)