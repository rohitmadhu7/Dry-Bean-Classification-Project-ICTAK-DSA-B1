from flask import Flask, render_template, request
import pickle
import numpy as np

model=pickle.load(open('model.pkl','rb'))
le1=pickle.load(open('le.pkl','rb'))
sc1=pickle.load(open('sc.pkl','rb'))

app= Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    area=request.form.get("Area")
    Aspect_Ration=request.form.get("AspectRation")
    Extend=request.form.get("Extend")
    Solidity=request.form.get("Solidity")
    Roundness=request.form.get("Roundness")
    ShapeFactor2=request.form.get("ShapeFactor2")
    ShapeFactor4=request.form.get("ShapeFactor4")
    result=model.predict(np.array([sc1.transform(np.array([area]).reshape(1,-1))[0][0],Aspect_Ration,Extend,Solidity,Roundness,ShapeFactor2,ShapeFactor4]).reshape(1,7))
  
    result="Your Dry Bean belongs to" +" " + le1.inverse_transform(np.array([result]).reshape(1,-1))[0]


    return render_template('index.html',result=result)

if __name__=='__main__':
    app.run(debug=True)
    

