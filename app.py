from flask import Flask, render_template, request
import pickle
import numpy as np
import os

filename = 'xg_model.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('main.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':

        soiltype = int(request.form.get('soiltype'))
        ph = float(request.form['ph'])
        Organic = float(request.form['Organic'])
        CEC = float(request.form['CEC'])
        Texture = int(request.form.get('Texture'))
        temp = float(request.form['temp'])
        Precipitation = float(request.form['Precipitation'])
        land = int(request.form.get('land'))
        modelSelected = int(request.form.get('selectModel'))
        
        sample_input = np.array([[soiltype,ph,Organic,CEC,Texture,temp,Precipitation,land]])
        filedata = ['dt_model.pkl' , 'knn_model.pkl', 'mlp_model.pkl', 'rf_model.pkl' ,'xg_model.pkl' ]
        filename = filedata[modelSelected]
        model = pickle.load(open(filename, 'rb'))
        my_prediction = model.predict(sample_input)
        prediction_list = my_prediction.tolist()

    return render_template('result.html', prediction=prediction_list,selection = modelSelected)
        
        

if __name__ == '__main__':
	#app.run(debug=True)
	port = int(os.environ.get('PORT', 5000))
	app.run(host='0.0.0.0', port=port, debug=True)