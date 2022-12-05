from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv('220511_monatszahlenmonatszahlen2204_verkehrsunfaelle.csv')
df = df[['MONATSZAHL', 'AUSPRAEGUNG' , 'JAHR', 'MONAT', 'WERT']]
df = df.dropna()
df['WERT'] = pd.to_numeric(df['WERT'], errors='coerce')
df_filtered = df[df['MONAT'] != 'Summe']
df_filtered = df_filtered.sort_values(by='MONAT')
df_filtered = df_filtered[df_filtered['AUSPRAEGUNG'] == 'insgesamt']
df_Alkoholunfalle = df_filtered[df_filtered['MONATSZAHL'] == 'Alkoholunfälle']
df = df_Alkoholunfalle[['MONAT', 'WERT']]
df.reset_index(drop=True)
dataset = df.iloc[:,1].values
dataset = dataset.reshape(-1,1)
df = df.set_index('MONAT')
train = df.iloc[:252]
test= df.iloc[252:]
test= test[0:12]


model = joblib.load('model.joblib', mmap_mode=None)


scaler = MinMaxScaler()
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test) 

trainPredict = model.predict(scaled_train)
testPredict = model.predict(scaled_test)
testPredict = scaler.inverse_transform(testPredict)
trainPredict = scaler.inverse_transform(trainPredict)

train['Predictions'] =trainPredict
test['Predictions'] =testPredict

test = test.reset_index()
train =train.reset_index()

test['MONAT'] = (test['MONAT']).astype(int)
train['MONAT'] = (train['MONAT']).astype(int)


app = Flask(__name__)


@app.route("/", methods=['GET'])
def Home():
    return render_template('index.html')




@app.route('/predict', methods=['POST'])
def predict():

    year = str(request.form.get('year'))
    month = str(request.form.get('month'))
    year_int = int(year)
    month_int = int(month)

    if month_int < 10: month = "0" + month

    date = year + month


    if year_int >= 2021:	
    	d = test['Predictions'][test['MONAT'] == float(date)]
    	prediction = round(d).astype(int)
    elif year_int < 2021:
    	d = train['Predictions'][train['MONAT'] == float(date)]
    	prediction = round(d).astype(int)
    
    prediction = prediction.reset_index(drop=True)
    prediction = prediction[0]
    prediction = prediction.tolist()
    
    
    return render_template(
        'index.html',
        prediction_text='prediction value is {}'.format(prediction))    

if __name__ == '__main__':
    app.run(debug=True) 


