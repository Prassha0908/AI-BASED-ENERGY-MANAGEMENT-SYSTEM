from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import statsmodels.api as sm

app = Flask(__name__)

def predict_ARIMA(data, forecast_days):
    model = sm.tsa.arima.ARIMA(data, order=(2, 1, 2))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_days)
    return forecast






def predict_SARIMA(data, forecast_days):
    model = sm.tsa.statespace.SARIMAX(data, order=(2, 1, 2), seasonal_order=(0, 1, 1, 12))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_days)
    return forecast

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get data from form
        forecast_days = int(request.form['forecast_days'])
        file = request.files['file']
        data = pd.read_csv(file)
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.set_index('Date')
        data = data.rename(columns={'#Passengers': 'Total'})
        
        # Predict using ARIMA
        arima_forecast = list(predict_ARIMA(data['Total'], forecast_days))
        
        # Predict using SARIMA
        sarima_forecast = list(predict_SARIMA(data['Total'], forecast_days))

        # Prepare data for rendering
        arima_dates = pd.date_range(start=data.index[-1], periods=forecast_days+1)[1:]
        sarima_dates = pd.date_range(start=data.index[-1], periods=forecast_days+1)[1:]

        return render_template('result.html', 
                                arima_forecast=arima_forecast, 
                                sarima_forecast=sarima_forecast,
                                arima_dates=arima_dates,
                                sarima_dates=sarima_dates)
    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True)
