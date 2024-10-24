**ARIMA, State Space Models, VAR, Error Correction Models etc.)**

Key concept in ARIMA is **Autocorrelation**. 

How does it different from the typical correlation? First of all, correlation relates two different sets of observations (eg. between housing prices and the number of available public amenities) 
while autocorrelation relates the same set of observation but across different timing (eg. between rainfall in the summer versus that in the fall).

Auto Regressive (AR) regression model is built on top of the autocorrelation concept, where the dependent variable depends on the past values of itself (eg. rainfall today may depend…)


Building the ARIMA Model

```
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(data['births'], order=(1, 0, 1))
model_fit = model.fit()
```

Training and Forecasting

We train the model on the data and perform a forecast.
```
forecast = model_fit.get_forecast(steps=30)
```

## data preprocessing step

 real-world projects involve a lot of cleaning and preparation.

    Detrending/ Stationarity: Before forecasting, we want our time series variables to be mean-variance stationery. This means that the statistical properties of a model do not vary depending on when the sample was taken. Models built on stationary data are generally more robust. This can be achieved by using differencing.
    Anomaly detection: Any outlier present in the data might skew the forecasting results so it’s often considered a good practice to identify and normalize outliers before moving on to forecasting. You could follow this blog here where I have explained anomaly detection algorithms at length.
    Check for sampling frequency: This is an important step to check the regularity of sampling. Irregular data has to be imputed or made uniform before applying any modeling techniques because irregular sampling leads to broken integrity of the time series and doesn’t fit well with the models.
    Missing data: At times there can be missing data for some datetime values and it ne



Simply put the Autoregressive Integrated Moving Average (ARIMA) tries to model a time series where your time series in question, y, can be explained by its own lagged values (Autoregressive part) and error terms (Moving Average part). The "Integrated" part of the model (the "I" in "ARIMA") refers to how many times the series has been differenced to achieve stationarity.

Stationarity is a must before you can model your data: what stationarity refers to is constant mean and variance. Think of these two moments as not being time dependent. The reason for this is quite simple, it's difficult to model something which changes over time.

Prediction = constant + linear combination lags of Y + linear combination of lagged forecast errors

ARIMA, p is AR, d is I and q is MA.  here our assumption is right. These parameters can be explained as follows

    p is the number of autoregressive terms,
    d is the number of nonseasonal differences,
    q is the number of lagged forecast errors in the prediction equation
    
    SARIMA stands for Seasonal-ARIMA and it includes seasonality contribution to the forecast. The importance of seasonality is quite evident and ARIMA fails to encapsulate that information implicitly.
