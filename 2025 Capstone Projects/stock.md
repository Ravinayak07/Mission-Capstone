Acts as a professional research paper write and write a research paper on "Stock Price Prediction with twitter sentimental analysisi using Machine Learning"
You need to write these three Sections only:

1. Abstract
2. Introduction
3. Literature Review
4. Proposed Methodology

Write in IEEE format. Mention refernces whenever required.
Write in descriptive and lengthy manner.

I am also giving you the project code below for refernces:

MODEL TRAINING CODE:
import pandas as pd
import numpy as np
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import re

# Initialize NLTK components

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

# Load news headlines and stock data

news_df = pd.read_csv("Dataset/india-news-headlines.csv")

# Define functions for data cleaning and feature engineering

def clean_text(text):
text = text.lower()
text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
stop_words = set(stopwords.words('english'))
words = text.split()
cleaned_text = ' '.join([word for word in words if word not in stop_words])
return cleaned_text

# Preprocess news headlines

news_df.drop("headline_category", axis=1, inplace=True)
news_df.drop_duplicates(inplace=True)
news_df['publish_date'] = pd.to_datetime(news_df['publish_date'], format='%Y%m%d')
news_df = news_df.groupby(['publish_date'])['headline_text'].apply(lambda x: ','.join(x)).reset_index()
news_df["cleaned_headline"] = news_df["headline_text"].apply(clean_text)

# Perform sentiment analysis on news headlines

analyzer = SentimentIntensityAnalyzer()
news_df["sentiment"] = news_df["cleaned_headline"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
news_df["sentiment_pos"] = news_df["cleaned_headline"].apply(lambda x: analyzer.polarity_scores(x)["pos"])
news_df["sentiment_neu"] = news_df["cleaned_headline"].apply(lambda x: analyzer.polarity_scores(x)["neu"])
news_df["sentiment_neg"] = news_df["cleaned_headline"].apply(lambda x: analyzer.polarity_scores(x)["neg"])

# Save the processed DataFrame to a new CSV file

news_df.to_csv('refined_news_data.csv', index=False)

stock_df = pd.read_csv("Dataset/stock_data.csv")

def calculate_rsi(series, window):
delta = series.diff().dropna()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=window, min_periods=1).mean()
avg_loss = loss.rolling(window=window, min_periods=1).mean()
rs = avg_gain / avg_loss
rsi = 100 - (100 / (1 + rs))
return rsi

def calculate_obv(df):
obv = (np.sign(df['Close'].diff()) \* df['Volume']).fillna(0).cumsum()
return obv

def calculate_technical_indicators(df): # Simple Moving Averages (SMA)
df['SMA_5'] = df['Close'].rolling(window=5).mean()
df['SMA_10'] = df['Close'].rolling(window=10).mean() # Exponential Moving Averages (EMA)
df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean() # Moving Average Convergence Divergence (MACD)
df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean() # Relative Strength Index (RSI)
df['RSI'] = calculate_rsi(df['Close'], window=14) # On-Balance Volume (OBV)
df['OBV'] = calculate_obv(df)
return df

# Preprocess stock data

# Drop unnecessary columns

stock_df.drop(["Adj Close"], axis=1, inplace=True)

stock_df['Date'] = pd.to_datetime(stock_df['Date'], format="%b %d, %Y")
numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
stock_df[numeric_columns] = stock_df[numeric_columns].replace('-', np.nan)
stock_df[numeric_columns] = stock_df[numeric_columns].replace(',', '', regex=True).astype(float)
stock_df["Volume"] = stock_df["Volume"].interpolate(method="linear")

# Feature engineering for stock data

stock_df = calculate_technical_indicators(stock_df.copy())

# Save the processed DataFrame to a new CSV file

stock_df.to_csv('refined_stock_data.csv', index=False)

stock_df= pd.read_csv('refined_stock_data.csv')

# Merge news and stock data using date as the key

news_df.rename(columns={"publish_date": "Date"}, inplace=True)
merged_df = pd.merge(news_df, stock_df, on="Date", how="inner")

# Drop unnecessary columns

merged_df.drop(["headline_text","cleaned_headline"], axis=1, inplace=True)

# Save the processed DataFrame to a new CSV file

merged_df.to_csv('refined_merged_data.csv', index=False)

def rolling_window_forecast(df, window_size):
predictions = []
actuals = []
dates = []

    for i in range(len(df) - window_size):
        train_df = df[i:i + window_size]
        test_df = df[i + window_size:i + window_size + 1]

        # Prepare training data
        X_train = train_df.drop(['Date', 'Close'], axis=1)
        y_train = train_df['Close']

        # Prepare test data
        X_test = test_df.drop(['Date', 'Close'], axis=1)
        y_test = test_df['Close']
        dates.append(test_df['Date'].values[0])

        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = model.predict(X_test_scaled)

        predictions.append(y_pred[0])
        actuals.append(y_test.values[0])

        # Debugging print statements
        print(f"Window {i+1}/{len(df) - window_size}")
        print(f"Train period: {train_df['Date'].values[0]} to {train_df['Date'].values[-1]}")
        print(f"Test date: {test_df['Date'].values[0]}")
        print(f"Prediction: {y_pred[0]}, Actual: {y_test.values[0]}")

    return predictions, actuals, dates

# Apply rolling window forecast

from sklearn.preprocessing import StandardScaler

predictions, actuals, dates = rolling_window_forecast(merged_df, window_size)

# Evaluate the model's performance

mae = mean_absolute_error(actuals, predictions)
rmse = np.sqrt(mean_squared_error(actuals, predictions))
r2 = r2_score(actuals, predictions)

print(f"MAE: {mae}, RMSE: {rmse}, R-squared: {r2}")

# Plotting the predicted vs actual stock prices

plt.figure(figsize=(14, 7))
plt.plot(actuals, label='Actual Prices')
plt.plot(predictions, label='Predicted Prices')
plt.title('Actual vs Predicted SENSEX Stock Prices')
plt.xlabel('Time')
plt.ylabel('Stock Prices')
plt.legend()
plt.show()

# Now we need to train on LSTM model as well:

import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime

import matplotlib.pyplot as plt

import numpy as np

from sklearn.preprocessing import MinMaxScaler

### Create the Stacked LSTM model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from keras.layers import Dropout

df = pd.read_csv("preprocessed_merged_data.csv")

#print the head
df.head()

df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date']
plt.figure(figsize=(20,8))
plt.plot(df['Close'], label='Close')

features = ["Date", "Close"]
all_data = df[features]
all_data.index = all_data.Date
all_data.drop('Date', axis=1, inplace=True)

all_data.head()
all_data.shape

dataset = all_data.values
train = dataset[2000:4500,:]
valid = dataset[4500:,:]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(90,len(train)):
x_train.append(scaled_data[i-90:i,0])
y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

inputs = all_data[len(all_data) - len(valid)-90:].values
inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)
inputs.shape
X_test = []
for i in range(90,inputs.shape[0]):
X_test.append(inputs[i-90:i,0])
X_test = np.array(X_test)

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(Dropout(rate = 0.3))

model.add(LSTM(units=50, return_sequences = True))
model.add(Dropout(rate = 0.3))

model.add(LSTM(units=50, return_sequences = True))
model.add(Dropout(rate = 0.3))

model.add(LSTM(units=50, return_sequences = False))
model.add(Dropout(rate = 0.3))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
#we use standard adam's optimizer

model.fit(x_train, y_train, epochs=10, batch_size=128, verbose=1

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
preds = model.predict(X_test)
preds = scaler.inverse_transform(preds)

print(valid[-1],preds[-1])

rms=np.sqrt(np.mean(np.power((valid-preds),2)))

train = all_data[2000:4500]
valid = all_data[4500:]
valid['Predictions'] = preds
plt.figure(figsize=(20,8))
plt.plot(train['Close'])
plt.plot(valid['Close'], color = 'blue', label = 'Real Price')
plt.plot(valid['Predictions'], color = 'red', label = 'Predicted Price')
plt.title('HDFCBANK price prediction')
plt.legend()
plt.show()

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
%matplotlib inline

# For reading stock data from yahoo

from pandas_datareader.data import DataReader

# For time stamps

from datetime import datetime

df = pd.read_csv("preprocessed_merged_data.csv")

from keras.models import load_model
model.save('lastmodel.h5') # creates a HDF5 file

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Calculate the actual values from valid and the predictions

valid_actual = valid["Close"].values # Actual stock prices
valid_predictions = valid["Predictions"].values # Predicted stock prices
print(valid_actual, valid_predictions)

# Calculate Mean Absolute Error

mae = mean_absolute_error(valid_actual, valid_predictions)
print(f"Mean Absolute Error (MAE): {mae:.4f}")

# Calculate Mean Squared Error

mse = mean_squared_error(valid_actual, valid_predictions)
print(f"Mean Squared Error (MSE): {mse:.4f}")

# Calculate Root Mean Squared Error

rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# Calculate R-squared value

r2 = r2_score(valid_actual, valid_predictions)
print(f"R-squared (R²): {r2:.4f}")
