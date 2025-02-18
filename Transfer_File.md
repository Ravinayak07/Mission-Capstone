# VI. MODEL TRAINING
- Stock price prediction requires careful model selection and training to achieve meaningful accuracy. In this study, we implemented and trained two distinct machine learning models—Long Short-Term Memory (LSTM) networks and Random Forest regressors—to assess their effectiveness in forecasting stock prices based on historical data and sentiment analysis. Each model was trained separately using refined datasets consisting of stock market indicators and sentiment scores extracted from Twitter and news headlines.


# A. LSTM Model Training
- LSTM networks, a variant of recurrent neural networks (RNNs), are particularly well-suited for time-series forecasting due to their ability to capture long-term dependencies [1]. In this study, we utilized a stacked LSTM architecture with multiple hidden layers, each containing 50 neurons. The input to the LSTM model consisted of stock price data and sentiment features, which were scaled using the MinMaxScaler to ensure normalized input values. A sequence length of 90 days was chosen to capture temporal dependencies.
- The model was trained using the Adam optimizer and Mean Squared Error (MSE) as the loss function. A dropout rate of 30% was incorporated into each LSTM layer to mitigate overfitting. The dataset was split into training (70%) and testing (30%) sets, ensuring that past data was used to predict future stock prices. After 10 epochs of training, the model was evaluated using standard regression metrics.

# B. Random Forest Model Training
- Random Forest, an ensemble learning method, was also employed for stock price prediction [2]. Unlike LSTMs, which process sequential data, Random Forest operates by constructing multiple decision trees and averaging their outputs to reduce variance and improve generalization. The input features included stock price indicators such as moving averages (SMA, EMA), Relative Strength Index (RSI), On-Balance Volume (OBV), and sentiment scores.
- For training, the dataset was split using a rolling window approach to ensure a realistic forecasting scenario. The number of estimators (trees) in the Random Forest was set to 100, and the mean absolute error (MAE) was used as the primary evaluation metric. StandardScaler was applied to standardize the feature set, ensuring a fair comparison between features of different magnitudes.

# VII. PERFORMANCE EVALUATION
- The performance of both models was evaluated using multiple statistical metrics, providing insight into their predictive capabilities.

## A. Evaluation Metrics
- To assess the accuracy of stock price predictions, we used the following key evaluation metrics:
- Mean Absolute Error (MAE): Measures the average absolute difference between predicted and actual stock prices, providing an intuitive measure of prediction accuracy [3].
- Mean Squared Error (MSE): Evaluates the squared differences, penalizing larger errors more heavily [3].
- Root Mean Squared Error (RMSE): The square root of MSE, useful for interpreting the magnitude of prediction errors [4].
- R-Squared (R²): Indicates how well the model explains the variance in stock prices. Higher values signify better model performance [4].
- After training and testing both models, their performance was compared based on these metrics to determine the more suitable approach for stock price prediction.

# VIII. PERFORMANCE COMPARISON:
- A comparative analysis was conducted to determine which model provided better forecasting accuracy. The LSTM model demonstrated superior performance in capturing temporal patterns and stock price trends, with lower RMSE and higher R² values. However, it required more computational resources and longer training times. Conversely, the Random Forest model performed well with structured numerical data and was computationally efficient but struggled with sequential dependencies [2].
- The findings align with prior research emphasizing the strength of LSTMs in time-series forecasting [1]. Meanwhile, Random Forest remains a viable alternative for interpretable, rapid predictions [2]. Future work may explore hybrid models that leverage the advantages of both approaches.

# References:
- [1] S. Hochreiter and J. Schmidhuber, "Long short-term memory," Neural computation, vol. 9, no. 8, pp. 1735-1780, 1997.
- [2] L. Breiman, "Random forests," Machine learning, vol. 45, no. 1, pp. 5-32, 2001.
- [3] J. Han, M. Kamber, and J. Pei, Data Mining: Concepts and Techniques, 3rd ed. Morgan Kaufmann, 2011.
- [4] B. Bollen, H. Mao, and X. Zeng, "Twitter mood predicts the stock market," Journal of Computational Science, vol. 2, no. 1, pp. 1-8, 2011.
