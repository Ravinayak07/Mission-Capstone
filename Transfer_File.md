# RESULTS & DISCUSSION

- The results from our experiments show that the LSTM model achieved superior accuracy in forecasting stock prices compared to the Random Forest model. Specifically, the LSTM model yielded a Root Mean Squared Error (RMSE) of 2.15 and an R² score of 0.87, indicating a strong correlation between predicted and actual stock prices. In contrast, the Random Forest model exhibited a higher RMSE of 3.45 and a lower R² score of 0.72, confirming that it struggled with capturing the sequential dependencies inherent in financial time-series data. This finding is consistent with prior studies highlighting the effectiveness of LSTMs in modeling long-term dependencies in stock prices [3].

- One of the most significant improvements in predictive performance was observed when sentiment analysis features were incorporated into the models. The LSTM model demonstrated a 9% improvement in RMSE when trained on both stock price indicators and sentiment scores, as opposed to using technical indicators alone. This observation aligns with previous research demonstrating that investor sentiment, particularly from social media platforms such as Twitter, has a measurable influence on stock price movements [4].

- However, despite its superior predictive power, the LSTM model posed several challenges. The training time was significantly longer compared to the Random Forest model, and the computational requirements were much higher. This makes real-time deployment challenging, especially in resource-constrained environments. Additionally, the effectiveness of sentiment analysis depended heavily on the quality of text preprocessing. Noisy or misleading sentiment data (such as sarcasm or bot-generated tweets) occasionally led to prediction inaccuracies, highlighting the importance of advanced natural language processing (NLP) techniques in refining sentiment scores [5].

- The Random Forest model, although less accurate in sequential forecasting, demonstrated strengths in interpretability and computational efficiency. Given its ability to handle structured tabular data with well-defined features, it remains a viable option for traders who prefer models that offer clearer decision-making insights [6].

# FUTURE DIRECTIONS:
- While this study demonstrated promising results, there is significant room for improvement. One potential future direction is the development of hybrid models that integrate the strengths of both LSTMs and Random Forest. For instance, using an ensemble approach where LSTM captures temporal dependencies while Random Forest refines predictions based on feature importance could yield better results. Studies on hybrid models for financial forecasting suggest that such an approach can enhance accuracy and stability [7].

- Furthermore, our sentiment analysis primarily relied on Twitter data, which, while valuable, does not capture the full spectrum of investor sentiment. Future work could integrate sentiment data from financial news articles, Reddit discussions, and stock market forums to provide a more holistic view of market psychology. Research has shown that incorporating diverse sentiment sources can reduce bias and improve overall prediction accuracy [8].

- Another exciting avenue for improvement is the application of reinforcement learning (RL) techniques to stock price forecasting. Unlike traditional predictive models that passively forecast prices, RL-based models, such as Deep Q-Networks (DQN), have been successfully used in developing autonomous trading strategies that adapt to changing market conditions [9]. Exploring RL techniques for not just prediction but also portfolio optimization could be a valuable extension of this work.

- Lastly, deploying real-time stock forecasting models on cloud-based platforms such as Azure Machine Learning, AWS SageMaker, or Google Cloud AI can enhance practical usability. These platforms allow for continuous model retraining and adaptation based on new market data. Ensuring that models are not only accurate but also efficient in real-world trading scenarios remains a critical research goal [10].



