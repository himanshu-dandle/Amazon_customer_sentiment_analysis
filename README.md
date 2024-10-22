Amazon Customer Sentiment Analysis
This project aims to perform sentiment analysis using the Amazon Customer Reviews dataset from Kaggle. We apply various machine learning models, including Random Forest, Logistic Regression, and LSTM to analyze customer sentiment and provide actionable insights.

Project Overview:
Goal: To predict customer sentiment based on reviews.
Dataset: Contains customer reviews, ratings, and sentiment labels (positive/negative).
Rows: ~500,000 customer reviews.
Columns: 10 features, including text reviews, product IDs, and user IDs.
Target Variable: Sentiment (positive/negative).
Techniques Used:
Data Preprocessing: Cleaned text, tokenized words, and applied vectorization (TF-IDF).
Exploratory Data Analysis (EDA): Visualized the distribution of sentiments, word frequencies, and customer feedback.
Modeling: Trained multiple models and evaluated their performance.
Project Steps:
Data Preprocessing:

Removed missing values and irrelevant characters.
Converted text data into numerical form using TF-IDF.
Exploratory Data Analysis (EDA):

Visualized sentiment distribution.
Explored word frequency and review patterns.
Analyzed relationships between word usage and sentiment.
Modeling:

Trained models like Logistic Regression, Random Forest, XGBoost, and LSTM.
Applied cross-validation and hyperparameter tuning using RandomizedSearchCV for better accuracy.
Model Evaluation:

Evaluated models using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.
Selected the best-performing model.
Results:
Best Model: The LSTM model achieved the highest accuracy:
Accuracy: 88%
ROC-AUC Score: 0.94
Feature Importance: Key influential features for sentiment prediction:
Review Text (keywords and word usage)
Review Length
Improvements:
Class Imbalance: Addressed class imbalance using techniques like SMOTE.
Hyperparameter Tuning: Optimized models for better accuracy using RandomizedSearchCV.
Visualizations:
Here are some key visualizations from the analysis:

Confusion Matrix:

ROC-AUC Curve:


These visualizations provide insights into model performance and the importance of features in predicting customer sentiment.

Model Performance Comparison:
Model	Accuracy	Precision	Recall	ROC-AUC
Logistic Regression	0.87	0.88	0.86	0.94
Random Forest	0.84	0.84	0.83	0.92
XGBoost	0.84	0.85	0.83	0.92
LSTM	0.88	0.89	0.86	0.94
How to Run:
Clone the repository:



git clone https://github.com/himanshu-dandle/Amazon_customer_sentiment_analysis.git
Install the dependencies:
conda env create -f environment.yml
conda activate sentiment-env
Download the dataset: Place the train.ft.txt and test.ft.txt files in the data/ folder.

Run the notebook:
jupyter notebook notebooks/sentiment_analysis.ipynb
