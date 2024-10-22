# Amazon Customer Sentiment Analysis

This project aims to perform sentiment analysis on Amazon product reviews, using various machine learning models such as Logistic Regression, Random Forest, XGBoost, and LSTM (Long Short-Term Memory). By analyzing customer feedback, we aim to predict the sentiment behind each review—whether positive or negative—based on the review text.

## Project Overview:
1. **Goal**-	:To analyze customer reviews and predict the sentiment (positive or negative) using different machine learning and deep learning models.
2. **Dataset**	: The dataset used for this project is sourced from publicly available Amazon product reviews.

	-**Rows**: Varies depending on your sample size.
	- **Columns**: label (sentiment: positive/negative), text (review content).
	- **Target Variable**: Sentiment (1 for positive, 0 for negative).

## Techniques Used:
1. **Data Preprocessing**:
	Handling missing values.
	Text preprocessing including tokenization, stemming, and lowercasing.
	Label encoding for sentiment classification.

2. **Exploratory Data Analysis (EDA)**:
	Word Cloud and frequency analysis to identify key terms in positive and negative reviews.
	Visualization of review length, term frequency, and class distributions.
	
3. **Modeling**:

	A comparison of classification models, including Logistic Regression, Random Forest, XGBoost, and LSTM.
	Cross-validation and hyperparameter tuning using GridSearchCV.

3. **Model Evaluation**:
	Accuracy, Precision, Recall, F1-Score, and ROC-AUC to evaluate and compare model performance.
	
**Results**:
	Best Model: The LSTM (Long Short-Term Memory) model provided the best performance:
	Accuracy: 88.00%
	ROC-AUC Score: 0.9407

**Model Performance Comparison**:

	| Model               | Accuracy | Precision | Recall | ROC-AUC |
	|---------------------|----------|-----------|--------|---------|
	| Logistic Regression  | 87.02%   | 88%       | 87%    | 0.9432  |
	| Random Forest        | 84.15%   | 85%       | 84%    | 0.9196  |
	| XGBoost              | 84.35%   | 86%       | 85%    | 0.9235  |
	| LSTM                 | 88.00%   | 89%       | 88%    | 0.9407  |


## Visualizations:

### Confusion Matrix:
Here is the confusion matrix for Logistic Regression:

![Confusion Matrix](output/confusion_matrix_logistic_regression.png)

### ROC-AUC Curves:
Below are the ROC-AUC curves for various models:

- **Logistic Regression**:
![ROC-AUC Logistic Regression](output/roc_curve_logistic%20regression.png)

- **LSTM**:
![ROC-AUC LSTM](output/roc_curve_lstm.png)

- **Random Forest**:
![ROC-AUC Random Forest](output/roc_curve_random%20forest.png)

- **XGBoost**:
![ROC-AUC XGBoost](output/roc_curve_xgboost.png)

## Improvements:
1.Data Imbalance: Use SMOTE or downsampling to address the imbalance in positive vs. negative reviews.
2.Tuning: Further hyperparameter tuning using RandomizedSearchCV to improve model performance.
3.Text Embeddings: Consider using advanced text embeddings like Word2Vec or GloVe for feature extraction.


## How to Run:
1.Clone the repository:



	git clone https://github.com/himanshu-dandle/Amazon_customer_sentiment_analysis.git

2.Install dependencies:


	pip install -r requirements.txt
	
3.Navigate to the Jupyter notebook in the notebooks/ directory and run the analysis.

4.To replicate the results, use the preprocessed datasets from data/ and execute the .ipynb notebook.

5.Download dataset:

You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
Place the train.ft.txt and test.ft.txt files in the data/ folder.

### Run the Jupyter notebook:

	jupyter notebook notebooks/sentiment_analysis.ipynb
	
## Conclusion
The LSTM model outperformed traditional machine learning models like Logistic Regression and Random Forest in sentiment analysis tasks for Amazon product reviews, demonstrating the effectiveness of deep learning models in text-based classification tasks.
