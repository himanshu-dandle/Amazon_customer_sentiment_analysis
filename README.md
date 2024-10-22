Amazon Customer Sentiment Analysis
This project performs sentiment analysis on Amazon customer reviews using various machine learning models like Random Forest, Logistic Regression, XGBoost, and LSTM. The dataset consists of customer reviews, aiming to predict whether the review has a positive or negative sentiment.

## Project Overview:
	### Goal: Classify customer reviews as positive or negative based on textual data.
	### Dataset: Amazon Customer Reviews dataset from Kaggle.
	### Rows: ~500,000 reviews
	### Columns: Includes review text, product IDs, and user IDs.
	### Target Variable: Sentiment (positive/negative)

## Techniques Used:
	### Data Preprocessing:
	Cleaned text (removing special characters, numbers, and stopwords).
	Tokenized and vectorized using TF-IDF to transform text into numerical data.

## Exploratory Data Analysis (EDA):
	Visualized distribution of sentiment labels, most frequent words, and the relationship between word frequency and sentiment.
	### Modeling:

		Built and evaluated multiple models: Logistic Regression, Random Forest, XGBoost, and LSTM.
		Applied cross-validation and hyperparameter tuning to improve model performance.

## Project Steps:
	### Data Preprocessing:
		Handled missing values and text cleaning.
		Used TF-IDF vectorization to convert text into numerical form for modeling.

## Exploratory Data Analysis (EDA):
	Analyzed key features such as word frequency and sentiment distribution.
	Created visualizations showing customer review sentiment trends.

## Modeling:
	Trained models like Random Forest, Logistic Regression, XGBoost, and LSTM.
	Applied hyperparameter tuning using RandomizedSearchCV to improve results.

## Model Evaluation:
	Evaluated models using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.
	### Results:
	Best Model: The LSTM model performed the best with:
	Accuracy: 88%
	ROC-AUC: 0.94
	Feature Importance:
	Important features for sentiment classification:
	Review Text (keyword frequency)
	Review Length

## Improvements:
	### Class Imbalance: Addressed using SMOTE to balance the number of positive and negative reviews.
	### Hyperparameter Tuning: Used RandomizedSearchCV for optimizing model performance.


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

These visualizations help demonstrate the performance of the models and show their ability to distinguish between positive and negative sentiments.



These visualizations provide insights into model performance and feature importance.


## How to Run:
### Clone the repository:



	git clone https://github.com/himanshu-dandle/Amazon_customer_sentiment_analysis.git

### Install dependencies:



	conda env create -f environment.yml
	conda activate sentiment-env
### Download dataset:

Place the train.ft.txt and test.ft.txt files in the data/ folder.
### Run the Jupyter notebook:

	jupyter notebook notebooks/sentiment_analysis.ipynb
