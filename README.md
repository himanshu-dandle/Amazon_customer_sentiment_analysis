Amazon Customer Sentiment Analysis
This project involves sentiment analysis on Amazon customer reviews, utilizing natural language processing (NLP) techniques to classify reviews as positive or negative.

Project Structure


AMZ_customer_sentiment/
│
├── data/                   # Dataset files (excluded from GitHub)
├── notebooks/              # Jupyter notebooks for analysis
├── output/                 # Outputs: Plots, models, etc.
├── environment.yml         # Conda environment configuration
├── README.md               # Project details and instructions
└── requirements.txt        # Python dependencies
Dataset
Source: Amazon customer reviews dataset (you can specify more details here if applicable).
Size: The dataset contains reviews with corresponding sentiment labels (positive/negative).
Workflow
Data Preprocessing:

Tokenization
Removing stopwords
Text vectorization using TF-IDF.
Modeling:

Models used:
Logistic Regression
Random Forest
XGBoost
LSTM (Deep Learning)
Evaluation: Accuracy, Precision, Recall, F1-Score, ROC-AUC.
Results:

The LSTM model outperformed other models with the highest accuracy.
Detailed comparison and model performance metrics are documented in the notebook.
How to Run
Clone the repository:



git clone https://github.com/himanshu-dandle/Amazon_customer_sentiment_analysis.git
Create and activate the Conda environment:



conda env create -f environment.yml
conda activate customer-sentiment-env
Launch the Jupyter notebook:



jupyter notebook notebooks/sentiment_analysis.ipynb
Ensure the dataset files are in the data/ folder and excluded from version control (.gitignore).

Future Improvements
Hyperparameter tuning for models like XGBoost and Random Forest.
Experiment with other advanced NLP techniques like BERT.
Add more sentiment categories (beyond binary classification).