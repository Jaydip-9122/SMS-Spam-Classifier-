# SMS-Spam-Classifier-
Automatically classify SMS messages as 'spam' or 'ham' (not spam) using machine learning to reduce manual effort and protect users from fraudulent and unwanted messages.


## Overview
This project implements an **SMS Spam Classifier** using Machine Learning to automatically classify SMS messages as 'spam' or 'ham' (not spam), helping users avoid fraudulent and unwanted messages.

## Dataset
- **File:** spam.csv
- **Source:** UCI SMS Spam Collection
- **Description:** 5,574 SMS messages tagged as `ham` or `spam`.

## Workflow

1. **Data Collection:** Load and clean dataset, convert labels (`ham` to 0, `spam` to 1).
2. **Preprocessing:** Lowercasing, punctuation removal, stopword removal, lemmatization.
3. **Feature Extraction:** TF-IDF Vectorization.
4. **Train-Test Split:** 80% train, 20% test with stratified split.
5. **Model Training:** Multinomial Naive Bayes (selected for performance on text).
6. **Evaluation:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC-AUC.

## Results
- **Accuracy:** ~98%
- **High Precision and Recall**
- **Low False Positives and False Negatives**
- Confusion Matrix:
  ```
  [[965, 1],
   [18, 131]]
  ```

## Visualizations
- Word clouds for spam and ham messages.
- Confusion matrix plots.

## Future Work
- Deploy using Streamlit or Flask for live testing.
- Extend to multilingual spam classification.
- Test advanced models (LSTM/Transformers).

## Requirements
- Python (>=3.8)
- pandas
- numpy
- scikit-learn
- nltk
- matplotlib
- seaborn

## How to Run

1. Clone this repository.
2. Install requirements:
   ```
   pip install -r requirements.txt
   ```
3. Run the notebook `SMS SPAM CLASSIFIER.ipynb` or execute the Python script if provided.
4. Adjust and test with your own SMS samples.

## License
MIT License.

