# Spam Email Classifier

ML Assignment 2 (M.Tech AIML, BITS WILP)

Live app: 
GitHub: 

## a. Problem statement

The goal is to predict whether an email is spam or not spam from numbers that describe the email (how often certain words and symbols appear). I trained five different classifiers on the same data, compared them using six metrics, and built a small Streamlit app so any of the models can be tried on a test CSV.

## b. Dataset description

I used the Spambase dataset from the UCI repository.

- 4601 emails and 57 input features
- Target column is `is_spam`: 1 = spam (1813 emails), 0 = not spam (2788 emails)
- No missing values and no text columns, so no encoding was needed
- Features are of three kinds:
  - 48 columns for how often a particular word appears (for example "free" or "money")
  - 6 columns for how often a symbol appears (`;`, `(`, `[`, `!`, `$`, `#`)
  - 3 columns about runs of capital letters (average, longest, total)

I split the data 80/20 with a fixed random seed, keeping the spam ratio the same in both parts. The inputs were standardised with a scaler fitted on the training part only. The test part is saved as `test_data.csv` and is what gets uploaded to the app.

## c. GitHub repository

<PASTE GITHUB LINK>

It contains `app.py`, `requirements.txt`, this README, `test_data.csv`, and the `model/` folder with the training script, the dataset, the scaler and the five saved models.

## d. Models used

All five models use the same training data and are tested on the same 921 test emails. Precision, recall and F1 are for the spam class.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9294 | 0.9702 | 0.9209 | 0.8981 | 0.9093 | 0.8518 |
| Decision Tree | 0.9110 | 0.9078 | 0.8828 | 0.8926 | 0.8877 | 0.8140 |
| KNN | 0.9077 | 0.9506 | 0.8861 | 0.8788 | 0.8824 | 0.8065 |
| Naive Bayes | 0.8328 | 0.9376 | 0.7146 | 0.9587 | 0.8188 | 0.6946 |
| Support Vector Machine | 0.9273 | 0.9667 | 0.9277 | 0.8843 | 0.9055 | 0.8471 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Gave the best accuracy, AUC and MCC. Its training accuracy (0.9307) is almost the same as its test accuracy, so it is not overfitting. |
| Decision Tree | Got about 99.97% on the training data but only 91.1% on the test data, so it memorised the training set. It also had the lowest AUC. |
| KNN | Decent results, but a bit behind logistic regression and SVM. It needs scaled data and keeps the whole training set, so it is slower to predict. |
| Naive Bayes | Lowest accuracy and MCC. Its recall is very high (0.9587) but precision is low (0.7146), meaning it catches nearly all spam but also marks many normal emails as spam. This is probably because the features are not independent of each other. |
| Support Vector Machine | Almost as good as logistic regression, and it has the best precision (0.9277), so it makes the fewest mistakes on normal emails. |
| Overall winner | Logistic Regression, since it is best on accuracy, AUC and MCC and does not overfit. SVM is very close, so the difference may just be noise on this test set. |

## e. Streamlit app

The app lets you upload a CSV (the test data), choose a model from a dropdown, and see the six metrics along with a confusion matrix and classification report. It loads the saved models and does not retrain.

## f. How to run

```
pip install -r requirements.txt
python3 model/model.py
streamlit run app.py
```