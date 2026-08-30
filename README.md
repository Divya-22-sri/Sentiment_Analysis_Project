# 🎬 IMDb Movie Review Sentiment Analysis

##  Project Overview

This project is a **Natural Language Processing (NLP) and Machine Learning** application that analyzes IMDb movie reviews and classifies them into two sentiment categories:

* 😊 Positive
* 😞 Negative

The system uses **TF-IDF (Term Frequency-Inverse Document Frequency)** for text feature extraction and compares multiple machine learning algorithms to identify the best-performing model.

The final model is a **Tuned Linear Support Vector Machine (SVM)**, which achieved an accuracy of approximately **91.02%** on the test dataset.

---

##  Objectives

* Perform sentiment analysis on movie reviews.
* Clean and preprocess textual data.
* Convert text into numerical features using TF-IDF.
* Train and compare multiple machine learning models.
* Tune the best-performing model.
* Evaluate the final model using accuracy, precision, recall and F1-score.
* Build an interactive Streamlit web application for real-time predictions.

---

##  Dataset

The project uses the **IMDb Movie Review Dataset**, containing movie reviews labeled as either positive or negative.

### Dataset Information

* Total original reviews: **50,000**
* Reviews after duplicate removal: **49,582**
* Positive reviews: **24,884**
* Negative reviews: **24,698**
* Number of classes: **2**

The dataset is not included in this repository. Download the IMDb dataset separately and place the CSV file inside the `dataset` folder.

Expected file:

```text
dataset/
└── IMDB Dataset.csv
```

---

##  Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **NLTK/NLP techniques**
* **TF-IDF Vectorization**
* **Logistic Regression**
* **Multinomial Naive Bayes**
* **Linear SVM**
* **Random Forest**
* **Joblib**
* **Streamlit**
* **Matplotlib**
* **Seaborn**
* **WordCloud**

---

##  Project Workflow

```text
IMDb Dataset
      ↓
Data Loading
      ↓
Data Inspection
      ↓
Duplicate Removal
      ↓
Exploratory Data Analysis
      ↓
Text Cleaning
      ↓
TF-IDF Feature Extraction
      ↓
Train/Test Split
      ↓
Model Training
      ↓
Model Comparison
      ↓
Hyperparameter Tuning
      ↓
Final Tuned Linear SVM
      ↓
Model Evaluation
      ↓
Model Saving
      ↓
Streamlit Application
```

---

##  Data Preprocessing

The reviews are cleaned before machine learning.

The preprocessing includes:

* Converting text to lowercase
* Removing HTML tags
* Removing URLs
* Removing special characters
* Removing unnecessary spaces

Example:

```text
Original:
A wonderful movie! <br /><br>Great acting...

Cleaned:
a wonderful movie great acting
```

---

##  Machine Learning Models

Four machine learning algorithms were evaluated:

1. Logistic Regression
2. Multinomial Naive Bayes
3. Linear SVM
4. Random Forest

### Model Comparison

| Model               | Accuracy | Precision | Recall | F1 Score |
| ------------------- | -------: | --------: | -----: | -------: |
| Logistic Regression |   90.72% |    89.58% | 92.24% |   90.89% |
| Naive Bayes         |   88.32% |    87.97% | 88.89% |   88.43% |
| Linear SVM          |   90.79% |    90.35% | 91.42% |   90.88% |
| Random Forest       |   83.90% |    84.42% | 83.28% |   83.85% |

---

## Model Tuning

Linear SVM achieved the best initial performance, so hyperparameter tuning was performed using GridSearchCV.

### Best Parameter

```text
C = 0.5
```

### Tuned Linear SVM Performance

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **91.02%** |
| Precision | **90.46%** |
| Recall    | **91.78%** |
| F1 Score  | **91.11%** |

The tuned model improved the original Linear SVM performance.

---

##  Evaluation

The final model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Classification Report
* Error Analysis

The confusion matrix and other visualizations are available in the `images` folder.

---

##  Exploratory Data Analysis

The project includes visual analysis such as:

* Sentiment distribution
* Review length distribution
* Review length comparison
* Positive word cloud
* Negative word cloud
* Confusion matrix

---

##  Streamlit Application

The project includes an interactive web application built using Streamlit.

The user can enter an IMDb-style movie review, and the application predicts whether the review is:

```text
😊 POSITIVE
```

or

```text
😞 NEGATIVE
```

### Application Flow

```text
User Review
     ↓
Text Cleaning
     ↓
TF-IDF Vectorization
     ↓
Tuned Linear SVM
     ↓
Sentiment Prediction
```

---

## Project Structure

```text
Sentiment_Analysis_Project/
│
├── app/
│   └── app.py
│
├── dataset/
│   └── IMDB Dataset.csv
│
├── models/
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebook/
│   └── Sentiment_Analysis.ipynb
│
├── images/
│   ├── positive_wordcloud.png
│   ├── negative_wordcloud.png
│   ├── sentiment_distribution.png
│   ├── review_length_distribution.png
│   ├── review_length_boxplot.png
│   └── final_confusion_matrix.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

##  Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Sentiment_Analysis_Project.git
```

Move into the project directory:

```bash
cd Sentiment_Analysis_Project
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

---

##  Run the Application

From the project root directory, run:

```bash
python -m streamlit run app/app.py
```

The application will open in your browser.

---

##  Example

### Positive Review

```text
This movie was fantastic. The acting was brilliant and the story was very enjoyable.
```

Expected:

```text
😊 POSITIVE SENTIMENT
```

### Negative Review

```text
The movie was boring and disappointing. The story was poorly written.
```

Expected:

```text
😞 NEGATIVE SENTIMENT
```

---

## ⚠️ Limitations

* The model performs **binary sentiment classification** only.
* It predicts either positive or negative sentiment.
* A neutral class is not included in the training dataset.
* Sarcasm and highly ambiguous reviews may be difficult to classify.
* Performance depends on the language and style of the input review.

---

## 🚀 Future Enhancements

Possible future improvements include:

* Add a neutral sentiment class using an appropriate labeled dataset.
* Experiment with deep learning models such as LSTM or Transformers.
* Support multiple languages.
* Improve handling of sarcasm and context-dependent language.
* Deploy the Streamlit application online.
* Add sentiment probability estimation using a calibrated classifier.

---

## 👩‍💻 Project Information

**Project:** IMDb Movie Review Sentiment Analysis

**Domain:** Natural Language Processing / Machine Learning

**Task:** Binary Sentiment Classification

**Final Model:** Tuned Linear SVM

**Feature Extraction:** TF-IDF

**Best Accuracy:** **91.02%**

**Application:** Streamlit
