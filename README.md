# 📰 Fake News Detection System

## 📌 Project Overview

This project focuses on building an **Improved Fake News Detection System** using **Machine Learning and Deep Learning techniques**. The goal is to classify news articles as **Fake** or **Real** by analyzing textual content. The system applies data preprocessing, feature extraction, class balancing, and multiple models to achieve reliable performance.

The implementation is provided in a Jupyter Notebook: **`Fake new Detection.ipynb`**.

---

## 🚀 Features

* Text preprocessing (cleaning, tokenization, stopword removal)
* Exploratory Data Analysis (EDA)
* Data balancing using **imbalanced-learn**
* Machine Learning & Deep Learning models
* Performance evaluation using multiple metrics
* Visualization using **Matplotlib**, **Seaborn**, and **WordCloud**

---

## 🧠 Models Used

The project experiments with multiple models to compare performance:

### 🔹 Machine Learning

* Random Forest Classifier

### 🔹 Deep Learning

* Convolutional Neural Network (CNN)
* Long Short-Term Memory Network (LSTM)

These models help capture both statistical and sequential patterns in textual data.

---

## 🛠️ Tech Stack & Libraries

* **Python 3.x**
* **Pandas, NumPy** – Data handling
* **NLTK** – Text preprocessing
* **Scikit-learn** – ML models & evaluation
* **TensorFlow / Keras** – Deep Learning models
* **Imbalanced-learn** – Handling class imbalance
* **Matplotlib, Seaborn** – Data visualization
* **WordCloud** – Text visualization

---

## 📂 Project Structure

```
Fake-News-Detection/
│
├── Fake new Detection.ipynb   # Main notebook
├── README.md                 # Project documentation
└── dataset/                  # (Optional) Dataset files
```

---

## 📊 Workflow

1. **Data Loading** – Load and inspect the dataset
2. **Data Cleaning** – Remove noise, punctuation, and stopwords
3. **Exploratory Data Analysis (EDA)** – Visual insights and word distributions
4. **Text Vectorization** – Convert text into numerical representations
5. **Handling Imbalanced Data** – Apply resampling techniques
6. **Model Training** – Train ML and DL models
7. **Evaluation** – Accuracy, Precision, Recall, F1-score
8. **Comparison** – Analyze model performance

---
<img width="1852" height="752" alt="image" src="https://github.com/user-attachments/assets/a62920c0-e5cf-463b-9d64-647e539ce299" />


## 📈 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

These metrics ensure a balanced and fair evaluation of fake news classification.

---

## ▶️ How to Run

1. Clone the repository

```bash
git clone <repository-url>
```

2. Install required libraries

```bash
pip install -r requirements.txt
```

3. Open the notebook

```bash
jupyter notebook
```

4. Run **`Fake new Detection.ipynb`** cell by cell

---

## 📌 Results

The models demonstrate that **Deep Learning approaches (CNN/LSTM)** are more effective in capturing contextual information compared to traditional ML models, especially on complex text patterns.

---

## 🔮 Future Improvements

* Use transformer-based models (BERT, RoBERTa)
* Deploy as a web application using Flask or FastAPI
* Add real-time news scraping
* Improve dataset size and diversity

---

## 👩‍💻 Author

**Mariam Tariq**
BS Computer Science – COMSATS University Islamabad

---

## 📜 License

This project is for **educational and research purposes only**.

---

⭐ *If you find this project helpful, feel free to give it a star!*
