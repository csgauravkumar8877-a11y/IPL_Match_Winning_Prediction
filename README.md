# 🏏 IPL Match Winning Probability Prediction Using Artificial Neural Network (ANN)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-ANN-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-yellow)
![Status](https://img.shields.io/badge/Project-Development-green)

## 📌 Project Overview

This project aims to predict the winning probability of an IPL cricket team during the second innings using an Artificial Neural Network (ANN).

The model analyzes historical IPL ball-by-ball cricket data and uses match conditions such as current score, target score, wickets lost, remaining balls, current run rate, and required run rate to estimate the probability of the chasing team winning.

The project uses Python, Data Science, Deep Learning, TensorFlow, and Keras.

## 🎯 Project Objectives

- Collect and analyze historical IPL cricket data.
- Perform data cleaning and preprocessing.
- Conduct Exploratory Data Analysis (EDA).
- Perform feature engineering.
- Develop an Artificial Neural Network.
- Predict the winning probability of the chasing team.
- Evaluate model performance using classification metrics.
- Generate predictions for different cricket match situations.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Google Colab | Development environment |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Data preprocessing and evaluation |
| TensorFlow | Deep Learning |
| Keras | Neural Network development |
| Cricsheet | Historical cricket dataset |

## 📂 Dataset Description

**Dataset Source:** [Cricsheet IPL Dataset](https://cricsheet.org/downloads/)

The project uses historical IPL ball-by-ball data in JSON format.

Dataset information includes:

- Match ID
- IPL Season
- Venue
- Batting Team
- Bowling Team
- Batsman
- Bowler
- Overs
- Runs Scored
- Extras
- Wickets
- Match Winner

The JSON files are converted into structured Pandas DataFrames for analysis and model development.

## 🔄 Project Workflow

1. Dataset Collection
2. JSON Data Extraction
3. Data Cleaning
4. Exploratory Data Analysis (EDA)
5. Feature Engineering
6. Data Preprocessing
7. Training and Testing Data Splitting
8. Artificial Neural Network Development
9. Model Training
10. Model Evaluation
11. Match Winning Probability Prediction

## 📊 Exploratory Data Analysis (EDA)

EDA helps understand historical IPL match patterns and identify relationships between match conditions and winning outcomes.

The analysis includes:

- Distribution of match outcomes.
- Runs remaining versus match outcomes.
- Wickets lost versus winning probability.
- Correlation between numerical features.
- Analysis of second-innings match situations.

Visualization libraries:

- Matplotlib
- Seaborn

## ⚙️ Feature Engineering

The following features are created for the ANN model:

| Feature | Description |
|---|---|
| Season | IPL season |
| Venue | Cricket stadium |
| Batting Team | Team chasing the target |
| Bowling Team | Team defending the target |
| Target | Total runs required to win |
| Current Score | Runs scored by the chasing team |
| Wickets Lost | Number of wickets lost |
| Legal Balls Bowled | Legal deliveries completed |
| Balls Remaining | Remaining legal deliveries |
| Runs Remaining | Runs required to win |
| Current Run Rate | Current scoring rate |
| Required Run Rate | Scoring rate required to win |

### Important Calculations

**Current Run Rate (CRR)**

CRR = Current Runs / Overs Completed

**Required Run Rate (RRR)**

RRR = Runs Remaining / Overs Remaining

**Runs Remaining**

Runs Remaining = Target Score - Current Score

**Balls Remaining**

Balls Remaining = 120 - Legal Balls Bowled

## 🧠 Artificial Neural Network (ANN) Architecture

The project uses an Artificial Neural Network to predict the probability of the chasing team winning.

### Model Architecture

| Layer | Neurons | Activation Function |
|---|---:|---|
| Input Layer | Based on processed features | — |
| Hidden Layer 1 | 128 | ReLU |
| Dropout | — | 30% |
| Hidden Layer 2 | 64 | ReLU |
| Dropout | — | 20% |
| Hidden Layer 3 | 32 | ReLU |
| Output Layer | 1 | Sigmoid |

### Model Configuration

- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Output Activation: Sigmoid
- Batch Size: 256
- Maximum Epochs: 40
- Early Stopping: Enabled
- Learning Rate Reduction: Enabled

### Prediction Target

The target variable is `chase_win`.

- `1` = Chasing team wins.
- `0` = Chasing team does not win.

The sigmoid output represents the estimated probability of the chasing team winning.

## 📈 Model Evaluation

The model is evaluated using:

| Metric | Purpose |
|---|---|
| Accuracy | Overall correct predictions |
| Precision | Correct positive predictions |
| Recall | Identification of actual positive outcomes |
| F1-Score | Balance between precision and recall |
| ROC-AUC | Ability to distinguish between two classes |
| Confusion Matrix | Comparison of actual and predicted outcomes |

Actual performance results will be added after model training.

## 🔮 Example Prediction

Consider a hypothetical IPL match situation:

| Match Information | Value |
|---|---:|
| Batting Team | Chennai Super Kings |
| Bowling Team | Mumbai Indians |
| Target Score | 181 |
| Current Score | 145 |
| Wickets Lost | 4 |
| Overs Completed | 15 |
| Runs Remaining | 36 |
| Balls Remaining | 30 |

The trained ANN model estimates:

- Chasing team's winning probability.
- Defending team's winning probability.

*Note: This is an illustrative example. Actual predictions are generated by the trained model.*

## 🚀 How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/IPL-Match-Winning-Prediction-ANN.git
