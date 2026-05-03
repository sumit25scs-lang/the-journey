import json
import os

notebook = {
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# 🎓 Retenza - Student Dropout Risk Prediction\n",
        "\n",
        "This notebook explores the Machine Learning pipeline (Random Forest) for the Retenza dropout prediction system. It includes data synthesis, model training, evaluation metrics, and the prediction engine logic from the Flask backend."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import warnings\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score\n",
        "\n",
        "warnings.filterwarnings('ignore')\n",
        "%matplotlib inline\n",
        "sns.set_theme(style='darkgrid')"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 1. Feature Definition\n",
        "Defining the predictive features utilized by the model to determine student dropout risk."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "FEATURES = [\n",
        "    'completion_rate_1', 'completion_rate_2',\n",
        "    'approved_1sem',     'approved_2sem',\n",
        "    'grade_1sem',        'grade_2sem',\n",
        "    'tuition_fees',      'financial_risk',\n",
        "    'scholarship',       'debt',\n",
        "    'attendance',        'study_hrs',\n",
        "    'exam_prep',         'employed',\n",
        "    'childcare',         'displaced',\n",
        "    'age',\n",
        "]\n",
        "\n",
        "FEATURE_DISPLAY_NAMES = {\n",
        "    'completion_rate_2': 'Completion Rate (Sem 2)',\n",
        "    'completion_rate_1': 'Completion Rate (Sem 1)',\n",
        "    'approved_2sem':     'Approved Courses (Sem 2)',\n",
        "    'grade_2sem':        'Average Grades (Sem 2)',\n",
        "    'grade_1sem':        'Average Grades (Sem 1)',\n",
        "    'financial_risk':    'Financial Risk',\n",
        "    'tuition_fees':      'Tuition Fees Status',\n",
        "    'attendance':        'Class Attendance',\n",
        "    'approved_1sem':     'Approved Courses (Sem 1)',\n",
        "    'study_hrs':         'Weekly Study Hours',\n",
        "    'age':               'Age of Enrollment',\n",
        "    'scholarship':       'Scholarship Holder',\n",
        "    'exam_prep':         'Exam Prep Level',\n",
        "    'employed':          'Employment Status',\n",
        "    'debt':              'Outstanding Debt',\n",
        "    'childcare':         'Childcare Responsibilities',\n",
        "    'displaced':         'Displaced / Refugee Status',\n",
        "}"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 2. Synthetic Data Generation\n",
        "Generating a realistic 3,630-record dataset. Dropouts and Current/Graduated students are modeled with distinct multi-variate statistical distributions representing known real-world patterns."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "def generate_training_data(n_total: int = 3630, dropout_rate: float = 0.3915, seed: int = 42) -> pd.DataFrame:\n",
        "    rng = np.random.default_rng(seed)\n",
        "    n_out = int(n_total * dropout_rate)   # ~1421 dropouts\n",
        "    n_in  = n_total - n_out               # ~2209 enrolled\n",
        "\n",
        "    def make_class(n, dropout: bool) -> pd.DataFrame:\n",
        "        if dropout:\n",
        "            cr1  = np.clip(rng.beta(2,   5,   n), 0.0, 1.0)\n",
        "            cr2  = np.clip(rng.beta(1.5, 5,   n), 0.0, 1.0)\n",
        "            ap1  = np.clip(rng.poisson(2.0, n),   0, 8)\n",
        "            ap2  = np.clip(rng.poisson(1.5, n),   0, 8)\n",
        "            g1   = np.clip(rng.normal(9.0, 3.0, n),  0, 20)\n",
        "            g2   = np.clip(rng.normal(8.0, 3.0, n),  0, 20)\n",
        "            tuit = rng.binomial(1, 0.25, n)\n",
        "            finr = rng.binomial(1, 0.55, n)\n",
        "            sch  = rng.binomial(1, 0.10, n)\n",
        "            debt = rng.binomial(1, 0.50, n)\n",
        "            att  = rng.choice([1,2,3,4], n, p=[0.35,0.35,0.20,0.10])\n",
        "            stdy = rng.choice([1,2,3,4], n, p=[0.40,0.35,0.18,0.07])\n",
        "            exam = rng.choice([1,2,3],   n, p=[0.50,0.35,0.15])\n",
        "            emp  = rng.choice([0,1,2],   n, p=[0.25,0.40,0.35])\n",
        "            chld = rng.binomial(1, 0.30, n)\n",
        "            disp = rng.binomial(1, 0.15, n)\n",
        "            age  = np.clip(rng.normal(26, 7, n), 17, 60)\n",
        "        else:\n",
        "            cr1  = np.clip(rng.beta(8, 2, n),   0.0, 1.0)\n",
        "            cr2  = np.clip(rng.beta(8, 2, n),   0.0, 1.0)\n",
        "            ap1  = np.clip(rng.poisson(5, n)+2,  0, 8)\n",
        "            ap2  = np.clip(rng.poisson(5, n)+2,  0, 8)\n",
        "            g1   = np.clip(rng.normal(14.0, 2.5, n), 0, 20)\n",
        "            g2   = np.clip(rng.normal(14.0, 2.5, n), 0, 20)\n",
        "            tuit = rng.binomial(1, 0.88, n)\n",
        "            finr = rng.binomial(1, 0.10, n)\n",
        "            sch  = rng.binomial(1, 0.35, n)\n",
        "            debt = rng.binomial(1, 0.10, n)\n",
        "            att  = rng.choice([1,2,3,4], n, p=[0.05,0.15,0.40,0.40])\n",
        "            stdy = rng.choice([1,2,3,4], n, p=[0.05,0.20,0.45,0.30])\n",
        "            exam = rng.choice([1,2,3],   n, p=[0.10,0.40,0.50])\n",
        "            emp  = rng.choice([0,1,2],   n, p=[0.55,0.35,0.10])\n",
        "            chld = rng.binomial(1, 0.10, n)\n",
        "            disp = rng.binomial(1, 0.05, n)\n",
        "            age  = np.clip(rng.normal(21, 3, n), 17, 40)\n",
        "\n",
        "        return pd.DataFrame({\n",
        "            'completion_rate_1': cr1, 'completion_rate_2': cr2,\n",
        "            'approved_1sem': ap1.astype(int), 'approved_2sem': ap2.astype(int),\n",
        "            'grade_1sem': g1, 'grade_2sem': g2,\n",
        "            'tuition_fees': tuit, 'financial_risk': finr,\n",
        "            'scholarship': sch, 'debt': debt,\n",
        "            'attendance': att, 'study_hrs': stdy,\n",
        "            'exam_prep': exam, 'employed': emp,\n",
        "            'childcare': chld, 'displaced': disp,\n",
        "            'age': age,\n",
        "            'dropout': int(dropout),\n",
        "        })\n",
        "\n",
        "    df = pd.concat([make_class(n_out, True), make_class(n_in, False)], ignore_index=True)\n",
        "    return df.sample(frac=1, random_state=seed).reset_index(drop=True)\n",
        "\n",
        "df = generate_training_data()\n",
        "print(f\"Dataset Shape: {df.shape}\")\n",
        "df.head()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 3. Exploratory Data Analysis\n",
        "Let's visualize the class distribution and some key feature differences."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "plt.figure(figsize=(12, 5))\n",
        "\n",
        "plt.subplot(1, 2, 1)\n",
        "sns.countplot(data=df, x='dropout', palette=['#10b981', '#ef4444'])\n",
        "plt.title('Class Distribution (0=Enrolled, 1=Dropout)')\n",
        "\n",
        "plt.subplot(1, 2, 2)\n",
        "sns.boxplot(data=df, x='dropout', y='completion_rate_2', palette=['#10b981', '#ef4444'])\n",
        "plt.title('Completion Rate (Sem 2) by Target Class')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 4. Model Training\n",
        "Training a Random Forest Classifier. We use an 80/20 train-test split."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "X, y = df[FEATURES], df['dropout']\n",
        "\n",
        "X_train, X_test, y_train, y_test = train_test_split(\n",
        "    X, y, test_size=0.20, random_state=42, stratify=y\n",
        ")\n",
        "\n",
        "rf = RandomForestClassifier(\n",
        "    n_estimators=200,\n",
        "    max_depth=15,\n",
        "    min_samples_split=5,\n",
        "    min_samples_leaf=2,\n",
        "    max_features='sqrt',\n",
        "    class_weight='balanced',\n",
        "    random_state=42,\n",
        "    n_jobs=-1,\n",
        ")\n",
        "rf.fit(X_train, y_train)\n",
        "\n",
        "y_pred = rf.predict(X_test)\n",
        "y_prob = rf.predict_proba(X_test)[:, 1]\n",
        "\n",
        "print(\"Model Evaluation Metrics on Test Set:\\n\")\n",
        "print(f\"Accuracy:  {accuracy_score(y_test, y_pred):.4f}\")\n",
        "print(f\"AUC Score: {roc_auc_score(y_test, y_prob):.4f}\")\n",
        "print(f\"Precision: {precision_score(y_test, y_pred):.4f}\")\n",
        "print(f\"Recall:    {recall_score(y_test, y_pred):.4f}\")\n",
        "print(f\"F1 Score:  {f1_score(y_test, y_pred):.4f}\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 5. Feature Importances\n",
        "Visualizing which features the model relies on the most."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "importances = rf.feature_importances_\n",
        "indices = np.argsort(importances)[::-1]\n",
        "\n",
        "plt.figure(figsize=(10, 6))\n",
        "plt.title('Top Predictive Features for Student Dropout in Retenza')\n",
        "sns.barplot(x=importances[indices], y=[FEATURE_DISPLAY_NAMES.get(FEATURES[i], FEATURES[i]) for i in indices], palette='viridis')\n",
        "plt.xlabel('Relative Importance')\n",
        "plt.tight_layout()\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 6. Real-time Prediction and Intervention Logic\n",
        "Below is an example of querying the trained model with a new student's data."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Simulating an at-risk student input\n",
        "sample_student = {\n",
        "    'completion_rate_1': 0.45, 'completion_rate_2': 0.30,\n",
        "    'approved_1sem': 3, 'approved_2sem': 1,\n",
        "    'grade_1sem': 9.5, 'grade_2sem': 7.0,\n",
        "    'tuition_fees': 0, 'financial_risk': 1,\n",
        "    'scholarship': 0, 'debt': 1,\n",
        "    'attendance': 2, 'study_hrs': 1,\n",
        "    'exam_prep': 1, 'employed': 2,\n",
        "    'childcare': 1, 'displaced': 0,\n",
        "    'age': 25\n",
        "}\n",
        "\n",
        "feat_df = pd.DataFrame([sample_student])\n",
        "\n",
        "# 1. Get Probability Score\n",
        "prob = rf.predict_proba(feat_df)[0][1]\n",
        "risk = 'High' if prob >= 0.6 else 'Medium' if prob >= 0.4 else 'Low'\n",
        "\n",
        "print(f\"Student Dropout Risk Score: {prob*100:.1f}%\")\n",
        "print(f\"Risk Category: {risk}\")\n"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.10.0"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 4
}

with open('student_dropout_predictor.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print("Notebook generated successfully!")
