# 🏦 Smart Loan Recovery System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2+-orange)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.22+-red)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive machine learning system that predicts borrower default risk and recommends optimal recovery strategies for financial institutions. Built with Python, Scikit-Learn, and Streamlit.

![Demo]("C:\Users\Vidya\venv\Videos\Captures\Smart Loan Recovery System and 1 more page - Personal - Microsoft​ Edge 2025-09-08 11-46-27.mp4")


## ✨ Features

### 🔍 Risk Analysis
- **Predictive Modeling**: Random Forest classifier for accurate default risk prediction
- **Risk Segmentation**: K-Means clustering to categorize borrowers into risk groups
- **Real-time Scoring**: Instant risk assessment with probability scores

### 📊 Interactive Dashboard
- **Single Borrower Assessment**: Individual risk evaluation with detailed insights
- **Batch Processing**: CSV upload for multiple borrower analysis
- **Visual Analytics**: Comprehensive charts and data visualizations
- **Export Capabilities**: Download results in CSV format

### 🎯 Smart Recommendations
- **Personalized Strategies**: Tailored recovery plans based on risk levels
- **Three-Tier System**: 
  - 🔴 High Risk: Immediate legal actions
  - 🟡 Medium Risk: Settlement offers & repayment plans
  - 🟢 Low Risk: Automated reminders & monitoring

## 🎥 Demo

### Video Demonstration
[![Watch the demo](https://img.shields.io/badge/YouTube-Demo_Video-red?style=for-the-badge&logo=youtube)](https://youtube.com) *Replace with your actual video link*


## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- Git

### Local Development Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Vinay-014/Smart-Loan-Recovery-System.git
   cd Smart-Loan-Recovery-System
2. **Using venv**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   # Or using conda
   conda create -n loan-recovery python=3.9
   conda activate loan-recovery
3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
4. **Model Training & Analysis**
   ```bash
   # Launch Jupyter Notebook
   jupyter notebook notebooks/Smart_Loan_Recovery.ipynb

   # Run all cells to:
   # - Preprocess data
   # - Train ML models
   # - Generate insights
   # - Save trained models
5. **Launch Web Application**
   ```bash
   streamlit run app/loan_recovery_app.py
   #The application will open at http://localhost:8501


## 🤖 Machine Learning Pipeline

### 📊 Overview
End-to-end machine learning workflow for predicting loan default risk and optimizing recovery strategies.

### 🔧 Data Preprocessing
- **Handled missing values** with median/mode imputation
- **Encoded categorical variables** (Gender, Employment Type, Payment History)
- **Scaled numerical features** using StandardScaler
- **Created train/test split** (80/20 ratio)

### ⚙️ Feature Engineering
- **DTI_Ratio**: Debt-to-Income ratio calculation
- **Payment_Score**: Behavioral scoring based on payment history
- **Collateral_Coverage**: Loan security coverage ratio
- **Risk flags** for high DTI and multiple missed payments

### 🧠 Model Training
**Algorithms Used:**
- **Random Forest Classifier** (Primary model)
- **K-Means Clustering** for borrower segmentation
- **Hyperparameter tuning** with GridSearchCV

### 📈 Model Performance
**Evaluation Metrics:**
| Metric | Score |
|--------|-------|
| **Accuracy** | 85% |
| **Precision** | 82% |
| **Recall** | 79% |
| **F1-Score** | 80% |
| **ROC AUC** | 89% |

### 🎯 Key Features
**Top Predictive Factors:**
1. Number of Missed Payments (24% importance)
2. Payment History (18% importance) 
3. Debt-to-Income Ratio (15% importance)
4. Monthly Income (12% importance)
5. Days Past Due (9% importance)

### 🚀 Deployment
- **Model saved as** `.pkl` files for production use
- **Streamlit web interface** for real-time predictions
- **Batch processing** capabilities for multiple borrowers
- **REST API ready** for integration with other systems










