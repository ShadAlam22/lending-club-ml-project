

#  Lending Club Loan Analysis & Optimization Project

### **Policy Optimization for Financial Decision-Making**

This project applies advanced **Machine Learning (ML)** and **Offline Reinforcement Learning (RL)** techniques to the **LendingClub Loan Dataset (2007–2018)** to optimize data-driven loan approval policies.

Instead of focusing only on predicting loan defaults, this project aims to **maximize long-term financial value** by training an AI agent capable of selecting profitable loans while avoiding high-risk ones.

Dataset- https://www.kaggle.com/datasets/wordsforthewise/lending-club?resource=download

---


##  Project Structure

```bash
lending-club-ml-project/
│
├── notebooks/                     
│   ├── 1_data_preprocessing.ipynb        # Data cleaning & feature engineering
│   ├── 2_exploratory_analysis.ipynb      # EDA & statistical tests
│   ├── 3_deep_learning_model.ipynb       # PyTorch MLP training (imbalanced data)
│   ├── 4_reinforcement_learning.ipynb    # Offline RL (CQL) training
│   │
│   ├── models/
│   │   ├── loan_default_dl_model.pth     # Saved Deep Learning model
│   │   └── q_network_model.pth           # Trained RL Q-network
│   │
│   └── results/
│       ├── dl_model_results_improved.csv
│       ├── eda_summary_statistics.csv
│       └── rl_model_results.csv
│
├── Project Report.pdf
├── requirements.txt
└── README.md
```

---

#  Analysis & Methodology

---

## **1. Data Preprocessing**

**Notebook:** `1_data_preprocessing.ipynb`

* Dataset: **LendingClub Accepted Loans 2007–2018 Q4**
* Removed columns with **>40% missing values**
* Cleaned remaining rows with missing data
* Selected **21 key borrower and loan features**
* Binary target:

  * **0 = Fully Paid**
  * **1 = Defaulted**
* Strictly removed **post-origination features** to avoid data leakage
* Created ML-ready dataset with balanced and standardized features

---

## **2. Exploratory Data Analysis**

**Notebook:** `2_exploratory_analysis.ipynb`

Analysis of **1.2M+ loans** revealed significant financial patterns.

### Key Insights

| Metric        | Fully Paid (Avg) | Defaulted (Avg) | Insight                           |
| ------------- | ---------------- | --------------- | --------------------------------- |
| Loan Amount   | $14,283          | $15,858         | Defaults have higher loan amounts |
| Interest Rate | 12.62%           | 15.75%          | Riskier loans have higher APR     |
| Annual Income | $79,197          | $72,432         | Lower income → higher risk        |
| DTI Ratio     | 17.68            | 19.96           | High debt burden → higher default |

 Statistical tests confirmed differences (**p < 0.05**)

---

## **3. Deep Learning Model (PyTorch MLP)**

**Notebook:** `3_deep_learning_model.ipynb`

### **Architecture**

```
Input → 256 → 128 → 64 → 32 → Output
```

### **Techniques Used**

* **Focal Loss** to prioritize hard-to-classify defaults
* **SMOTE** oversampling to address 4:1 imbalance
* **Class Weights** to penalize false negatives

### **Performance (Optimal Threshold = 0.45)**

* **Accuracy:** 77.74%
* **AUC-ROC:** 0.716
* **F1-Score:** 0.426
* **Recall:** ~63% (important for capturing high-risk loans)

---

## **4. Reinforcement Learning (Conservative Q-Learning)**

**Notebook:** `4_reinforcement_learning.ipynb`

Traditional models treat all errors equally—but in lending:

* **False Negative = Approve a default → huge loss**
* **False Positive = Reject a good loan → small opportunity cost**

 **CQL was used to optimize decisions for financial reward, not accuracy.**

### **Reward Function**

* **If paid back:** `+ interest profit`
* **If defaulted:** `- principal loss`

### **Policy Evaluation**

| Policy                 | Approval Rate | Avg Value per Loan | Net Improvement  |
| ---------------------- | ------------- | ------------------ | ---------------- |
| Baseline (Approve All) | 100%          | **-$1,622.30**     | N/A              |
| RL Agent (CQL)         | 13.10%        | **-$29.06**        | +$1,593 per loan |

###  Impact

* RL agent becomes **highly selective**, approving only safest loans
* Massive reduction in financial losses
* Demonstrates RL superiority for **risk-sensitive financial decisions**

---

#  Getting Started

### **1️ Clone the Repository**

```bash
git clone https://github.com/ShadAlam22/lending-club-ml-project.git
cd lending-club-ml-project
```

### **2️ Install Dependencies**

```bash
pip install -r requirements.txt
```

### **3️Run the Notebooks (1 → 4)**

Navigate to the `notebooks/` directory:

```
1_data_preprocessing.ipynb  
2_exploratory_analysis.ipynb  
3_deep_learning_model.ipynb  
4_reinforcement_learning.ipynb  
```

Each notebook contains step-by-step experiments to fully reproduce the results.

---

#  Project Highlights

* End-to-end ML & RL financial decision system
* Real-world scale dataset (1.3M+ loans)
* Demonstrates application of **offline RL for credit risk**
* Includes clean notebooks, saved models, and documentation

---


