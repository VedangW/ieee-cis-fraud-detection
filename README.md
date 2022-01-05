# ieee-cis-fraud-detection
Machine Learning to detect transaction fraud using the IEEE-CIS Fraud Detection Dataset.

---

### Introduction

The [IEEE-CIS Fraud Detection competition](https://www.kaggle.com/c/ieee-fraud-detection/overview) was held on Kaggle during 2019 and focused on benchmarking Machine Learning models on a large-scale transaction dataset. The task was a binary classification task: classify an online transaction as `fraud` or `not fraud` .

We decided to explore this dataset on our own and build models to detect online fraud. Our features and models are built keeping in mind if it is feasible to create a “product” out of our solution. Our XGBoost model was able to reach an AUC value of `0.885` on the validation dataset.

### Approach

Join tables → EDA → Feature Selection → Feature Engineering → Modeling → Evaluation

### Highlights

- For Feature Selection we dropped features according to:
    - Percentage of null values
    - High correlation (for each highly correlated pair, remove one feature by number of unique values) as in
    - Adversarial Validation (ensures time consistency of features in the train and validation set)
    - Permutation Importance
- For Feature Engineering we tried creating time-based expanding window features (see description below) along with other features. We encoded categorical features as follows:
    - Binary encode features with 2 unique values
    - One-hot encoding features with low cardinality (e.g. 3-10 unique values is fine)
    - Frequency encode features with high cardinality (e.g. 100 unique values in 1 column)
    
    We keep in mind while encoding that at each point in time, we only have the information from the time before it (see section below). Our feature selection and engineering notebook can be found at [`feature-selection-and-engineering.ipynb`](https://github.com/VedangW/ieee-cis-fraud-detection/blob/main/experiments/feature-selection-and-engineering.ipynb).
    
- Our main focus for modeling was tree-based gradient boosting models (XGBoost, LightGBM, CatBoost) and Deep Neural Networks (Multi-layer perceptron). XGBoost gave the best performance with `0.885` AUC. This notebook can be found in [`xgboost-plus-adversarialvalidation.ipynb`](https://github.com/VedangW/ieee-cis-fraud-detection/blob/main/experiments/xgboost-plus-adversarialvalidation.ipynb).
- The notebook for training and testing Deep Learning models can be found at [`fraud-detection-nn.ipynb`](https://github.com/VedangW/ieee-cis-fraud-detection/blob/main/experiments/fraud-detection-nn.ipynb).

### Methodology

To create a practical solution, we need to ensure that the value of each feature needs to be created only by considering samples before that timestamp. This mainly impacts 3 processes: cross-validation, categorical feature encoding, and time-based feature creation.

Before all of these processes, we sort the training dataset by timestamp (`TransactionDT`).

**Cross Validation**

We cannot use the traditional k-fold cross validation procedure, and instead use an expanding time-based cross validation, as shown in the diagram below (taken from [https://stats.stackexchange.com/questions/14099/using-k-fold-cross-validation-for-time-series-model-selection](https://stats.stackexchange.com/questions/14099/using-k-fold-cross-validation-for-time-series-model-selection)).

![time-based-cross-validation.png](images/time-based-cross-validation.png)

**Feature Encoding**

Consider a sample categorical column (e.g. `P_emailDomain`) we frequency encode it as follows:

| Timestamp | P_emailDomain | Encoded Value |
| --- | --- | --- |
| 1 | gmail.com | 0.0 |
| 2 | yahoo.com | 0.0 |
| 3 | gmail.com | 0.5 |
| 4 | yahoo.com | 0.33 |
| 5 | gmail.com | 0.5 |
| 6 | rediffmail.com | 0.0 |
| 7 | gmail.com | 0.5 |
| 8 | yahoo.com | 0.285 |
| 9 | yahoo.com | 0.375 |

For each value, we encode it as follows:

```
f_e = n_e/n
```

Where `n_e` denotes the number of rows before the sample where the value `e` occurs and `n` denotes the total number of rows before the current timestamp.

**Time-Based Feature Creation**

Time based features are created in a similar way, where the feature is a time-wise aggregation of a numerical variable (e.g. `isFraud`) using an ID column like `P_emailDomain`. For e.g. we have a feature denoting the ratio of frauds committed from a particular email domain before the given timestamp. 

### Future Work and References

- [EDA Kaggle Notebook](https://www.kaggle.com/vedangw/frauddetection-ieee-cis)
- [Feature Selection and Engineering Kaggle Notebook](https://www.kaggle.com/vedangw/feature-selection-and-engineering)
- Future Work
    - Add Permutation Importance code
    - Add Time-split cross validation code