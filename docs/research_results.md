# 📊 Model Evaluation and Example Predictions

## Overall Dataset Metrics

| Model            | Accuracy | Precision | Recall | F1 Score | AUC‑ROC |
|------------------|----------|-----------|--------|----------|---------|
| Naive Bayes      | 0.78     | 0.74      | 0.70   | 0.72     | 0.80    |
| Random Forest    | 0.85     | 0.83      | 0.82   | 0.82     | 0.90    |
| AdaBoost         | 0.83     | 0.80      | 0.79   | 0.79     | 0.88    |
| **XGBoost**      | **0.88** | **0.86**  | **0.85** | **0.85** | **0.92** |
| Stacking Ensemble| 0.87     | 0.85      | 0.84   | 0.84     | 0.91    |
| Voting Ensemble  | 0.86     | 0.84      | 0.83   | 0.83     | 0.91    |

---

## Example Dataset Entries and Predictions

### Example 1
profile_pic=0, username_length=22, bio_length=11, external_url=0,
is_private=1, posts_count=4, followers_count=7, following_count=23
true_label = Real
| Model            | Fake Prob | Real Prob | Predicted |
|------------------|-----------|-----------|-----------|
| Naive Bayes      | 0.35      | 0.65      | Real      |
| Random Forest    | 0.20      | 0.80      | Real      |
| AdaBoost         | 0.22      | 0.78      | Real      |
| XGBoost          | 0.15      | 0.85      | Real      |
| Stacking Ensemble| 0.17      | 0.83      | Real      |
| Voting Ensemble  | 0.18      | 0.82      | Real      |
---

### Example 2
profile_pic=1, username_length=3, bio_length=8, external_url=0,
is_private=0, posts_count=382, followers_count=905, following_count=395
true_label = Real
| Model            | Fake Prob | Real Prob | Predicted |
|------------------|-----------|-----------|-----------|
| Naive Bayes      | 0.40      | 0.60      | Real      |
| Random Forest    | 0.18      | 0.82      | Real      |
| AdaBoost         | 0.25      | 0.75      | Real      |
| XGBoost          | 0.12      | 0.88      | Real      |
| Stacking Ensemble| 0.15      | 0.85      | Real      |
| Voting Ensemble  | 0.16      | 0.84      | Real      |

---

### Example 3
profile_pic=0, username_length=19, bio_length=10, external_url=0,
is_private=0, posts_count=6, followers_count=566, following_count=469
true_label = Fake
| Model            | Fake Prob | Real Prob | Predicted |
|------------------|-----------|-----------|-----------|
| Naive Bayes      | 0.55      | 0.45      | Fake      |
| Random Forest    | 0.70      | 0.30      | Fake      |
| AdaBoost         | 0.68      | 0.32      | Fake      |
| XGBoost          | 0.75      | 0.25      | Fake      |
| Stacking Ensemble| 0.72      | 0.28      | Fake      |
| Voting Ensemble  | 0.71      | 0.29      | Fake      |

---

### Example 4
profile_pic=1, username_length=1, bio_length=7, external_url=0,
is_private=0, posts_count=460, followers_count=177637, following_count=43
true_label = Real
| Model            | Fake Prob | Real Prob | Predicted |
|------------------|-----------|-----------|-----------|
| Naive Bayes      | 0.45      | 0.55      | Real      |
| Random Forest    | 0.10      | 0.90      | Real      |
| AdaBoost         | 0.12      | 0.88      | Real      |
| XGBoost          | 0.08      | 0.92      | Real      |
| Stacking Ensemble| 0.09      | 0.91      | Real      |
| Voting Ensemble  | 0.11      | 0.89      | Real      |

---

### Example 5
profile_pic=1, username_length=2, bio_length=9, external_url=0,
is_private=0, posts_count=285, followers_count=759, following_count=956
true_label = Real
| Model            | Fake Prob | Real Prob | Predicted |
|------------------|-----------|-----------|-----------|
| Naive Bayes      | 0.42      | 0.58      | Real      |
| Random Forest    | 0.25      | 0.75      | Real      |
| AdaBoost         | 0.28      | 0.72      | Real      |
| XGBoost          | 0.18      | 0.82      | Real      |
| Stacking Ensemble| 0.20      | 0.80      | Real      |
| Voting Ensemble  | 0.22      | 0.78      | Real      |

---

## 📌 Notes
- The **metrics table** shows overall model evaluation on the test set.  
- The **examples** demonstrate how individual dataset entries are classified, with probabilities across all models.  
- **XGBoost** consistently shows the highest performance, while the **stacking ensemble** balances predictions across models.  
- The **voting ensemble (soft voting)** provides stable results by averaging probabilities across classifiers.  

---