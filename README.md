# Instagram Fake Profile Detection using Ensemble Learning Methods

## Overview
This project presents a machine learning system designed to identify fake Instagram profiles using structured data. The solution combines multiple classification algorithms and integrates them through an ensemble approach to improve prediction reliability.

The system is implemented as a complete pipeline that includes data preprocessing, feature engineering, model training, backend API development, and an interactive frontend for user interaction.

---

## Key Features
- Implementation of multiple machine learning models for comparative analysis
- Use of ensemble learning to improve prediction performance
- Feature engineering to extract meaningful indicators such as engagement score and follower ratio
- REST API built using Flask for model inference
- Interactive dashboard developed using Streamlit
- Visualization of model performance metrics and prediction probabilities

---

## Models Used

| Model            | Category        |
|------------------|-----------------|
| Naive Bayes      | Probabilistic   |
| Random Forest    | Bagging         |
| XGBoost          | Boosting        |
| AdaBoost         | Boosting        |
| Voting Ensemble  | Hybrid Ensemble |

---

## Features Considered

- profile_pic  
- username_length  
- bio_length  
- external_url  
- is_private  
- posts_count  
- followers_count  
- following_count  
- followers_following_ratio  
- engagement_score  

---

## System Architecture

Dataset -> Preprocessing -> Feature Engineering -> Model Training
-> Ensemble Model -> Model Storage -> Flask API -> Streamlit Interface


---

## Project Structure

instagram_fake_profile_detector/
│
# Trained machine learning models
├── models/
# Dataset files
├── data/
# Static assets such as images
├── static/
# Documentation(architecture_diagram)
├── docs/
│
# Initial training pipeline
├── main.py
# Final training script
├── train_models.py
# Backend API
├── flask_app.py
# Streamlit frontend
├── app.py
│
# Dependencies
├── requirements.txt
# Project documentation
├── README.md
# License file
├── LICENSE
# Ignored files
└── .gitignore


---

## Installation

Clone the repository and navigate to the project directory:

git clone https://github.com/your-username/instagram_fake_profile_detector.git

cd instagram_fake_profile_detector

Install the required dependencies:
pip install -r requirements.txt

---

## Execution Steps

Train the models:
python train_models.py

Start the backend server:
python flask_app.py

Launch the frontend application:
streamlit run app.py

---

## API Endpoints

| Endpoint        | Method | Description                     |
|----------------|--------|---------------------------------|
| /metrics       | GET    | Returns model evaluation metrics |
| /predict/all   | POST   | Returns predictions from all models |

---

## Output

The system provides:
- Classification result (fake or real)
- Probability scores for each class
- Voting-based ensemble decision
- Model performance metrics including accuracy, precision, recall, F1 score, and ROC-AUC

---

## Future Enhancements

- Integration of deep learning models for image-based analysis
- Natural language processing of profile bios
- Real-time data integration using external APIs
- Network-based detection of coordinated fake accounts

---

## Author

Vaishnavi Naga Sai Ravula  
Bachelor of Technology Student  
Focus area: Machine Learning and Data Science

---

## License

This project is licensed under the MIT License.