# Fish Market Species Classification

## Overview
This project focuses on building a **Machine Learning classification model** to predict the species of fish based on physical attributes from the **Fish Market dataset**.

## Dataset
The dataset used is the **Fish Market Dataset** available on Kaggle: [Fish Market Dataset](https://www.kaggle.com/aungpyaeap/fish-market)

### Features in the dataset:
- `Species`: The species of the fish (Target Variable)
- `Weight`: The weight of the fish (in grams)
- `Length1`: Vertical length of the fish (in cm)
- `Length2`: Diagonal length of the fish (in cm)
- `Length3`: Cross length of the fish (in cm)
- `Height`: The height of the fish (in cm)
- `Width`: The diagonal width of the fish (in cm)

## Model Details
- **Model Type**: Classification
- **Algorithm Used**: Random Forest Classifier
- **Data Processing**:
  - Encoded the `Species` column using `LabelEncoder`.
  - Standardized numerical features using `StandardScaler`.
- **Performance Evaluation**:
  - Model accuracy was calculated using a test dataset.
  - Confusion matrix visualization was created for better evaluation.

## Installation & Usage
### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd fish-market-classification
```

### 2. Install Required Libraries
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### 3. Run the Model Training Script
```bash
python train_model.py
```

This will generate the trained model file: `fish_species_classifier.pkl`

### 4. Load & Use the Model
To use the trained model for predictions, you can load it as follows:
```python
import joblib
import numpy as np

# Load the trained model
model = joblib.load("fish_species_classifier.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# Example prediction (random input values)
sample_data = np.array([[230, 23.2, 25.4, 30.0, 11.52, 4.02]])
sample_data_scaled = scaler.transform(sample_data)
predicted_species = model.predict(sample_data_scaled)
predicted_species_label = le.inverse_transform(predicted_species)
print(f"Predicted Fish Species: {predicted_species_label[0]}")
```

## Deployment
- The model can be deployed using **Flask API**.
- It can be further hosted on **Heroku** to make predictions online.

## Files & Directories
- `fish.csv`: Dataset file
- `train_model.py`: Script to train and save the model
- `fish_species_classifier.pkl`: Trained model file
- `scaler.pkl`: Scaler file for preprocessing
- `label_encoder.pkl`: Encoder file for species labels

## Future Improvements
- Try different ML models like **XGBoost, SVM, Neural Networks** to improve accuracy.
- Develop a **Flask Web App** to allow users to input values and get predictions.
- Deploy the API to **Heroku** for real-time usage.

## Author
- **Your Name**
- **GitHub**: [Your GitHub Profile]

