import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
from pathlib import Path
import os


# Data Loading and Preprocessing

def load_data():
    data = pd.read_csv('data/ThoracicCancerSurgery.csv')
    data = augment_data(data, ['FVC', 'FEV1'], seed_value=144)
    return data

def augment_data(input_data, target_columns, seed_value):
    np.random.seed(seed_value)
    augmented_data = input_data.copy()
    for col in target_columns:
        if col in augmented_data.columns and augmented_data[col].dtype in ['float64', 'int64']:
            augmented_data[col] += np.random.normal(0, 0.1, size=len(input_data))
            augmented_data[col] = shuffle(augmented_data[col], random_state=seed_value)
            np.random.seed(seed_value)
            augmented_data[col] = np.random.permutation(augmented_data[col])
    
    augmented_data = shuffle(augmented_data, random_state=seed_value)
    
    # Save augmented data
    project_data_path = os.path.join(os.getcwd(), "data/AugmentedThoracicSurgery.csv")
    augmented_data.to_csv(project_data_path, index=False)
    
    return augmented_data

# Model Training and Evaluation (EXACTLY AS IN YOUR ORIGINAL EVALUATION CODE)
def train_and_evaluate():
    # Load and preprocess data exactly as in your evaluation code
    augmented_data = load_data()
    
    # Your exact preprocessing code
    label_encoder = LabelEncoder()
    augmented_data['DGN'] = label_encoder.fit_transform(augmented_data['DGN'])
    augmented_data['performance'] = label_encoder.fit_transform(augmented_data['performance'])
    augmented_data['Tumor-size'] = label_encoder.fit_transform(augmented_data['Tumor-size'])
    augmented_data.replace({'F': 0, 'T': 1}, inplace=True)
    
    # Your exact feature selection
    X = augmented_data[['FVC','FEV1','Asthama','Smoking','PAD','mi-6-mo','Diabetes-mellitus','AGE']]
    y = augmented_data['Risk1Yr'].values
    
    # Your exact train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=20)

    # Your exact LSTM model architecture
    X_train_reshaped = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_reshaped = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

    model = models.Sequential()
    model.add(layers.LSTM(64, input_shape=(1, X_train.shape[1])))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train_reshaped, y_train.reshape(-1, 1), epochs=10, batch_size=128, validation_split=0.30)

    # Your exact evaluation code
    y_pred = model.predict(X_test_reshaped)
    y_pred_classes = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred_classes) * 100
    conf_matrix = confusion_matrix(y_test, y_pred_classes)
    tn, fp, fn, tp = conf_matrix[1][1], conf_matrix[0][1], conf_matrix[1][0], conf_matrix[0][0]
    sensitivity = tp / (tp + fn)
    specificity = accuracy - sensitivity
    precision = tp / (tp + fp)
    recall = sensitivity
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
    roc_auc = roc_auc_score(y_test, y_pred)

    # Your exact print statements
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Specificity: {specificity:.2f}")
    print(f"Sensitivity (Recall): {sensitivity:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"F1-Score: {f1_score:.2f}")
    print(f"ROC AUC: {roc_auc:.2f}")

    # Your exact plotting code
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    # Save the ROC curve to static/images folder
    static_folder = 'static/images'
    os.makedirs(static_folder, exist_ok=True)
    plt.savefig(os.path.join(static_folder, 'roc_curve.png'), bbox_inches='tight')
    plt.show()

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')  
    plt.legend()
    
    # Save the Loss curve to static/images folder
    plt.savefig(os.path.join(static_folder, 'loss_curve.png'), bbox_inches='tight')
    plt.show()
    
    # Generate and save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(static_folder, 'confusion_matrix.png'), bbox_inches='tight')
    plt.show()

    return model

# Prediction Function (from your prediction code)
def predict_risk(patient_data):
    # Load standard data for XGBoost prediction (not augmented)
    data = pd.read_csv('data/AugmentedThoracicSurgery.csv')
    
    # Convert binary columns
    binary_cols = ['Pain', 'Haemoptysis', 'Dysponoea', 'Cough', 'Weakness', 
                   'Diabetes-mellitus', 'mi-6-mo', 'PAD', 'Smoking', 'Asthama']
    for col in binary_cols:
        data[col] = data[col].map({'T': 1, 'F': 0})
    
    data['Risk1Yr'] = data['Risk1Yr'].map({'T': 1, 'F': 0})
    
    # Select features and scale
    features = ['FVC', 'FEV1', 'AGE', 'Diabetes-mellitus', 'mi-6-mo', 'PAD', 'Smoking', 'Asthama']
    X = data[features].copy()
    y = data['Risk1Yr']
    
    scaler = StandardScaler()
    num_cols = ['FVC', 'FEV1', 'AGE']
    X.loc[:, num_cols] = scaler.fit_transform(X[num_cols])
    
    # Train XGBoost model (or load if exists)
    try:
        model = joblib.load('thoracic_surgery_risk_model.pkl')
    except:
        model = XGBClassifier(random_state=42, eval_metric='logloss')
        model.fit(X, y)
        joblib.dump(model, 'thoracic_surgery_risk_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
    
    # Prepare input
    input_df = pd.DataFrame([patient_data])
    input_df.loc[:, num_cols] = scaler.transform(input_df[num_cols])
    
    # Predict
    probability = model.predict_proba(input_df)[0, 1]
    risk_level = "High Risk" if probability >= 0.5 else "Low Risk"
    
    return {
        "risk_level": risk_level,
        "probability": round(probability * 100, 2)
    }

# Main Execution
if __name__ == "__main__":
    # Run your exact model evaluation
    print("Running LSTM model evaluation (exact implementation from your code)...")
    lstm_model = train_and_evaluate()
    
    # Example prediction using XGBoost
    example_patient = {
        'FVC': 2.3,
        'FEV1': 1.6,
        'AGE': 102,
        'Diabetes-mellitus': 0,  # No
        'mi-6-mo': 0,           # No
        'PAD': 0,               # No
        'Smoking': 1,           # Yes
        'Asthama': 1            # Yes
    }
    
    result = predict_risk(example_patient)
    print("\n# Prediction Results")
    print("## Thoracic Surgery Risk Assessment")
    print(f"# {result['risk_level']}")
    print(f"**Probability: {result['probability']}%**")
    print("\nThis indicates a high probability of complications after thoracic surgery." 
          if result['risk_level'] == "High Risk" else
          "\nThis indicates a low probability of complications after thoracic surgery.")
