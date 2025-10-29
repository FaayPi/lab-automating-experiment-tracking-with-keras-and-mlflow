from __future__ import print_function

import warnings
warnings.filterwarnings('ignore')

import mlflow
import warnings
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
import keras
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import mlflow.sklearn

print(f"MLflow version: {mlflow.__version__}")

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()
print(f"Tracking URI: {mlflow.get_tracking_uri()}")

# to create a new experiment
experiment_name = "reuters-classification"
mlflow.set_experiment(experiment_name)

# to restore an already deleted experiment
#experiment = client.get_experiment_by_name("reuters-classification")
#client.restore_experiment(experiment.experiment_id)

# Create model

max_words = 1000
batch_size = 32
epochs = 5

# Lade Daten
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_words, test_split=0.2)

# Vektorisiere Sequenzen
tokenizer = Tokenizer(num_words=max_words)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')

print(x_train.shape, x_test.shape, len(set(y_train)))

# Klassen bestimmen
num_classes = np.max(y_train) + 1
print(num_classes, 'classes')

# Labels in One-Hot-Form umwandeln
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)

model = Sequential([
    Dense(512, input_shape=(max_words,)),
    Activation('relu'),
    Dropout(0.5),
    Dense(num_classes),
    Activation('softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Start MLflow Run

# Labels für RandomForest müssen 1D sein
y_train_rf = np.argmax(y_train, axis=1)
y_test_rf  = np.argmax(y_test, axis=1)

with mlflow.start_run(run_name="randomforest-baseline") as run:
    mlflow.set_tag("author", "Fee Pieper")
    mlflow.set_tag("model_family", "RandomForest")
    mlflow.set_tag("project", "reuters_classification")
    mlflow.set_tag("description", "Baseline RandomForest model for Reuters classification") 

    # Hyperparameter
    params_v1 = {
        "max_depth": 5,
        "n_estimators": 10,
        "random_state": 42,
        "n_jobs": -1
    }
  
    # Log parameters
    mlflow.log_params(params_v1)
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("dataset", "reuters")
    
    # Train the RandomForest model
    model_v1 = RandomForestClassifier(**params_v1)
    model_v1.fit(x_train, y_train_rf)
    
    # Predictions and metrics
    y_pred_v1 = model_v1.predict(x_test)
    accuracy_v1 = accuracy_score(y_test_rf, y_pred_v1)
    precision_v1 = precision_score(y_test_rf, y_pred_v1, average='weighted')
    recall_v1 = recall_score(y_test_rf, y_pred_v1, average='weighted')
    f1_v1 = f1_score(y_test_rf, y_pred_v1, average='weighted')
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy_v1)
    mlflow.log_metric("precision", precision_v1)
    mlflow.log_metric("recall", recall_v1)
    mlflow.log_metric("f1_score", f1_v1)

    # Log model
    model_name_randomforest = "reuters-classifier-randomforest"
    mlflow.sklearn.log_model(
        sk_model=model_v1,
        name="model",
        registered_model_name=model_name_randomforest,
        input_example=x_test[:1]  # Beispielinput für die Signatur
        )
    
    # Store run_id
    run_id_v1 = run.info.run_id

    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETE")
    print("="*60)
    print(f"Run ID: {run_id_v1}")
    print(f"\nMetrics:")
    print(f"  - Accuracy:  {accuracy_v1:.4f}")
    print(f"  - Precision: {precision_v1:.4f}")
    print(f"  - Recall:    {recall_v1:.4f}")
    print(f"  - F1 Score:  {f1_v1:.4f}")
    print("\n✓ Model logged to MLflow")
    print(f"\n👉 View this run in the UI: http://localhost:5000")

# Transition version 1 to Production stage
client.transition_model_version_stage(
    name=model_name_randomforest,
    version="1",
    stage="Production",
    archive_existing_versions=False  # Keep other versions in their current stages
)

# Update the model version description
client.update_model_version(
    name=model_name_randomforest,
    version="1",
    description="Baseline RandomForest model for reuters classification. "
                "Using conservative hyperparameters (max_depth=5). "
                "Currently serving testing traffic in 'Staging' stage."
)

# update the registered model description
client.update_registered_model(
    name=model_name_randomforest,
    description="Reuters classifier using RandomForest. "
)

print("✓ Model stage and descriptions updated via API")
print(f"\nVersion 1 stage: 'Production'")
print(f"\n👉 View in UI: http://127.0.0.1:5000/#/models/{model_name_randomforest}")

# Model improvement

with mlflow.start_run(run_name="randomforest-improvement") as run:
    mlflow.set_tag("author", "Fee Pieper")
    mlflow.set_tag("model_family", "RandomForest")
    mlflow.set_tag("project", "reuters_classification")
    mlflow.set_tag("description", "Improvement of RandomForest model for Reuters classification") 

    # Hyperparameter
    params_v2 = {
        "max_depth": 3,
        "n_estimators": 50,
        "random_state": 42
    }
  
    # Log parameters
    mlflow.log_params(params_v2)
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("dataset", "reuters")
    
    # Train the RandomForest model
    model_v2 = RandomForestClassifier(**params_v2)
    model_v2.fit(x_train, y_train_rf)
    
    # Predictions and metrics
    y_pred_v2 = model_v2.predict(x_test)
    accuracy_v2 = accuracy_score(y_test_rf, y_pred_v2)
    precision_v2 = precision_score(y_test_rf, y_pred_v2, average='weighted')
    recall_v2 = recall_score(y_test_rf, y_pred_v2, average='weighted')
    f1_v2 = f1_score(y_test_rf, y_pred_v2, average='weighted')
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy_v2)
    mlflow.log_metric("precision", precision_v2)
    mlflow.log_metric("recall", recall_v2)
    mlflow.log_metric("f1_score", f1_v2)

    # Log model
    model_name_randomforest = "reuters-classifier-randomforest"
    mlflow.sklearn.log_model(
        sk_model=model_v2,
        artifact_path="model",
        registered_model_name=model_name_randomforest,
        input_example=x_test[:2],  # Beispielinput für die Signatur
        )
    
    # Store run_id
    run_id_v2 = run.info.run_id

    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETE")
    print("="*60)
    print(f"Run ID: {run_id_v2}")
    print(f"\nMetrics:")
    print(f"  - Accuracy:  {accuracy_v2:.4f}")
    print(f"  - Precision: {precision_v2:.4f}")
    print(f"  - Recall:    {recall_v2:.4f}")
    print(f"  - F1 Score:  {f1_v2:.4f}")
    print("\n✓ Model logged to MLflow")
    print(f"\n👉 View this run in the UI: http://localhost:5000")

# Transition version 2 to Staging stage
client.transition_model_version_stage(
    name=model_name_randomforest,
    version="2",
    stage="Staging",
    archive_existing_versions=False  # Keep other versions in their current stages
)

# Update the model version description
client.update_model_version(
    name=model_name_randomforest,
    version="2",
    description="Baseline RandomForest model for reuters classification. "
                "Using conservative hyperparameters (max_depth=3, n_estimators=50). "
                "Currently serving testing traffic in 'Staging' stage."
)

# Also update the registered model description
client.update_registered_model(
    name=model_name_randomforest,
    description="Reuters classifier using RandomForest. "
)

print("✓ Model stage and descriptions updated via API")
print(f"\nVersion 2 stage: 'Staging' (testing model)")
print(f"\n👉 View in UI: http://127.0.0.1:5000/#/models/{model_name_randomforest}")

# Create a comparison table
comparison_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Production (v1)': [accuracy_v1, precision_v1, recall_v1, f1_v1],
    'Staging (v2)': [accuracy_v2, precision_v2, recall_v2, f1_v2],
    'Improvement': [
        f"{((accuracy_v2 - accuracy_v1) / accuracy_v1 * 100):+.2f}%",
        f"{((precision_v2 - precision_v1) / precision_v1 * 100):+.2f}%",
        f"{((recall_v2 - recall_v1) / recall_v1 * 100):+.2f}%",
        f"{((f1_v2 - f1_v1) / f1_v1 * 100):+.2f}%"
    ]
})

print("\n" + "="*60)
print("MODEL COMPARISON: Production vs Staging")
print("="*60)
print(comparison_df.to_string(index=False))

if accuracy_v2 > accuracy_v1:
    print("\n✅ Staging model shows improvement! Ready for promotion consideration.")
else:
    print("\n⚠️  Staging model did not improve. Consider keeping Production model.")