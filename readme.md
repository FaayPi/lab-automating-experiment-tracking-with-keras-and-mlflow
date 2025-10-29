![IronHack_Logo](https://user-images.githubusercontent.com/92721547/180665853-e52e3369-9973-4c1e-8d88-1ecef1eb8e9e.png)

# LAB |  Automating Experiment Tracking with Keras and MLflow

### Overview

In this lab, you will enhance a standard Keras script for training a classification model on the Reuters dataset and supercharge it with `MLflow` for automatic experiment tracking. Your task is to use MLflow's `autolog()` functionality to capture metrics, parameters, and artifacts with minimal manual logging. This exercise will deepen your understanding of experiment lifecycle management using MLflow.

### Learning Goals

Upon completion, you will be able to:

- Integrate MLflow tracking in a Keras text classification pipeline.
- Use `mlflow.tensorflow.autolog()` to automatically log model training details.
- Manually log additional parameters and metrics with MLflow API.
- Navigate the MLflow UI to compare different experiment runs.
- Understand MLflow's organization of experiments, runs, and artifacts.

### Prerequisites

- Basic knowledge of Python and Keras.
- MLflow installed and configured.
- Provided script: `keras_reuters_mlp.py` to be modified into `reuters_mlflow.py`.

### Instructions

#### Part A: Integrate MLflow Autologging

1. Use the Keras MLP model from `keras_reuters_mlp.py`, and add MLflow import at the top of `reuters_mlflow.py`.
2. Within the `train_model()` function, call `mlflow.tensorflow.autolog()` before compiling the model to enable automatic logging.

#### Part B: Manual Logging Enhancement

1. After loading the data, log the following parameters with `mlflow.log_param()`:
   - Learning Rate (`learning_rate`)
   - Batch Size (`batch_size`)
   
2. After model evaluation, explicitly log the final test accuracy and loss with `mlflow.log_metric()`.

3. Add a tag to your experiment using `mlflow.set_tag()` to identify it, e.g., `"project": "reuters_classification"`.

#### Part C: Run and Compare Experiments

1. Execute your enhanced script:

ç

2. Modify hyperparameters such as `num_words`, `maxlen`, batch size, or model architecture and rerun to create variations.

3. Launch the MLflow UI:

```bash
mlflow ui
```

Access the UI at `http://localhost:5000` to view logged experiments and compare metrics.


### Resource

Feel free to check the official MLFlow docuemntion & tutorial at [MLflow Tracking APIs](https://mlflow.org/docs/3.1.3/ml/tracking/tracking-api/)

### Deliverables

- Submit your modified `reuters_mlflow.py` file with all MLflow integrations.
- Provide screenshots or exported views from the MLflow UI showing experiments, metrics, and runs.


## Submission

Upon completion, add your deliverables to git. Then commit git and push your branch to the remote.

***