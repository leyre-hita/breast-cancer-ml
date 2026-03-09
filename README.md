# Breast Cancer Classification using Neural Networks

## Project Overview

This project builds a neural network to classify breast tumors as **benign** or **malignant** using the Breast Cancer Wisconsin Diagnostic Dataset.

The goal is to apply **machine learning and deep learning techniques** to perform binary classification on medical data.

## Dataset

The dataset used is the **Breast Cancer Wisconsin Diagnostic Dataset**, which contains features computed from digitized images of breast mass cell nuclei.

Features include measurements such as:

* radius
* texture
* perimeter
* area
* smoothness

The target variable indicates whether the tumor is **benign (0)** or **malignant (1)**.

## Project Structure

```
breast-cancer-project
│
├── data
│   └── breast_cancer_wisconsin_diagnostic.csv
│
├── notebooks
│   └── exploration.ipynb
│
├── src
│   ├── preprocessing
│   ├── model
│   └── train
│
├── models
│   └── cancer_model.keras
│
├── requirements.txt
└── README.md
```

## Model

The model used is a **Neural Network built with TensorFlow/Keras**.

Architecture:

* Dense layer (30 neurons, ReLU)
* Dropout (0.5)
* Dense layer (15 neurons, ReLU)
* Dropout (0.5)
* Output layer (Sigmoid)

Techniques used:

* **Dropout** to reduce overfitting
* **Early stopping** to prevent unnecessary training

## Technologies

* Python
* TensorFlow / Keras
* Scikit-learn
* Pandas
* Matplotlib
* Seaborn
* Jupyter Notebook

## Results

The model is able to classify tumors with high accuracy using neural networks and proper regularization techniques.

## Author

Data Science / Machine Learning project.
