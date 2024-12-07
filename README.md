# Student Performance and Medical Prediction Neural Networks

## Overview
This repository contains three Jupyter notebooks demonstrating different machine learning applications using neural networks, focusing on performance prediction, medical diagnosis, and image classification.

## Notebooks

### 1. Performance Index Prediction using Feedforward Neural Network
**Notebook**: [`Performance_Index_Prediction_using_Feedforward_NN.ipynb`](./Performance_Index_Prediction_using_Feedfoward_NN.ipynb)  

**Objective:** Predict student academic performance using a feedforward neural network.

**Dataset Features:**
- Hours Studied
- Previous Scores
- Extracurricular Activities
- Sleep Hours
- Sample Question Papers Practiced

**Target Variable:** Performance Index (10-100)
- Represents overall academic performance
- Higher values indicate better performance

**Methodology:**
- Developed a neural network to predict performance index
- Preprocessed and analyzed student-related features
- Trained and evaluated the predictive model

### 2. Diabetes Diagnosis using Neural Networks
**Notebook**: [`Diabetes_diagnosis_and_bean_classification_using_NN.ipynb`](./Diabetes_diagnosis_and_bean_classification_using_NN.ipynb)

**Objective:** Develop a classification model to predict diabetes in female patients.

**Key Aspects:**
- Medical dataset with female patients
- Binary classification problem (Diabetes: Yes/No)
- Neural network-based diagnostic model

**Approach:**
- Built a neural network classifier
- Implemented appropriate validation techniques
- Evaluated model performance using classification metrics

### 3. Natural Scene Image Classification
**Notebook**: [`Image_classification.ipynb`](./Image_classification.ipynb)  

**Objective:** Develop and compare convolutional neural network (CNN) models for image classification.

**Dataset:** Natural scene images across six predefined categories

**Model Architectures Explored:**
- Dense Layer Model
- Convolutional and Dense Layer Model
- Residual Identity Block Model
- Transfer Learning Model (using TensorFlow pre-trained backbones)

**Methodology:**
- Implemented multiple neural network architectures
- Comparative analysis of model performance
- Utilized convolutional and transfer learning techniques

## Technologies Used
- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-learn

## Installation
1. Clone the repository
2. Install required dependencies:
   ```bash
   pip install tensorflow numpy pandas scikit-learn
   ```

## Usage
Open the Jupyter notebooks and run the cells sequentially to reproduce the experiments and results.

## Contributing
Contributions, issues, and feature requests are welcome. Feel free to check the issues page.
