# Fire Detection in Images Using Computer Vision

## Introduction

This repository contains a project focused on detecting fire in images using advanced computer vision techniques. The primary goal is to develop an accurate model that can identify the presence of fire in images, which is crucial for applications such as early warning systems, surveillance, and safety monitoring.

Deployment: https://huggingface.co/spaces/utomon/fire_classification

## Project Description

The project follows these key steps:

1. **Data Collection**:
   - A comprehensive dataset of images, including both fire and non-fire scenes, was compiled from various sources.
   
2. **Data Preprocessing**:
   - Images were resized and normalized to ensure consistency across the dataset.
   - Data augmentation techniques were applied to enhance the model's robustness by simulating various real-world conditions.
   
3. **Modeling**:
   - We experimented with various deep learning models, including Convolutional Neural Networks (CNNs) and Artificial Neural Networks (ANNs).
   - Special attention was given to optimizing the architecture to improve detection accuracy.
   - The **Global Average Pooling** layer was found to be particularly effective, improving model performance by reducing overfitting and capturing global spatial information.

4. **Evaluation**:
   - The models were evaluated using accuracy, precision, recall, F1-score, and confusion matrix metrics.
   - The final model, leveraging Global Average Pooling, outperformed other configurations, achieving the best balance between sensitivity and specificity in detecting fire.

5. **Deployment**:
   - The trained model can be deployed in various environments for real-time fire detection, though this project primarily focuses on the detection in static images.

## Technology / Tools

- **Python**: Core programming language for the project.
- **OpenCV**: For image processing tasks such as reading, writing, and transforming images.
- **TensorFlow / Keras**: Used for building and training deep learning models, including CNNs and ANNs.
- **NumPy**: For efficient numerical computations.
- **Pandas**: For managing and manipulating the dataset.
- **Matplotlib / Seaborn**: For visualizing data distributions, model performance, and results.
- **scikit-learn**: For evaluation metrics and model performance analysis.

## Results

- The final model, incorporating **Global Average Pooling**, achieved superior results, with an accuracy of 93% and strong performance across other evaluation metrics.
- This model is well-suited for integration into systems requiring reliable fire detection from images, enhancing safety and response times.

