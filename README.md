# TensorFlow Image Classification Project

## Overview
This project implements an image classification model using TensorFlow and Keras. It trains a Convolutional Neural Network (CNN) on a dataset of images, evaluates performance, and saves the trained model for future use.

## Features
- **Data Augmentation**: Enhances training data using transformations like rotation, zoom, and flipping.
- **CNN Architecture**: A deep learning model with convolutional, max-pooling, batch normalization, and dropout layers.
- **Binary Classification**: Classifies images into two categories.
- **Model Optimization**:
  - Early Stopping to prevent overfitting.
  - Learning Rate Scheduling for better convergence.
- **Performance Metrics**:
  - Accuracy, Precision, Recall, F1-Score, and Intersection over Union (IoU).
  - Confusion Matrix and Classification Report for detailed analysis.

## Requirements
- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn

Install dependencies using:
```bash
pip install tensorflow numpy matplotlib scikit-learn
```

## Dataset Structure
Place images in the following directory structure:
```
Problem_2/
│-- Train/
│   ├── Class1/
│   ├── Class2/
│-- Test/
│   ├── Class1/
│   ├── Class2/
```

## Model Architecture
The CNN model consists of:
- **4 Convolutional Layers** (32, 64, 128, 128 filters)
- **Batch Normalization** after each convolution
- **Max Pooling** layers
- **Fully Connected Layers** with Dropout (to prevent overfitting)
- **Sigmoid Activation** for binary classification

## Training
Run the following command to train the model:
```bash
python train.py
```
- The model is trained for 100 epochs with batch size 32.
- Uses **Adam Optimizer** with a learning rate of 0.0001.
- Training history is saved as `history.json`.

## Model Evaluation
After training, the model evaluates performance using:
- **Accuracy** and **Loss Curves**
- **Confusion Matrix & Classification Report**

Run the evaluation script:
```bash
python evaluate.py
```

## Model Saving and Loading
- The trained model is saved as `Model.h5`.
- To load the model for inference:
```python
from tensorflow.keras.models import load_model
model = load_model("Model.h5")
```

## Results & Performance
- **Accuracy, Precision, Recall, F1-Score, and IoU** are printed in the console.
- **Confusion Matrix** helps visualize misclassifications.
- **Plots** for accuracy and loss trends during training.

## Future Enhancements
- Implement multi-class classification.
- Use Transfer Learning for better accuracy.
- Deploy the model using Flask or FastAPI for real-time predictions.

## License
This project is open-source and licensed under the **MIT License**.

---
**Developed by:** *Majd Ghazal*
