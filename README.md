
# Skin Cancer Detection using Deep Learning

A deep learning-based skin cancer classification system trained on the HAM10000 dataset using CNNs.

## Dataset

This project uses the HAM10000 dataset from Kaggle.

Classes:
- Melanocytic Nevi
- Melanoma
- Benign Keratosis
- Basal Cell Carcinoma
- Actinic Keratoses
- Vascular Lesions
- Dermatofibroma

Source:
https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

## Features

- CNN-based classification
- Image preprocessing
- Trained Keras model
- Prediction application
- Training notebook

## Technologies

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib

## Project Structure

skin-cancer-detection/
├── src/
│   └── app.py
├── models/
│   └── model.h5
├── notebooks/
│   └── training.ipynb
├── data/ (download separately)
├── requirements.txt
├── README.md
└── .gitignore

## Setup

1. Install dependencies

pip install -r requirements.txt

2. Download dataset and place in:

data/ham10000/

3. Run application

python src/app.py

## Author

Ankit Mourya  
AI/ML Engineer  
GitHub: https://github.com/theankitmourya
