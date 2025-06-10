import tensorflow as tf
import cv2
import numpy as np
import os

def load_trained_model(model_path):
    """Memuat model yang sudah dilatih."""
    return tf.keras.models.load_model(model_path)

def preprocess_image(image_path):
    """Preprocessing gambar: resize dan normalisasi."""
    img = cv2.imread(image_path)
    img = cv2.resize(img, (222, 222))  # Sesuaikan dengan input model (222x222)
    img = img / 255.0  # Normalisasi
    img = np.expand_dims(img, axis=0)  # Tambah dimensi batch
    return img

def predict_plate(model, image):
    """Melakukan prediksi pada gambar."""
    prediction = model.predict(image)
    return np.argmax(prediction, axis=1)  # Ambil indeks kelas dengan probabilitas tertinggi