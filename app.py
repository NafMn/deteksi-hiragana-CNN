from flask import Flask, render_template, request
import cv2
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Load model
model = load_model('model/model.h5')

# Load label map from CSV
df_labels = pd.read_csv('data-label/hiragana_labels.csv')

# Pastikan nama kolom yang benar digunakan
if 'Hiragana' in df_labels.columns and 'Romaji' in df_labels.columns and 'Label' in df_labels.columns:
    hiragana_labels = dict(zip(df_labels['Hiragana'], zip(df_labels['Romaji'], df_labels['Label'])))
else:
    raise KeyError("Column names 'Hiragana', 'Romaji', and 'Label' are required in hiragana_labels.csv")


app = Flask(__name__)

def predict_image_cv(model, image, label_map):
    # Ubah gambar menjadi skala abu-abu
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize gambar ke ukuran yang diharapkan oleh model
    resized_img = cv2.resize(gray, (32, 32))
    # Normalisasi nilai pixel
    resized_img = resized_img / 255.0
    # Ubah gambar menjadi array numpy dan tambahkan dimensi batch
    img_array = np.expand_dims(img_to_array(resized_img), axis=0)
    img_array = np.expand_dims(img_array, axis=-1)

    # Melakukan prediksi menggunakan model
    predictions = model.predict(img_array)

    # Mendapatkan kelas prediksi
    predicted_class = np.argmax(predictions[0])

    # Mendapatkan label dari kelas prediksi
    predicted_label = None
    predicted_romaji = None
    for label, (romaji, class_id) in label_map.items():
        if class_id == predicted_class:
            predicted_label = label
            predicted_romaji = romaji
            break

    return predicted_label, predicted_romaji

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Ambil file gambar dari form
    image = request.files['image']
    # Baca file gambar
    img_np = np.frombuffer(image.read(), np.uint8)
    # Baca gambar menggunakan OpenCV
    frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    # Lakukan prediksi pada frame yang didapat
    predicted_label, predicted_romaji = predict_image_cv(model, frame, hiragana_labels)

    # Tampilkan hasil prediksi pada layar
    if predicted_label is not None and predicted_romaji is not None:
        cv2.putText(frame, f'Prediksi: {predicted_label} ({predicted_romaji})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Konversi frame OpenCV ke format yang bisa ditampilkan di HTML
    ret, buffer = cv2.imencode('.jpg', frame)
    img_str = buffer.tobytes()

    return img_str

if __name__ == '__main__':
    app.run(debug=True)
