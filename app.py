from flask import Flask, request, render_template
import numpy as np
from utils import load_trained_model, preprocess_image, predict_plate

app = Flask(__name__)

# Muat model saat aplikasi dimulai
model = load_trained_model('model/plate_classifier.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Ambil file gambar dari form
        if 'file' not in request.files:
            return render_template('index.html', error="Tidak ada file yang diunggah")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="Nama file kosong")
        
        if file:
            # Simpan gambar sementara
            filepath = 'static/uploads/' + file.filename
            file.save(filepath)
            
            # Preprocess dan prediksi
            img = preprocess_image(filepath)
            prediction = predict_plate(model, img)
            result = "Ganjil" if prediction[0] % 2 != 0 else "Genap"
            
            return render_template('index.html', result=result, image=file.filename)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)