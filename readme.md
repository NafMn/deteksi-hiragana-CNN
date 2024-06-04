# Deteksi Hiragana Basic Menggunakan CNN

Deteksi Hiragana Basic adalah aplikasi berbasis web yang menggunakan Convolutional Neural Network (CNN) untuk mendeteksi huruf Hiragana dari gambar. Aplikasi ini dibuat menggunakan TensorFlow, Keras, Pandas, OpenCV, dan Matplotlib.

## Author

Mohamad Nafis

## Fitur

- Pemrosesan gambar menggunakan OpenCV
- Pendeteksian huruf Hiragana menggunakan model CNN
- Antarmuka web sederhana untuk mengunggah gambar dan melihat hasil deteksi

## Prasyarat

Pastikan Anda menggunakan Python versi 3.9. Instal Python dari [python.org](https://www.python.org/downloads/release/python-390/).

## Instalasi

Ikuti langkah-langkah berikut untuk menginstal semua dependensi yang diperlukan:

1. **Clone repositori ini:**

    ```bash
    git clone https://github.com/username/repo-name.git
    cd repo-name
    ```

2. **Buat virtual environment dan aktifkan:**

    - **Windows:**
      ```bash
      python -m venv myenv
      myenv\Scripts\activate
      ```

    - **MacOS/Linux:**
      ```bash
      python3 -m venv myenv
      source myenv/bin/activate
      ```

3. **Instal dependensi dari `requirements.txt`:**

    ```bash
    pip install -r requirements.txt
    ```

## Penggunaan

1. Pastikan virtual environment sudah aktif.

2. Jalankan aplikasi:

    ```bash
    python app.py
    ```

3. Buka browser dan akses `http://localhost:5000` untuk mengakses antarmuka web.

4. Unggah gambar yang ingin Anda deteksi huruf Hiragana-nya.

5. Lihat hasil deteksi pada halaman hasil.

## Dependencies

- tensorflow
- keras
- pandas
- opencv-python
- matplotlib
- numpy

## Lisensi

Proyek ini dilisensikan di bawah MIT License. Lihat [LICENSE](./LICENSE) untuk informasi lebih lanjut.

---

Terima kasih telah menggunakan Deteksi Hiragana Basic! Jika Anda memiliki pertanyaan atau masalah, silakan buka issue di repositori ini atau hubungi saya langsung.
