# 🎙️ Hệ Thống Phát Hiện Giọng Nói Giả (Fake Voice Detection)

Đây là một hệ thống phát hiện **giọng nói thật và giọng nói giả mạo** được xây dựng bằng **Machine Learning và Deep Learning**. Người dùng có thể **tải lên file âm thanh**, chọn **mô hình đã huấn luyện**, và hệ thống sẽ dự đoán xem giọng nói đó là **thật hay giả**.

---

# 📌 Tổng Quan Dự Án

Sự phát triển của công nghệ **AI Voice Cloning và Speech Synthesis** khiến việc giả mạo giọng nói ngày càng dễ dàng. Vì vậy việc phát hiện **giọng nói giả mạo (spoofed speech)** là một vấn đề quan trọng trong lĩnh vực bảo mật và xử lý tín hiệu âm thanh.

Hệ thống này được xây dựng nhằm:

* Phân loại giọng nói thật và giọng nói giả
* Hỗ trợ nhiều mô hình học máy khác nhau
* Xử lý và trích xuất đặc trưng từ file âm thanh
* Cung cấp giao diện web để người dùng dễ dàng sử dụng

---

# 🧠 Các Mô Hình Được Sử Dụng

Hệ thống hỗ trợ nhiều loại mô hình đã được huấn luyện:

### Deep Learning

* WaveNet
* BiLSTM

### Machine Learning

* SVM
* CNN
* Các mô hình từ scikit-learn

Các định dạng mô hình được hỗ trợ:

```
.h5
.pkl
.joblib
```

Tất cả các mô hình được đặt trong thư mục **models/** và hệ thống sẽ **tự động load khi khởi động**.

---

# 🎧 Các Định Dạng Âm Thanh Hỗ Trợ

Người dùng có thể tải lên các định dạng:

```
wav
mp3
ogg
flac
mp4
```

Nếu người dùng tải lên **file mp4**, hệ thống sẽ tự động **trích xuất âm thanh từ video** bằng FFmpeg.

---

# ⚙️ Trích Xuất Đặc Trưng Âm Thanh

Hệ thống sử dụng đặc trưng:

### MFCC (Mel-Frequency Cepstral Coefficients)

MFCC giúp biểu diễn các đặc trưng phổ của tín hiệu âm thanh và thường được sử dụng trong các bài toán **nhận dạng giọng nói**.

Quy trình xử lý:

1. Tải file âm thanh
2. Chuẩn hóa tần số lấy mẫu
3. Trích xuất MFCC
4. Chuẩn hóa dữ liệu
5. Padding hoặc cắt độ dài tín hiệu
6. Đưa dữ liệu vào mô hình để dự đoán

Thư viện sử dụng:

* librosa
* numpy
* tensorflow
* scikit-learn

---

# 🌐 Ứng Dụng Web

Hệ thống có giao diện web được xây dựng bằng **Flask**.

Chức năng chính:

* Tải file âm thanh lên hệ thống
* Chọn mô hình đã huấn luyện
* Dự đoán giọng nói thật hoặc giả
* Hiển thị xác suất dự đoán của từng lớp

Luồng xử lý:

```
Upload Audio
      ↓
Trích xuất MFCC
      ↓
Chọn Model
      ↓
Dự đoán
      ↓
Hiển thị kết quả
```

---

# 📂 Cấu Trúc Thư Mục Dự Án

```
Fake-Voice-Detection/
│
├── app.py              # Ứng dụng web Flask
│
├── models/             # Các mô hình đã huấn luyện
│   ├── model.h5
│   ├── model.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── uploads/            # File âm thanh tải lên (tạm thời)
│
├── templates/
│   └── index.html      # Giao diện web
│
├── notebooks/          # Các file notebook huấn luyện và phân tích
│
└── README.md
```

---

# 🚀 Cài Đặt

Clone repository:

```
git clone https://github.com/your-username/fake-voice-detection.git
cd fake-voice-detection
```

Cài đặt các thư viện cần thiết:

```
pip install -r requirements.txt
```

Các thư viện chính:

```
flask
numpy
librosa
tensorflow
scikit-learn
joblib
```

---

# ▶️ Chạy Ứng Dụng

Khởi động server Flask:

```
python app.py
```

Sau đó mở trình duyệt tại địa chỉ:

```
http://127.0.0.1:5000
```

---

# 📊 Ví Dụ Kết Quả

Hệ thống sẽ trả về kết quả dự đoán:

```
Model: BiLSTM
Prediction: Fake
Confidence: 94.2%
```

Hoặc hiển thị xác suất của từng lớp.

---

# 👨‍💻 Tác Giả

Dự án nghiên cứu và học tập về **phát hiện giọng nói giả mạo bằng Machine Learning và Deep Learning**.

---

# 📜 License

Dự án được sử dụng cho **mục đích học tập và nghiên cứu**.
