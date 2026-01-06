# 🏠 Dự Đoán Định Giá Bất Động Sản (Real Estate Price Prediction)

## 📋 Giới Thiệu Dự Án

Dự án này xây dựng và so sánh **3 phương pháp Machine Learning** để dự đoán giá nhà dựa trên các đặc trưng bất động sản:
- 🔹 **Linear Regression** (Hồi quy tuyến tính - Baseline)
- 🔹 **Random Forest Regressor** (Mô hình Tree-based)
- 🔹 **Artificial Neural Network (ANN)** (Deep Learning)

## 🎯 Ứng Dụng Thực Tế

### Bài Toán Kinh Doanh
- **Cho người bán**: Định giá nhà hợp lý để bán nhanh
- **Cho người mua**: Tránh mua hớ, đánh giá giá trị thực
- **Cho ngân hàng**: Định giá tài sản thế chấp chính xác
- **Cho sàn BĐS**: Xây dựng hệ thống định giá tự động (như Batdongsan.com.vn, Zillow)

### Tính Thực Tế Cao
Đây là bài toán cốt lõi của các trang web bất động sản, có thể demo trực quan:
- **Input**: Diện tích, số phòng, vị trí, tuổi nhà...
- **Output**: Giá tiền dự đoán

## 📊 Dataset

**California Housing Dataset** - Dataset nổi tiếng với 20,640 mẫu dữ liệu

### Các Đặc Trưng (Features):
| Feature | Mô Tả | Đơn Vị |
|---------|-------|---------|
| `MedInc` | Thu nhập trung bình | $10,000 |
| `HouseAge` | Tuổi nhà | Năm |
| `AveRooms` | Số phòng trung bình | Phòng |
| `AveBedrms` | Số phòng ngủ trung bình | Phòng |
| `Population` | Dân số khu vực | Người |
| `AveOccup` | Số người/hộ trung bình | Người |
| `Latitude` | Vĩ độ | Độ |
| `Longitude` | Kinh độ | Độ |

### Target Variable:
- `MedHouseVal`: Giá nhà trung bình (đơn vị: $100,000)

## 🛠️ Cài Đặt

### 1. Clone hoặc tải về dự án

```bash
cd "Real Estate Price Prediction"
```

### 2. Cài đặt các thư viện cần thiết

```bash
pip install -r requirements.txt
```

### 3. Mở Jupyter Notebook

```bash
jupyter notebook real_estate_price_prediction.ipynb
```

### 4. (Optional) Chạy Web App Demo

```bash
# Cài thêm streamlit
pip install -r requirements_app.txt

# Chạy web app
streamlit run app.py
```

Sau khi train xong models trong notebook, bạn có thể mở web interface tại `http://localhost:8501`

## 📈 Quy Trình Thực Hiện

### 1️⃣ **Data Loading & Exploration**
- Load California Housing Dataset
- Kiểm tra dữ liệu thiếu
- Phân tích thống kê mô tả

### 2️⃣ **Exploratory Data Analysis (EDA)**
- Phân phối giá nhà
- Ma trận tương quan
- Visualization theo địa lý
- Phân tích đặc trưng

### 3️⃣ **Data Preprocessing**
- Train-Test Split (80-20)
- Feature Scaling với StandardScaler

### 4️⃣ **Model Training & Evaluation**

#### Model 1: Linear Regression
- Phương pháp baseline đơn giản
- Giả định quan hệ tuyến tính
- Nhanh, dễ giải thích

#### Model 2: Random Forest Regressor
- Xử lý tốt quan hệ phi tuyến
- Ensemble của nhiều Decision Trees
- Robust với outliers

#### Model 3: Artificial Neural Network (ANN)
- Deep Learning với TensorFlow/Keras
- Architecture: 128 → 64 → 32 → 1
- Dropout layers để tránh overfitting
- Early Stopping để tối ưu training

### 5️⃣ **Model Comparison**
So sánh 3 models dựa trên các chỉ số:
- **RMSE** (Root Mean Squared Error): Sai số bình phương trung bình
- **MAE** (Mean Absolute Error): Sai số tuyệt đối trung bình
- **R² Score**: Hệ số xác định (càng gần 1 càng tốt)

### 6️⃣ **Visualization**
- Actual vs Predicted plots
- Residual plots
- Feature importance
- Model comparison charts

### 7️⃣ **Demo & Deployment**
- Function dự đoán giá real-time
- Save models cho production
- Example predictions

## 🏆 Kết Quả Dự Kiến

### Performance Metrics (Test Set):

| Model | RMSE | MAE | R² Score |
|-------|------|-----|----------|
| Linear Regression | ~0.73 | ~0.53 | ~0.60 |
| Random Forest | ~0.50 | ~0.33 | ~0.81 |
| ANN | ~0.55 | ~0.38 | ~0.77 |

### Key Insights:
1. **Random Forest** thường cho kết quả tốt nhất
2. **Median Income** là yếu tố quan trọng nhất
3. **Vị trí địa lý** ảnh hưởng mạnh đến giá
4. ANN có thể tốt hơn với dataset lớn hơn

## 📁 Cấu Trúc Dự Án

```
Real Estate Price Prediction/
│
├── README.md                              # File này
├── requirements.txt                       # Dependencies
├── real_estate_price_prediction.ipynb    # Jupyter Notebook chính
├── predict.py                            # Script dự đoán standalone
│
└── models/                               # Thư mục lưu models
    ├── linear_regression_model.pkl
    ├── random_forest_model.pkl
    ├── ann_model.h5
    └── scaler.pkl
```

## 🚀 Cách Sử Dụng Models Đã Train

```python
import joblib
import numpy as np
from tensorflow import keras

# Load models
lr_model = joblib.load('models/linear_regression_model.pkl')
rf_model = joblib.load('models/random_forest_model.pkl')
ann_model = keras.models.load_model('models/ann_model.h5')
scaler = joblib.load('models/scaler.pkl')

# Prepare features
features = np.array([[
    3.5,      # MedInc
    15,       # HouseAge
    6,        # AveRooms
    1.2,      # AveBedrms
    1200,     # Population
    3,        # AveOccup
    34.05,    # Latitude
    -118.25   # Longitude
]])

# Scale
features_scaled = scaler.transform(features)

# Predict
price_lr = lr_model.predict(features_scaled)[0]
price_rf = rf_model.predict(features_scaled)[0]
price_ann = ann_model.predict(features_scaled)[0][0]

print(f"Linear Regression: ${price_lr*100000:,.0f}")
print(f"Random Forest: ${price_rf*100000:,.0f}")
print(f"ANN: ${price_ann*100000:,.0f}")
```

## 📊 Đánh Giá & So Sánh

### Ưu Điểm Từng Model:

**Linear Regression:**
- ✅ Đơn giản, dễ hiểu
- ✅ Training nhanh
- ✅ Giải thích được feature importance
- ❌ Giả định tuyến tính không thực tế

**Random Forest:**
- ✅ Accuracy cao nhất
- ✅ Xử lý tốt non-linear relationships
- ✅ Robust với outliers
- ❌ Model phức tạp, khó giải thích

**ANN:**
- ✅ Học được patterns phức tạp
- ✅ Scalable với big data
- ✅ Flexible architecture
- ❌ Cần nhiều dữ liệu
- ❌ Training lâu hơn
- ❌ Black box (khó giải thích)

## 🎓 Kiến Thức Áp Dụng

### Machine Learning:
- Regression problems
- Feature engineering
- Train-test split
- Cross-validation
- Hyperparameter tuning

### Deep Learning:
- Neural network architecture
- Backpropagation
- Activation functions (ReLU)
- Regularization (Dropout)
- Optimization (Adam)

### Data Science:
- Exploratory Data Analysis (EDA)
- Data visualization
- Statistical analysis
- Model evaluation metrics

## 💻 Requirements

- Python 3.8+
- numpy
- pandas
- scikit-learn
- tensorflow
- matplotlib
- seaborn
- jupyter

## 📝 Tác Giả & Liên Hệ

Dự án này được xây dựng cho mục đích học tập và demo.

## 📄 License

MIT License - Free to use for educational purposes

## 🙏 Tài Liệu Tham Khảo

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [California Housing Dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)

---

**⭐ Nếu dự án hữu ích, đừng quên star repo nhé! ⭐**
