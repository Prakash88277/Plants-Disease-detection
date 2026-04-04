# 🌱 Plant Disease Detection using CNN

A modern, highly accurate, and scalable Plant Disease Detection System powered by Convolutional Neural Networks (CNNs). This system utilizes a cascaded deep-learning approach to confidently classify plant species and diagnose diseases from leaf images, providing agriculturalists with a crucial tool to protect crops and ensure high output quality.

## # Overview

Agriculturalists often lose crops to illnesses that are difficult to track or identify with the naked eye. This project provides a robust solution with a **Cascaded Inference Pipeline** using two highly optimized CNN architectures:
1. **ResNet50:** Acts as the Stage 1 primary feature extractor to categorize the plant type.
2. **EfficientNetB0:** Acts as the Stage 2 high-precision classifier, efficiently identifying the exact disease on tasks with limited data constraints.

This hybrid technique ensures superior accuracy compared to traditional machine learning, diagnosing plant conditions accurately without requiring human intervention.

---

## # Features

- **Cascading Model Pipeline:** Dual-stage processing (ResNet50 → EfficientNetB0) maximizing output confidence.
- **FastAPI Backend:** Ultra-fast, synchronous model loading with asynchronous endpoints.
- **Modern Responsive Frontend UI:** A dynamic, beautiful web interface written in Vanilla HTML/CSS/JS ensuring maximum browser compatibility.
- **Image Upload & Live Prediction:** Smooth drag-and-drop mechanics supporting instant real-time diagnosis.
- **Model Performance Insights:** Expandable metric analyses, visualizing confusion matrices and loss/accuracy trends for both ResNet and EfficientNet dynamically.
- **Extremely Optimized:** Image telemetry processed mostly in-memory (PIL to Tensor pipelines) bridging frontend and backend flawlessly.

---

## # Tech Stack

- **Frontend:** Vanilla HTML5, CSS3, JavaScript
- **Backend:** FastAPI, Uvicorn, Python
- **Machine Learning Engine:** TensorFlow, Keras
- **Underlying Models:** ResNet50 (Transfer Learning), EfficientNetB0
- **Image Processing & Math:** OpenCV, NumPy, Pillow, Scikit-learn

---

## # Project Structure

```
├── Images/                 # Documentation and analysis visual graphs
├── backend/
│   ├── config/             # Environment values and Model Output Classes
│   ├── model/              # Singleton ML Loaders preventing reload latency  
│   ├── routes/             # FastAPI /predict controllers 
│   ├── saved_models/       # Tracked .h5 Deep Learning Weights
│   ├── src/                # Dataset loaders and underlying training templates
│   └── utils/              # Cascaded predictive pipeline parsing logic (pipeline.py)
├── frontend/
│   ├── css/                # Custom cascading stylesheets (Responsive & Animations)
│   ├── js/                 # API connection logic and DOM manipulation (script.js)
│   └── index.html          # Clean SaaS-style landing user interface
├── notebooks/              # Jupyter notebooks used for initial model training
└── .gitignore              # Defines ignored dependencies and __pycache__
```

---

## # How It Works

1. **User Upload:** An image is submitted via the frontend UI.
2. **Preprocessing:** The image is passed to memory, resized appropriately (`224x224`), and converted into model-ready arrays.
3. **Stage 1 - ResNet50:** Determines the underlying **plant** category, filtering invalid items.
4. **Stage 2 - EfficientNetB0:** Given the accepted plant leaf, it extracts targeted features to accurately identify the **disease**.
5. **Output Generation:** The cascaded inference parses out raw tensor output into human-readable strings (e.g., *Tomato / Early Blight / 98.4% Confidence*).

---

## # Dataset

We utilized the incredible PlantVillage dataset repository for evaluation and training:  
🔗 **[Plant Diseases Training Dataset (Kaggle)](https://www.kaggle.com/datasets/nirmalsankalana/plant-diseases-training-dataset)**

*Usage:* Over several thousands of leaf images categorized logically. This dataset powers the `58` class vectors identified by our settings configurations.

---

## # Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Prakash88277/Plants-Disease-detection.git
cd Plants-Disease-detection
```

### 2. Prepare the Python Environment
Ensure you have `Python 3.9` or similar installed.
```bash
python -m venv .venv
# Activate on Windows:
.venv\Scripts\activate
# Activate on Mac/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
Ensure TensorFlow and FastAPI environments are satisfied:
```bash
pip install fastapi uvicorn tensorflow pillow numpy opencv-python scikit-learn python-multipart matplotlib seaborn
```
*(Note: NumPy < 2.0 is highly recommended to avoid TensorFlow runtime clashes)*

### 4. Run the Backend (FastAPI Server)
```bash
cd backend
python main.py
```
*The API will boot up on `http://127.0.0.1:8000`.*

### 5. Run the Frontend
Simply open your preferred live server or directly load the HTML file:
```bash
cd ../frontend
# Double click index.html or run via Python HTTP:
python -m http.server 3000
```
*Open `http://localhost:3000/index.html` to view the UI.*

---

## # API Endpoint

### `POST /predict`
Uploads a file to process through the pipeline.
- **Input Content Type:** `multipart/form-data`
- **Payload Attribute:** `file` (Image buffer)
- **Output:**
```json
{
  "plant": "Tomato",
  "disease": "Early Blight",
  "confidence": "98.45%",
  "stage1_confidence": "92.12%",
  "prediction_time_ms": 320.15,
  "model_name": "Cascaded ResNet50 + EfficientNetB0"
}
```

---

## # Results

Our implementation drastically transforms how farming models analyze data. By employing transfer learning with **EfficientNetB0**, the pipeline retains an exceptional learning curve minimizing val-loss even upon extremely limited samples. Concurrently, **ResNet50** utilizes its deeper benchmarking accuracy to isolate general anomalies initially.

Combined, the inference pipeline avoids the strict limitations of human perception—averting traditional, slow, and potentially inaccurate visual inspections. To review confusion matrices and training histories for both respective components, run the frontend interface and click **"View Model Analysis"**.
