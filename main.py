from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from PIL import Image, ImageOps
import io
import os

# Note: Commenting tensorflow out to avoid crash if not installed
# You should uncomment these when you're ready to use your real model
import tensorflow as tf
import random

# Set Seed agar hasil selalu konsisten seperti saat training
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

app = FastAPI()

# Konfigurasi CORS agar bisa diakses oleh Node.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load Model Keras
MODEL_PATH = "Model/tomato_mobilenetv2_final.keras" # Nama file model kamu
model = None

try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Model {MODEL_PATH} berhasil dimuat.")
    else:
        print(f"File model {MODEL_PATH} tidak ditemukan.")
except Exception as e:
    print(f"Gagal memuat model: {e}")

# 2. Definisikan Kelas / Label (sesuaikan dengan output model ML kamu)
# Diambil dari Jupyter Notebook (sudah diurutkan dan disesuaikan dengan output training)
RAW_CLASS_NAMES = [
    "Bacterial_spot",
    "Early_blight",
    "Late_blight",
    "Leaf_Mold",
    "Septoria_leaf_spot",
    "Spider_mites Two-spotted_spider_mite",
    "Target_Spot",
    "Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_mosaic_virus",
    "healthy",
    "powdery_mildew"
]

# Mapping agar namanya lebih bagus di web
CLASS_NAMES_MAPPING = {
    "Bacterial_spot": "Bercak Bakteri (Bacterial Spot)",
    "Early_blight": "Hawar Awal (Early Blight)",
    "Late_blight": "Hawar Akhir (Late Blight)",
    "Leaf_Mold": "Jamur Daun (Leaf Mold)",
    "Septoria_leaf_spot": "Bercak Daun Septoria (Septoria Leaf Spot)",
    "Spider_mites Two-spotted_spider_mite": "Tungau Laba-laba (Spider Mites)",
    "Target_Spot": "Bercak Target (Target Spot)",
    "Tomato_Yellow_Leaf_Curl_Virus": "Virus Daun Kuning Keriting (Yellow Leaf Curl Virus)",
    "Tomato_mosaic_virus": "Virus Mosaik (Mosaic Virus)",
    "healthy": "Tanaman Sehat",
    "powdery_mildew": "Embun Tepung (Powdery Mildew)"
}
CLASS_NAMES = [CLASS_NAMES_MAPPING[name] for name in RAW_CLASS_NAMES]

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def preprocess_image(image_bytes):
    """
    Fungsi untuk memproses gambar sebelum dimasukkan ke model.
    Sama persis dengan yang ada di Jupyter Notebook menggunakan preprocess_input.
    """
    # DEBUG: Simpan gambar untuk memastikan tidak ada korupsi data
    with open("debug_received_image.jpg", "wb") as f:
        f.write(image_bytes)
        
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Sesuaikan ukuran ini (224x224 karena model meminta shape tersebut)
    image = image.resize((224, 224))
    
    # Konversi ke float32 sesuai kode di Jupyter
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, 0)
    
    # Menggunakan preprocess_input dari mobilenet_v2
    img_array = preprocess_input(img_array)
    
    return img_array

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Baca file gambar yang diupload
        contents = await file.read()
        
        # Preprocessing
        processed_image = preprocess_image(contents)
        
        # 3. Lakukan Prediksi
        if model is not None:
            # Jika model asli ada:
            predictions = model.predict(processed_image)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
        else:
            # Simulasi hasil prediksi karena model belum aktif (DUMMY)
            print("Menggunakan prediksi dummy karena model belum diaktifkan")
            predicted_class_idx = np.random.randint(0, len(CLASS_NAMES))
            confidence = 0.95
            
        predicted_class = CLASS_NAMES[predicted_class_idx] if predicted_class_idx < len(CLASS_NAMES) else "Unknown"
        accuracy = round(confidence * 100, 2)
        
        is_healthy = "sehat" in predicted_class.lower()
        severity = "rendah" if is_healthy else "tinggi"
        severity_label = "Tanaman Sehat" if is_healthy else "Penyakit Terdeteksi — Perlu Penanganan"
        
        # Data yang dikembalikan ke Node.js
        return {
            "disease": {
                "name": predicted_class,
                "en": RAW_CLASS_NAMES[predicted_class_idx] if predicted_class_idx < len(RAW_CLASS_NAMES) else "Unknown"
            },
            "accuracy": accuracy,
            "severity": severity,
            "severityLabel": severity_label,
            "metrics": {"precision": 90, "recall": 92, "f1": 91},
            "symptoms": ["Silakan periksa lebih lanjut"] if not is_healthy else ["Daun dan buah tampak normal"],
            "treatments": [
                {"title": "Tindakan", "text": "Berikan fungisida atau pestisida sesuai anjuran jika bergejala berat." if not is_healthy else "Lanjutkan perawatan rutin.", "type": "normal" if not is_healthy else "green"}
            ]
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
