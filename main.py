from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
from google.cloud import firestore
from uuid import uuid4
from datetime import datetime
import numpy as np
from PIL import Image

# Inisialisasi Firestore
db = firestore.Client()

# Inisialisasi FastAPI
app = FastAPI()

# Muat model TensorFlow
model = tf.keras.models.load_model("model")

# Endpoint untuk prediksi gambar
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.size > 1000000:  # Validasi ukuran file
        return {"status": "fail", "error": "File terlalu besar. Maksimum 1MB"}
    
    try:
        image = Image.open(file.file)
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        prediction = model.predict(image_array)
        label = "Cancer" if prediction[0][0] > 0.5 else "Non-cancer"
        suggestion = "Segera periksa ke dokter!" if label == "Cancer" else "Anda sehat!"
        
        prediction_id = str(uuid4())
        db.collection("predictions").document(prediction_id).set({
            "id": prediction_id,
            "result": label,
            "suggestion": suggestion,
            "createdAt": datetime.utcnow().isoformat()
        })
        
        return {"status": "success", "result": label, "suggestion": suggestion, "id": prediction_id}

    except Exception as e:
        return {"status": "fail", "error": str(e)}

# Endpoint untuk riwayat prediksi
@app.get("/predict/histories")
async def get_histories():
    try:
        docs = db.collection("predictions").stream()
        histories = []
        for doc in docs:
            data = doc.to_dict()
            histories.append({
                "id": data["id"],
                "history": {
                    "result": data["result"],
                    "createdAt": data["createdAt"],
                    "suggestion": data["suggestion"],
                    "id": data["id"]
                }
            })
        return {"status": "success", "data": histories}
    except Exception as e:
        return {"status": "fail", "error": str(e)}
