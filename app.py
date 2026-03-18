from fastapi import FastAPI, File, UploadFile
import numpy as np
from PIL import Image
import io

from tensorflow.keras.models import load_model

# Initialize app
app = FastAPI()

# Load model
model = load_model("model.h5", compile=False)

# Home route
@app.get("/")
def home():
    return {"message": "Breast Cancer API is running"}

# Prediction route
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")

        # Resize (IMPORTANT: same size as training)
        image = image.resize((224, 224))

        # Convert to array
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = model.predict(img_array)

        # Binary classification
        if prediction[0][0] > 0.5:
            result = "Malignant"
        else:
            result = "Benign"

        return {
            "prediction": result,
            "confidence": float(prediction[0][0])
        }

    except Exception as e:
        return {"error": str(e)}
