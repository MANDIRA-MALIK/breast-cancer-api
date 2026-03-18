import os
import uvicorn
from fastapi import FastAPI, File, UploadFile
import numpy as np
from PIL import Image
import io
from tensorflow.keras.models import load_model

# Initialize app
app = FastAPI()

# Load model (FIXED)
model = load_model("model.h5", compile=False)

# Home route
@app.get("/")
def home():
    return {"message": "Breast Cancer API is running"}

# Prediction route
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # Convert image
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((224, 224))

        # Preprocess
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)

        # Result
        if prediction[0][0] > 0.5:
            result = "Malignant"
        else:
            result = "Benign"

        return {"prediction": result}

    except Exception as e:
        return {"error": str(e)}


# IMPORTANT: Render के लिए dynamic port binding
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
