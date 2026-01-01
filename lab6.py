from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn
import os

# --------------------
# Configuration
# --------------------
MODEL_PATH = "titanic_best_model.joblib"

# --------------------
# FastAPI app
# --------------------
app = FastAPI(
    title="Titanic Survival Prediction API",
    description="Serve predictions from a pre-trained Titanic model",
    version="1.0"
)

# Allow common CORS origins (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------
# Load model
# --------------------
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded from", MODEL_PATH)
    except Exception as e:
        print("❌ Failed to load model:", e)
        model = None
else:
    print(f"❌ Model file not found at '{MODEL_PATH}'. Place the file in the same folder.")

# --------------------
# Request schema
# --------------------
# This matches the preprocessed features your model expects.
class PassengerData(BaseModel):
    Pclass: int
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    FamilySize: int
    IsAlone: int
    Sex_male: int
    Embarked_Q: int
    Embarked_S: int

# --------------------
# Routes
# --------------------
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Titanic Prediction API is up. Use /predict/ to POST passenger data."}

@app.post("/predict/")
def predict_survival(data: PassengerData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server. Check logs.")

    # Convert to DataFrame in same column order as trained model expects
    input_df = pd.DataFrame([data.dict()])

    try:
        pred = model.predict(input_df)[0]
        prob = None
        # Some models may not implement predict_proba (e.g., certain SVC kernels without probability=True)
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_df)[0][1]

        return {
            "prediction": int(pred),
            "probability_survived": round(float(prob), 4) if prob is not None else None,
            "result": "Survived" if int(pred) == 1 else "Did Not Survive"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

# Optional: endpoint to check loaded model type
@app.get("/model-info/")
def model_info():
    if model is None:
        return {"loaded": False, "message": f"No model loaded from {MODEL_PATH}"}
    return {"loaded": True, "model_type": type(model).__name__, "model_path": MODEL_PATH}

# --------------------
# Entrypoint
# --------------------
if __name__ == "__main__":
    # Starts the server if you run `python main.py`
    # For development with auto-reload use: `uvicorn main:app --reload`
    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info")
