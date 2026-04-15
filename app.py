# ---------------- LOGGING ----------------
import logging

logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

request_count = 0

# ---------------- IMPORTS ----------------
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import time
import joblib

# ---------------- AUTH ----------------
API_KEY = "mysecretkey"

# ---------------- APP ----------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- LOAD MODEL ----------------
model = None

try:
    model = joblib.load("model.joblib")
    print("Model loaded successfully")
except Exception as e:
    print("Model loading failed:", e)
    logging.error(f"Model loading failed: {str(e)}")
    model = None

# ---------------- INPUT SCHEMA ----------------
class DiabetesInput(BaseModel):
    pregnancies: float
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree_function: float
    age: float

# ---------------- HOME ROUTE ----------------
@app.get("/")
def home():
    return {"message": "API running successfully"}

# ---------------- PREDICT ROUTE ----------------
@app.post("/predict")
def predict(data: DiabetesInput, x_api_key: str = Header(None)):
    global request_count
    request_count += 1

    # API KEY CHECK
    if x_api_key != API_KEY:
        logging.warning("Unauthorized access attempt")
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # MODEL CHECK
    if model is None:
        return {"error": "Model not loaded properly"}

    try:
        start_time = time.time()

        logging.info(f"Request #{request_count} - Input: {data}")

        # Convert input to array
        input_array = np.array([[ 
            data.pregnancies,
            data.glucose,
            data.blood_pressure,
            data.skin_thickness,
            data.insulin,
            data.bmi,
            data.diabetes_pedigree_function,
            data.age
        ]])

        # Prediction
        prediction = model.predict(input_array)[0]
        result = "Diabetic" if prediction == 1 else "Not Diabetic"

        # Timing
        end_time = time.time()
        response_time = end_time - start_time

        logging.info(f"Prediction: {result}")
        logging.info(f"Response time: {response_time:.4f}s")

        return {
            "prediction": int(prediction),
            "result": result,
            "response_time": round(response_time, 4),
            "request_number": request_count
        }

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return {"error": str(e)}

# ---------------- GLOBAL ERROR HANDLER ----------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.error(f"Unhandled error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"message": "Something went wrong"}
    )