#Exp10: Monitoring & Logging 
#loggiong in
import logging

logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

request_count = 0

#Exp 2
from fastapi import FastAPI, Request   # ✅ added Request here (needed below)
from pydantic import BaseModel
import pickle
import numpy as np
import time
from fastapi import Header, HTTPException
from fastapi.responses import JSONResponse  # ✅ added here (needed below)

#Expt 4: Authentication (API Key)
API_KEY = "mysecretkey"

# Create app
app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
with open("final_diabetes_model.pkl", "rb") as file:
    model = pickle.load(file)

# Input schema
class DiabetesInput(BaseModel):
    pregnancies: float
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree_function: float
    age: float

# Home route
@app.get("/")
def home():
    return {"message": "API running successfully"}

# Prediction route
@app.post("/predict")
def predict(data: DiabetesInput, x_api_key: str = Header(None)):
    global request_count
    request_count += 1

    # Authentication
    if x_api_key != API_KEY:
        logging.warning("Unauthorized access attempt")
        raise HTTPException(status_code=401, detail="Invalid API Key")

    try:
        # ⏱ START TIME
        start_time = time.time()

        # Log input
        logging.info(f"Request #{request_count} - Received input: {data}")

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

        prediction = model.predict(input_array)[0]
        result = "Diabetic" if prediction == 1 else "Not Diabetic"

        # Log result
        logging.info(f"Prediction result: {result}")

        # ⏱ END TIME
        end_time = time.time()
        response_time = end_time - start_time

        # Log response time
        logging.info(f"Response time: {response_time:.4f} seconds")

        return {
            "prediction": int(prediction),
            "result": result,
            "response_time": round(response_time, 4),
            "request_number": request_count
        }

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return {"error": "Something went wrong"}


#EXP 3 - Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.error(f"Unhandled error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"message": "Something went wrong"}
    )