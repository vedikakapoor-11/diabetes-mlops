#Exp 3 LOGGING SETUP
import logging

logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

#Exp 2
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

from fastapi import Header, HTTPException

#Expt 4: Authentication (API Key)
API_KEY = "mysecretkey"
# Create app
app = FastAPI()

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
   #Expt 4: Authentication (API Key)
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    try:
        # Log input
        logging.info(f"Received input: {data}")

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

        # Log output
        logging.info(f"Prediction result: {result}")

        return {
            "prediction": int(prediction),
            "result": result
        }

    except Exception as e:
        # Log error
        logging.error(f"Error occurred: {str(e)}")
        return {"error": str(e)}
    
#EXP 3 - Global error handler
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import pickle
import numpy as np
import logging
# EXP 3 - Global error handler

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.error(f"Unhandled error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"message": "Something went wrong"}
    )