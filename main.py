from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import uvicorn
from datetime import datetime
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
origins = [
    "http://localhost:3000",
    "http://localhost:3001"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOADS_DIR = "./uploads"

Path(UPLOADS_DIR).mkdir(parents=True, exist_ok=True)

# Define Pydantic BaseModel for request object
class WeatherPredictionRequest(BaseModel):
    year: int
    season: int
    area: float
    temperature: float
    humidity: float
    rainfall: float
    sunshine: float

# Load the machine learning model from .joblib file
def load_model(model_path):
    model = joblib.load(model_path)
    return model

# Make prediction using the loaded model
def make_prediction(model, input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)
    return prediction

@app.post("/weatherPrediction")
async def predict_info(request: WeatherPredictionRequest):
    try:
        # Load the model
        model_path = 'model_tea_weather.joblib'
        model = load_model(model_path)

        # Extract data from the request
        input_data = [request.year, request.season, request.area, request.temperature, 
                      request.rainfall, request.humidity, request.sunshine]
        
        # Make prediction
        prediction = make_prediction(model, input_data)
        
        # Convert prediction to a serializable format
        prediction_json = prediction.tolist()  # Assuming prediction is a NumPy array
        
        return {"prediction": prediction_json}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port="8000")
