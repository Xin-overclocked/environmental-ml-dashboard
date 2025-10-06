from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from supabase import create_client, Client

app = FastAPI(
    title="Environmental ML Dashboard API",
    description="Backend API for Environmental ML Dashboard",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class EmissionsMeasurement(BaseModel):
    vehicle_model: str
    vehicle_year: int
    meter_reading: float
    vehicle_cc: float
    fuel_quality: str
    fuel_brand: str
    hc_before: float
    hc_after: float
    co_before: float
    co_after: float
    nox_before: float
    nox_after: float
    o2_before: float
    o2_after: float

class EmissionsResult(BaseModel):
    hc_reduction: float
    co_reduction: float
    nox_reduction: float
    co2_equivalent: float

# Routes
@app.get("/")
async def root():
    return {"message": "Environmental ML Dashboard API"}

@app.post("/analyze-emissions", response_model=EmissionsResult)
async def analyze_emissions(measurement: EmissionsMeasurement):
    try:
        # Calculate reductions
        hc_reduction = measurement.hc_before - measurement.hc_after
        co_reduction = measurement.co_before - measurement.co_after
        nox_reduction = measurement.nox_before - measurement.nox_after

        # Convert to CO2 equivalent
        HC_TO_CO2_FACTOR = 2.5
        CO_TO_CO2_FACTOR = 1.8
        NOX_TO_CO2_FACTOR = 3.0

        co2_equivalent = (
            (hc_reduction * HC_TO_CO2_FACTOR) +
            (co_reduction * CO_TO_CO2_FACTOR) +
            (nox_reduction * NOX_TO_CO2_FACTOR)
        )

        return EmissionsResult(
            hc_reduction=hc_reduction,
            co_reduction=co_reduction,
            nox_reduction=nox_reduction,
            co2_equivalent=co2_equivalent
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)