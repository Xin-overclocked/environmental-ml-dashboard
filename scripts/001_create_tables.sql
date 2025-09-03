-- Create tables for Environmental Engineering Dashboard

-- Training data table
CREATE TABLE IF NOT EXISTS training_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vehicle_model TEXT NOT NULL,
    vehicle_year INTEGER NOT NULL,
    engine_cc INTEGER NOT NULL,
    odometer_reading INTEGER NOT NULL,
    fuel_quality TEXT NOT NULL,
    fuel_brand TEXT NOT NULL,
    hc_ppm DECIMAL(10,2),
    co_ppm DECIMAL(10,2),
    nox_ppm DECIMAL(10,2),
    smoke_particulates DECIMAL(10,2),
    afr DECIMAL(10,2),
    lambda_value DECIMAL(10,2),
    o2_percent DECIMAL(10,2),
    rpm INTEGER,
    engine_load DECIMAL(10,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vehicle_model TEXT NOT NULL,
    vehicle_year INTEGER NOT NULL,
    engine_cc INTEGER NOT NULL,
    odometer_reading INTEGER NOT NULL,
    fuel_quality TEXT NOT NULL,
    fuel_brand TEXT NOT NULL,
    predicted_cleaning_time DECIMAL(10,2),
    predicted_hc_reduction DECIMAL(10,2),
    predicted_co_reduction DECIMAL(10,2),
    predicted_nox_reduction DECIMAL(10,2),
    predicted_particulates_reduction DECIMAL(10,2),
    co2_equivalent_saved DECIMAL(10,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_training_data_vehicle ON training_data(vehicle_model, vehicle_year);
CREATE INDEX IF NOT EXISTS idx_predictions_vehicle ON predictions(vehicle_model, vehicle_year);
CREATE INDEX IF NOT EXISTS idx_training_data_created ON training_data(created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at);
