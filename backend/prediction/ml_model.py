import joblib
import pandas as pd
from datetime import datetime

model = joblib.load("ML/car_price_model.pkl")

def extract_brand_main(brand_text: str) -> str:
    if not brand_text:
        return "unknown"
    return brand_text.split("،")[0].strip()


def predict_price(data: dict):

    current_year = 1403  # یا داینامیک
    car_age = current_year - int(data["Model_Year"])

    brand_main = extract_brand_main(data["Brand"])

    input_data = {
        "Brand_Main": brand_main,
        "Car_Age": car_age,
        "Mileage": data["Mileage"],
        "Gearbox": data["Gearbox"],
        "Fuel_Type": data["Fuel_Type"],
        "Body_Condition": data["Body_Condition"],
        "Engine_Condition": data["Engine_Condition"],
        "Chassis_Condition": data["Chassis_Condition"],
        "City": data["City"],
    }

    df = pd.DataFrame([input_data])

    predicted_price = model.predict(df)[0]
    return int(predicted_price)
