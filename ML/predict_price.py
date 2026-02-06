import pandas as pd
import joblib
import os

def predict_car_price(
    brand_main='پراید',
    car_age=10,
    mileage=100000,
    city='tehran',
    gearbox='دنده ای',
    fuel_type='بنزینی',
    body_condition='سالم و بی خط و خش',
    engine_condition='سالم',
    chassis_condition='سالم و پلمپ'
):
    """
    پیش‌بینی قیمت خودرو
    
    Parameters:
    -----------
    brand_main : str - برند (مثلا: پراید، پژو، سمند)
    car_age : int - سن خودرو (سال)
    mileage : int - کیلومتر
    city : str - شهر
    gearbox : str - گیربکس (دنده ای، اتوماتیک)
    fuel_type : str - نوع سوخت (بنزینی، CNG، هیبریدی)
    body_condition : str - وضعیت بدنه
    engine_condition : str - وضعیت موتور
    chassis_condition : str - وضعیت شاسی
    
    Returns:
    --------
    float : قیمت پیش‌بینی شده (تومان)
    """
    
    # بارگذاری مدل
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, 'car_price_model.pkl')
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model not found at: {MODEL_PATH}")
    
    model = joblib.load(MODEL_PATH)
    
    # ساخت DataFrame ورودی
    input_data = pd.DataFrame([{
        'Brand_Main': brand_main,
        'Car_Age': car_age,
        'Mileage': mileage,
        'City': city,
        'Gearbox': gearbox,
        'Fuel_Type': fuel_type,
        'Body_Condition': body_condition,
        'Engine_Condition': engine_condition,
        'Chassis_Condition': chassis_condition
    }])
    
    # پیش‌بینی
    predicted_price = model.predict(input_data)[0]
    
    return predicted_price


# ===== مثال استفاده =====

if __name__ == "__main__":
    print("🚗 Car Price Prediction System")
    print("="*50)
    
    # مثال 1: پراید 132 مدل 1396
    price1 = predict_car_price(
        brand_main='پراید',
        car_age=7,  # 1403 - 1396
        mileage=80000,
        city='tehran',
        gearbox='دنده ای',
        fuel_type='بنزینی',
        body_condition='سالم و بی خط و خش',
        engine_condition='سالم',
        chassis_condition='سالم و پلمپ'
    )
    print(f"\n🔹 پراید 132 مدل 1396:")
    print(f"   قیمت پیش‌بینی: {price1:,.0f} تومان")
    print(f"   قیمت پیش‌بینی: {price1/10:,.0f} هزار تومان")
    
    # مثال 2: پژو 405 مدل 1393
    price2 = predict_car_price(
        brand_main='پژو',
        car_age=10,  # 1403 - 1393
        mileage=150000,
        city='tehran',
        gearbox='دنده ای',
        fuel_type='بنزینی',
        body_condition='خط و خش جزیی',
        engine_condition='سالم',
        chassis_condition='سالم و پلمپ'
    )
    print(f"\n🔹 پژو 405 مدل 1393:")
    print(f"   قیمت پیش‌بینی: {price2:,.0f} تومان")
    print(f"   قیمت پیش‌بینی: {price2/10:,.0f} هزار تومان")
    
    # مثال 3: سمند مدل 1398
    price3 = predict_car_price(
        brand_main='سمند',
        car_age=5,
        mileage=60000,
        city='tehran',
        gearbox='دنده ای',
        fuel_type='بنزینی',
        body_condition='سالم و بی خط و خش',
        engine_condition='سالم',
        chassis_condition='سالم و پلمپ'
    )
    print(f"\n🔹 سمند مدل 1398:")
    print(f"   قیمت پیش‌بینی: {price3:,.0f} تومان")
    print(f"   قیمت پیش‌بینی: {price3/10:,.0f} هزار تومان")
    
    print("\n" + "="*50)
    
    # پیش‌بینی تعاملی
    print("\n📝 Custom Prediction:")
    try:
        custom_price = predict_car_price(
            brand_main='پراید',
            car_age=5,
            mileage=50000,
            city='tehran'
        )
        print(f"قیمت پیش‌بینی: {custom_price:,.0f} تومان")
    except Exception as e:
        print(f"❌ Error: {e}")