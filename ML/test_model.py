import pandas as pd
import joblib
import os

def load_model():
    """بارگذاری مدل و اطلاعات"""
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, 'car_price_model.pkl')
    INFO_PATH = os.path.join(BASE_DIR, 'model_features.pkl')
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"❌ Model not found: {MODEL_PATH}\n"
            f"💡 Please run 'train_model.py' first to create the model!"
        )
    
    print("📥 Loading model...")
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
    
    if os.path.exists(INFO_PATH):
        info = joblib.load(INFO_PATH)
        print("✅ Model info loaded!")
    else:
        info = None
        print("⚠️ Model info not found (optional)")
    
    return model, info

def show_model_info(info):
    """نمایش اطلاعات مدل"""
    if not info:
        print("⚠️ No model info available")
        return
    
    print("\n" + "="*70)
    print("📊 Model Information")
    print("="*70)
    
    print(f"\n📅 Training Details:")
    print(f"   Date:              {info.get('training_date', 'N/A')}")
    print(f"   Total samples:     {info.get('sample_size', 'N/A'):,}")
    print(f"   Training samples:  {info.get('train_samples', 'N/A'):,}")
    print(f"   Test samples:      {info.get('test_samples', 'N/A'):,}")
    print(f"   Training time:     {info.get('train_time_seconds', 0)/60:.2f} minutes")
    
    print(f"\n🎯 Performance Metrics:")
    print(f"   Test MAE:          {info.get('test_mae', 0):,.0f} تومان ({info.get('test_mae', 0)/1_000_000:.2f}M)")
    print(f"   Test RMSE:         {info.get('test_rmse', 0):,.0f} تومان ({info.get('test_rmse', 0)/1_000_000:.2f}M)")
    print(f"   Test R²:           {info.get('test_r2', 0):.4f} ({info.get('test_r2', 0)*100:.2f}%)")
    print(f"   Test MAPE:         {info.get('test_mape', 0):.2f}%")
    
    if 'config' in info:
        config = info['config']
        print(f"\n⚙️ Model Configuration:")
        print(f"   n_estimators:      {config.get('n_estimators', 'N/A')}")
        print(f"   max_depth:         {config.get('max_depth', 'N/A')}")
        print(f"   sample_size:       {config.get('sample_size_used', 'N/A'):,}")
    
    print("="*70)

def predict_price(
    model,
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
    model : trained pipeline
        مدل آموزش‌دیده
    brand_main : str
        برند خودرو (مثال: پراید، پژو، سمند، ...)
    car_age : int
        سن خودرو به سال
    mileage : int
        کیلومتر کارکرد
    city : str
        شهر
    gearbox : str
        نوع گیربکس (دنده ای، اتوماتیک)
    fuel_type : str
        نوع سوخت (بنزینی، CNG، دوگانه‌سوز، ...)
    body_condition : str
        وضعیت بدنه
    engine_condition : str
        وضعیت موتور
    chassis_condition : str
        وضعیت شاسی
    
    Returns:
    --------
    float
        قیمت پیش‌بینی شده به تومان
    """
    
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

# ========================================
# اجرای اصلی برنامه
# ========================================

if __name__ == "__main__":
    print("="*70)
    print("🚗 Car Price Prediction System")
    print("="*70)
    
    try:
        # بارگذاری مدل
        model, info = load_model()
        
        # نمایش اطلاعات
        show_model_info(info)
        
        # ========================================
        # مثال‌های پیش‌بینی
        # ========================================
        
        print("\n" + "="*70)
        print("🔮 Sample Predictions")
        print("="*70)
        
        # تعریف مثال‌ها
        examples = [
            {
                'name': '1️⃣ پراید 131 - مدل 1396 (7 سال، کم‌کار)',
                'params': {
                    'brand_main': 'پراید',
                    'car_age': 7,
                    'mileage': 80000,
                    'city': 'tehran',
                    'gearbox': 'دنده ای',
                    'fuel_type': 'بنزینی',
                    'body_condition': 'سالم و بی خط و خش',
                    'engine_condition': 'سالم',
                    'chassis_condition': 'سالم و پلمپ'
                }
            },
            {
                'name': '2️⃣ پژو 405 - مدل 1393 (10 سال)',
                'params': {
                    'brand_main': 'پژو',
                    'car_age': 10,
                    'mileage': 150000,
                    'city': 'tehran',
                    'gearbox': 'دنده ای',
                    'fuel_type': 'بنزینی',
                    'body_condition': 'خط و خش جزیی',
                    'engine_condition': 'سالم',
                    'chassis_condition': 'سالم و پلمپ'
                }
            },
            {
                'name': '3️⃣ سمند - مدل 1398 (5 سال، دوگانه)',
                'params': {
                    'brand_main': 'سمند',
                    'car_age': 5,
                    'mileage': 60000,
                    'city': 'tehran',
                    'gearbox': 'دنده ای',
                    'fuel_type': 'دوگانه‌سوز',
                    'body_condition': 'سالم و بی خط و خش',
                    'engine_condition': 'سالم',
                    'chassis_condition': 'سالم و پلمپ'
                }
            },
            {
                'name': '4️⃣ پراید صفر - مدل 1402 (1 سال)',
                'params': {
                    'brand_main': 'پراید',
                    'car_age': 1,
                    'mileage': 0,
                    'city': 'tehran',
                    'gearbox': 'دنده ای',
                    'fuel_type': 'بنزینی',
                    'body_condition': 'سالم و بی خط و خش',
                    'engine_condition': 'سالم',
                    'chassis_condition': 'سالم و پلمپ'
                }
            },
            {
                'name': '5️⃣ تویوتا کمری - مدل 1395 (8 سال)',
                'params': {
                    'brand_main': 'تویوتا',
                    'car_age': 8,
                    'mileage': 120000,
                    'city': 'tehran',
                    'gearbox': 'اتوماتیک',
                    'fuel_type': 'بنزینی',
                    'body_condition': 'سالم و بی خط و خش',
                    'engine_condition': 'سالم',
                    'chassis_condition': 'سالم و پلمپ'
                }
            },
            {
                'name': '6️⃣ پژو پارس - مدل 1385 (18 سال، پرکار)',
                'params': {
                    'brand_main': 'پژو',
                    'car_age': 18,
                    'mileage': 250000,
                    'city': 'tehran',
                    'gearbox': 'دنده ای',
                    'fuel_type': 'بنزینی',
                    'body_condition': 'تمام رنگ',
                    'engine_condition': 'سالم',
                    'chassis_condition': 'سالم و پلمپ'
                }
            },
            {
                'name': '7️⃣ سمند EF7 - مدل 1390 (13 سال)',
                'params': {
                    'brand_main': 'سمند',
                    'car_age': 13,
                    'mileage': 180000,
                    'city': 'tehran',
                    'gearbox': 'دنده ای',
                    'fuel_type': 'بنزینی',
                    'body_condition': 'خط و خش جزیی',
                    'engine_condition': 'سالم'
                }
            }
        ]
        
        # پیش‌بینی برای همه مثال‌ها
        predictions = []
        
        print("\n")
        for example in examples:
            try:
                price = predict_price(model, **example['params'])
                predictions.append((example['name'], price))
                
                print(f"{example['name']}")
                print(f"   💰 Price: {price:>15,.0f} تومان")
                print(f"   💰 Price: {price/1_000_000:>15.2f} میلیون تومان")
                print()
                
            except Exception as e:
                print(f"{example['name']}")
                print(f"   ❌ Error: {e}\n")
        
        # ========================================
        # مقایسه قیمت‌ها
        # ========================================
        
        if predictions:
            print("="*70)
            print("📊 Price Comparison (Sorted by Price)")
            print("="*70)
            
            sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
            
            max_price = max(p[1] for p in predictions)
            
            print()
            for name, price in sorted_predictions:
                # نمودار میله‌ای ساده
                bar_length = int((price / max_price) * 40)
                bar = '█' * bar_length
                
                # حذف شماره از نام برای نمایش بهتر
                clean_name = name.split(' ', 1)[1] if ' ' in name else name
                
                print(f"{clean_name:40s} {price/1_000_000:>6.1f}M {bar}")
        
        # ========================================
        # راهنمای استفاده
        # ========================================
        
        print("\n" + "="*70)
        print("💡 How to Use")
        print("="*70)
        
        print("""
📝 Example 1: Single Prediction
--------------------------------
from test_model import load_model, predict_price

model, _ = load_model()

price = predict_price(
    model,
    brand_main='پژو',
    car_age=12,
    mileage=180000,
    city='tehran',
    gearbox='دنده ای',
    fuel_type='بنزینی'
)

print(f"قیمت: {price:,.0f} تومان")


📝 Example 2: Batch Prediction
--------------------------------
import pandas as pd

model, _ = load_model()

# ساخت DataFrame با چند خودرو
cars = pd.DataFrame([
    {
        'Brand_Main': 'پراید',
        'Car_Age': 5,
        'Mileage': 70000,
        'City': 'tehran',
        'Gearbox': 'دنده ای',
        'Fuel_Type': 'بنزینی',
        'Body_Condition': 'سالم و بی خط و خش',
        'Engine_Condition': 'سالم',
        'Chassis_Condition': 'سالم و پلمپ'
    },
    {
        'Brand_Main': 'پژو',
        'Car_Age': 8,
        'Mileage': 120000,
        'City': 'mashhad',
        'Gearbox': 'دنده ای',
        'Fuel_Type': 'دوگانه‌سوز',
        'Body_Condition': 'خط و خش جزیی',
        'Engine_Condition': 'سالم',
        'Chassis_Condition': 'سالم و پلمپ'
    }
])

# پیش‌بینی برای همه
predictions = model.predict(cars)

# اضافه کردن به DataFrame
cars['Predicted_Price'] = predictions

print(cars[['Brand_Main', 'Car_Age', 'Predicted_Price']])
""")
        
        print("="*70)
        print("✅ Test completed successfully!")
        print("="*70)
        
    except FileNotFoundError as e:
        print(f"\n{e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()