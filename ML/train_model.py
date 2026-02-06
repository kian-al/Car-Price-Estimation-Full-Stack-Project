import pandas as pd
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import os
import time
import numpy as np

# ========================================
# 🎯 تنظیمات اصلی
# ========================================

SAMPLE_SIZE = 200_000  # 👈 تعداد سطرهایی که می‌خوای استفاده کنی
USE_SAMPLING = True     # 👈 True = استفاده از نمونه | False = استفاده از همه

# ========================================
# توابع کمکی
# ========================================

def persian_to_english_numbers(text):
    """تبدیل اعداد فارسی به انگلیسی"""
    if pd.isna(text):
        return text
    text = str(text)
    persian_digits = '۰۱۲۳۴۵۶۷۸۹'
    english_digits = '0123456789'
    translation = str.maketrans(persian_digits, english_digits)
    text = text.replace('٬', '').replace(',', '').replace(' ', '')
    return text.translate(translation)

def extract_brand(brand_text):
    """استخراج برند اصلی"""
    if pd.isna(brand_text):
        return 'نامشخص'
    brand_text = str(brand_text)
    brand = brand_text.split('،')[0].split(',')[0].strip()
    return brand if brand else 'نامشخص'

# ========================================
# شروع برنامه
# ========================================

print("="*70)
print("🚗 Car Price Prediction Model - Training System")
print("="*70)
print(f"📊 Configuration:")
print(f"   Sample size: {SAMPLE_SIZE:,} rows")
print(f"   Use sampling: {USE_SAMPLING}")
print("="*70)

start_time = time.time()

# ========================================
# 🔹 بارگذاری داده
# ========================================

print("\n📥 Loading dataset...")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, 'final2024.csv')

# بررسی وجود فایل
if not os.path.exists(CSV_PATH):
    print(f"❌ File not found: {CSV_PATH}")
    print(f"💡 Expected location: {os.path.abspath(CSV_PATH)}")
    print("💡 Make sure 'final2024.csv' is in the correct folder")
    exit(1)

df = pd.read_csv(CSV_PATH, low_memory=False)

print(f"✅ Dataset loaded successfully!")
print(f"   Total rows: {len(df):,}")
print(f"   Total columns: {len(df.columns)}")
print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# ========================================
# 🔹 پاک‌سازی داده
# ========================================

print("\n" + "="*70)
print("🧹 Data Cleaning...")
print("="*70)

# 1. تبدیل اعداد فارسی
print("\n1️⃣ Converting Persian numbers to English...")
for col in ['Model_Year', 'Mileage', 'Price']:
    if col in df.columns:
        df[col] = df[col].apply(persian_to_english_numbers)
print("   ✅ Conversion completed")

# 2. تبدیل به عددی
print("\n2️⃣ Converting to numeric types...")
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df['Model_Year'] = pd.to_numeric(df['Model_Year'], errors='coerce')
df['Mileage'] = pd.to_numeric(df['Mileage'], errors='coerce')

# 3. فیلتر قیمت معتبر
print("\n3️⃣ Filtering valid prices...")
initial_count = len(df)
df = df[df['Price'].notna()]
df = df[(df['Price'] > 10_000_000) & (df['Price'] < 10_000_000_000)]
print(f"   Valid prices: {len(df):,} / {initial_count:,} ({len(df)/initial_count*100:.1f}%)")

# 4. استخراج برند
print("\n4️⃣ Extracting main brand...")
df['Brand_Main'] = df['Brand'].apply(extract_brand)
unique_brands = df['Brand_Main'].nunique()
top_brands = df['Brand_Main'].value_counts().head(5)
print(f"   Unique brands: {unique_brands}")
print(f"   Top 5 brands:")
for brand, count in top_brands.items():
    print(f"      - {brand}: {count:,} ({count/len(df)*100:.1f}%)")

# 5. محاسبه سن خودرو
print("\n5️⃣ Calculating car age...")
current_year = 1403
df['Car_Age'] = current_year - df['Model_Year']
df = df[(df['Car_Age'] >= 0) & (df['Car_Age'] <= 50)]
print(f"   Age range: {df['Car_Age'].min():.0f} - {df['Car_Age'].max():.0f} years")
print(f"   Average age: {df['Car_Age'].mean():.1f} years")

# 6. مدیریت کیلومتر
print("\n6️⃣ Handling mileage...")
mileage_median = df['Mileage'].median()
outliers_count = len(df[(df['Mileage'] < 0) | (df['Mileage'] > 1_000_000)])
df.loc[df['Mileage'] < 0, 'Mileage'] = mileage_median
df.loc[df['Mileage'] > 1_000_000, 'Mileage'] = mileage_median
print(f"   Mileage median: {mileage_median:,.0f} km")
print(f"   Fixed outliers: {outliers_count:,}")

print(f"\n✅ Cleaned dataset: {len(df):,} rows")

# ========================================
# 🔹 نمونه‌برداری
# ========================================

print("\n" + "="*70)
print("📊 Sampling Strategy...")
print("="*70)

if USE_SAMPLING and len(df) > SAMPLE_SIZE:
    print(f"\n⚡ Sampling {SAMPLE_SIZE:,} rows from {len(df):,} total rows")
    df_sample = df.sample(n=SAMPLE_SIZE, random_state=42)
    print(f"   Sample ratio: {SAMPLE_SIZE/len(df)*100:.1f}%")
    print(f"   Remaining data: {len(df) - SAMPLE_SIZE:,} rows not used")
elif not USE_SAMPLING:
    print(f"\n📦 Using ALL {len(df):,} rows (no sampling)")
    df_sample = df
else:
    print(f"\n⚠️ Dataset has only {len(df):,} rows (less than {SAMPLE_SIZE:,})")
    print(f"   Using all available data")
    df_sample = df

print(f"✅ Final data for training: {len(df_sample):,} rows")

# ========================================
# 🔹 انتخاب Feature ها
# ========================================

print("\n" + "="*70)
print("🎯 Feature Engineering...")
print("="*70)

features = [
    "Brand_Main",
    "Car_Age",
    "Mileage",
    "City",
    "Gearbox",
    "Fuel_Type",
    "Body_Condition",
    "Engine_Condition",
    "Chassis_Condition",
]

print(f"\nSelected features ({len(features)}):")
for i, feat in enumerate(features, 1):
    print(f"   {i}. {feat}")

# حذف ردیف‌های بدون داده اصلی
df_clean = df_sample.dropna(subset=['Brand_Main', 'Car_Age', 'Mileage'], how='all')
dropped = len(df_sample) - len(df_clean)

X = df_clean[features]
y = df_clean['Price']

print(f"\nData after feature selection:")
print(f"   Valid rows: {len(df_clean):,}")
print(f"   Dropped rows: {dropped:,}")
print(f"   Feature matrix shape: {X.shape}")

# ========================================
# 🔹 آمار داده
# ========================================

print("\n" + "="*70)
print("📊 Dataset Statistics:")
print("="*70)

print(f"\n💰 Price Distribution:")
print(f"   Min:       {y.min():>15,.0f} تومان ({y.min()/1_000_000:.1f}M)")
print(f"   Max:       {y.max():>15,.0f} تومان ({y.max()/1_000_000:.1f}M)")
print(f"   Mean:      {y.mean():>15,.0f} تومان ({y.mean()/1_000_000:.1f}M)")
print(f"   Median:    {y.median():>15,.0f} تومان ({y.median()/1_000_000:.1f}M)")
print(f"   Std Dev:   {y.std():>15,.0f} تومان ({y.std()/1_000_000:.1f}M)")

print(f"\n🚗 Car Age Distribution:")
print(f"   Min:       {df_clean['Car_Age'].min():>6.0f} years")
print(f"   Max:       {df_clean['Car_Age'].max():>6.0f} years")
print(f"   Mean:      {df_clean['Car_Age'].mean():>6.1f} years")
print(f"   Median:    {df_clean['Car_Age'].median():>6.0f} years")

print(f"\n📏 Mileage Distribution:")
print(f"   Min:       {df_clean['Mileage'].min():>10,.0f} km")
print(f"   Max:       {df_clean['Mileage'].max():>10,.0f} km")
print(f"   Mean:      {df_clean['Mileage'].mean():>10,.0f} km")
print(f"   Median:    {df_clean['Mileage'].median():>10,.0f} km")

# ========================================
# 🔹 Preprocessing Pipeline
# ========================================

print("\n" + "="*70)
print("⚙️ Building Preprocessing Pipeline...")
print("="*70)

numeric_features = ['Car_Age', 'Mileage']
categorical_features = [f for f in features if f not in numeric_features]

print(f"\n📊 Feature types:")
print(f"   Numeric ({len(numeric_features)}): {numeric_features}")
print(f"   Categorical ({len(categorical_features)}): {categorical_features}")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='نامشخص')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_features)
    ]
)

# ========================================
# 🔹 تنظیمات مدل
# ========================================

print("\n" + "="*70)
print("🧠 Model Configuration:")
print("="*70)

model = RandomForestRegressor(
    n_estimators=150,      # تعداد درخت‌ها
    max_depth=18,          # عمق هر درخت
    min_samples_split=8,   # حداقل نمونه برای تقسیم
    min_samples_leaf=4,    # حداقل نمونه در برگ
    max_features='sqrt',   # تعداد feature برای هر تقسیم
    random_state=42,
    n_jobs=-1,             # استفاده از همه هسته‌های CPU
    verbose=1              # نمایش پیشرفت
)

print(f"\nModel hyperparameters:")
print(f"   n_estimators:      {model.n_estimators}")
print(f"   max_depth:         {model.max_depth}")
print(f"   min_samples_split: {model.min_samples_split}")
print(f"   min_samples_leaf:  {model.min_samples_leaf}")
print(f"   max_features:      {model.max_features}")
print(f"   n_jobs:            {model.n_jobs} (all available cores)")

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# ========================================
# 🔹 تقسیم داده
# ========================================

print("\n" + "="*70)
print("🔀 Splitting Dataset...")
print("="*70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n📦 Data split:")
print(f"   Training set:   {X_train.shape[0]:>8,} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"   Test set:       {X_test.shape[0]:>8,} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"   Train/Test:     {X_train.shape[0]/X_test.shape[0]:.1f}:1")

# ========================================
# 🔹 آموزش مدل
# ========================================

print("\n" + "="*70)
print("🚀 TRAINING MODEL...")
print("="*70)
print(f"\n⏳ Expected time: 10-15 minutes for {len(X_train):,} samples")
print(f"💡 Progress bars will appear below...")
print(f"☕ Time for a coffee break!\n")

train_start = time.time()
pipeline.fit(X_train, y_train)
train_time = time.time() - train_start

print(f"\n✅ Training completed!")
print(f"⏱️ Training time: {train_time:.1f} seconds ({train_time/60:.2f} minutes)")
print(f"⚡ Speed: {len(X_train)/train_time:.0f} samples/second")

# ========================================
# 🔹 ارزیابی مدل
# ========================================

print("\n" + "="*70)
print("📊 Evaluating Model Performance...")
print("="*70)

print("\n🔮 Making predictions...")
eval_start = time.time()

train_preds = pipeline.predict(X_train)
test_preds = pipeline.predict(X_test)

eval_time = time.time() - eval_start
print(f"   Prediction time: {eval_time:.2f} seconds")

# محاسبه متریک‌ها
train_mae = mean_absolute_error(y_train, train_preds)
test_mae = mean_absolute_error(y_test, test_preds)

train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))

train_r2 = r2_score(y_train, train_preds)
test_r2 = r2_score(y_test, test_preds)

train_mape = np.mean(np.abs((y_train - train_preds) / y_train)) * 100
test_mape = np.mean(np.abs((y_test - test_preds) / y_test)) * 100

# ========================================
# 🔹 نمایش نتایج
# ========================================

print("\n" + "="*70)
print("📊 FINAL MODEL PERFORMANCE")
print("="*70)

print(f"\n🎯 Training Set:")
print(f"   MAE:     {train_mae:>15,.0f} تومان  ({train_mae/1_000_000:>6.2f} M)")
print(f"   RMSE:    {train_rmse:>15,.0f} تومان  ({train_rmse/1_000_000:>6.2f} M)")
print(f"   R²:      {train_r2:>15.4f}          ({train_r2*100:>6.2f}%)")
print(f"   MAPE:    {train_mape:>15.2f}%")

print(f"\n🎯 Test Set:")
print(f"   MAE:     {test_mae:>15,.0f} تومان  ({test_mae/1_000_000:>6.2f} M)")
print(f"   RMSE:    {test_rmse:>15,.0f} تومان  ({test_rmse/1_000_000:>6.2f} M)")
print(f"   R²:      {test_r2:>15.4f}          ({test_r2*100:>6.2f}%)")
print(f"   MAPE:    {test_mape:>15.2f}%")

# ارزیابی کیفیت
print(f"\n💡 Model Quality:")
if test_r2 > 0.90:
    quality = "🟢 EXCELLENT - Outstanding performance!"
elif test_r2 > 0.85:
    quality = "🟢 VERY GOOD - Great results!"
elif test_r2 > 0.80:
    quality = "🟡 GOOD - Solid performance"
elif test_r2 > 0.70:
    quality = "🟡 ACCEPTABLE - Usable but could improve"
else:
    quality = "🔴 NEEDS IMPROVEMENT - Consider more data/features"

print(f"   {quality}")
print(f"   Model explains {test_r2*100:.1f}% of price variance")
print(f"   Average prediction error: ±{test_mape:.1f}%")

# بررسی Overfitting
overfit_score = train_r2 - test_r2
print(f"\n📈 Overfitting Check:")
if overfit_score < 0.05:
    print(f"   ✅ Excellent - Low overfitting (Δ={overfit_score:.3f})")
elif overfit_score < 0.10:
    print(f"   ⚠️ Moderate overfitting (Δ={overfit_score:.3f})")
else:
    print(f"   ❌ High overfitting detected (Δ={overfit_score:.3f})")
    print(f"      Consider: reducing max_depth or n_estimators")

# ========================================
# 🔹 نمونه پیش‌بینی
# ========================================

print("\n" + "="*70)
print("🔮 Sample Predictions:")
print("="*70)

sample_size = min(15, len(y_test))
sample_results = pd.DataFrame({
    'Actual': y_test.head(sample_size).values,
    'Predicted': test_preds[:sample_size],
    'Error': np.abs(y_test.head(sample_size).values - test_preds[:sample_size]),
    'Error%': np.abs(y_test.head(sample_size).values - test_preds[:sample_size]) / y_test.head(sample_size).values * 100
})

# فرمت
sample_results['Actual_Fmt'] = sample_results['Actual'].apply(lambda x: f"{x:>12,.0f}")
sample_results['Predicted_Fmt'] = sample_results['Predicted'].apply(lambda x: f"{x:>12,.0f}")
sample_results['Error_Fmt'] = sample_results['Error'].apply(lambda x: f"{x:>12,.0f}")
sample_results['Error%_Fmt'] = sample_results['Error%'].apply(lambda x: f"{x:>6.1f}%")

print(f"\n{'Actual':>15s} {'Predicted':>15s} {'Error':>15s} {'Error%':>10s}")
print("-" * 60)
for _, row in sample_results.iterrows():
    print(f"{row['Actual_Fmt']} {row['Predicted_Fmt']} {row['Error_Fmt']} {row['Error%_Fmt']}")

# ========================================
# 🔹 ذخیره مدل
# ========================================

print("\n" + "="*70)
print("💾 Saving Model...")
print("="*70)

# ذخیره مدل
MODEL_PATH = os.path.join(BASE_DIR, 'car_price_model.pkl')
joblib.dump(pipeline, MODEL_PATH)
model_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
print(f"\n✅ Model saved: {MODEL_PATH}")
print(f"   File size: {model_size:.2f} MB")

# ذخیره اطلاعات
model_info = {
    'features': features,
    'numeric_features': numeric_features,
    'categorical_features': categorical_features,
    'sample_size': len(df_clean),
    'train_samples': len(X_train),
    'test_samples': len(X_test),
    'train_time_seconds': train_time,
    'train_mae': train_mae,
    'test_mae': test_mae,
    'train_rmse': train_rmse,
    'test_rmse': test_rmse,
    'train_r2': train_r2,
    'test_r2': test_r2,
    'train_mape': train_mape,
    'test_mape': test_mape,
    'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
    'config': {
        'n_estimators': model.n_estimators,
        'max_depth': model.max_depth,
        'sample_size_used': SAMPLE_SIZE
    }
}

INFO_PATH = os.path.join(BASE_DIR, 'model_features.pkl')
joblib.dump(model_info, INFO_PATH)
print(f"✅ Model info saved: {INFO_PATH}")

# ========================================
# 🔹 خلاصه نهایی
# ========================================

total_time = time.time() - start_time

print("\n" + "="*70)
print("✅ TRAINING COMPLETED SUCCESSFULLY!")
print("="*70)

print(f"\n📊 Summary:")
print(f"   Dataset size:        {len(df):,} rows")
print(f"   Samples used:        {len(df_clean):,} rows")
print(f"   Training samples:    {len(X_train):,}")
print(f"   Test samples:        {len(X_test):,}")
print(f"   Training time:       {train_time/60:.2f} minutes")
print(f"   Total execution:     {total_time/60:.2f} minutes")

print(f"\n🎯 Performance:")
print(f"   Test MAE:            {test_mae/1_000_000:.2f} million Toman")
print(f"   Test R²:             {test_r2:.4f} ({test_r2*100:.1f}%)")
print(f"   Test MAPE:           {test_mape:.2f}%")
print(f"   Model quality:       {quality}")

print(f"\n📦 Output files:")
print(f"   1. car_price_model.pkl ({model_size:.2f} MB)")
print(f"   2. model_features.pkl")

print(f"\n🚀 Next steps:")
print(f"   - Run 'test_model.py' to test predictions")
print(f"   - Use the model in your application")
print(f"   - Share the .pkl files for deployment")

print("\n" + "="*70)
print("✅ Ready to make predictions!")
print("="*70)