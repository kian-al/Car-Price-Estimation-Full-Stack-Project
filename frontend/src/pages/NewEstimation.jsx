"use client"

import { useState } from "react"
import { useNavigate } from "react-router-dom"
import Navbar from "../components/Navbar"
import axios from "../api/axios"

export default function NewEstimation() {
  const navigate = useNavigate()
  
  // ۱. اضافه کردن فیلدهای جدید به State
  const [formData, setFormData] = useState({
    brand: "",
    model_year: "", // تغییر نام برای هماهنگی ذهنی (اختیاری)
    mileage: "",
    gearbox: "دنده ای", // مقدار پیش‌فرض
    fuel_type: "بنزینی", // مقدار پیش‌فرض
    body_condition: "سالم", // مقدار پیش‌فرض
    engine_condition: "سالم", // مقدار پیش‌فرض
    chassis_condition: "سالم و پلمپ", // مقدار پیش‌فرض
    city: "",
  })
  
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")

  // لیست‌های داده (می‌توانید کامل‌تر کنید)
  const brands = ["پراید", "پژو 206", "سمند", "تویوتا", "هیوندای"]
  const cities = ["tehran", "mashhad", "isfahan", "shiraz"]
  const gearboxes = ["دنده ای", "اتوماتیک"]
  const fuelTypes = ["بنزینی", "دوگانه سوز", "هیبریدی"]
  const conditions = ["سالم", "تصادفی", "نیاز به تعمیر", "رنگ دار"]

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    })
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError("")
    setLoading(true)

    // ۲. ساخت آبجکت نهایی دقیقاً مشابه چیزی که Postman تست کردید
    const payload = {
        "Brand": formData.brand,
        "Model_Year": Number.parseInt(formData.model_year),
        "Mileage": Number.parseInt(formData.mileage),
        "Gearbox": formData.gearbox,
        "Fuel_Type": formData.fuel_type,
        "Body_Condition": formData.body_condition,
        "Engine_Condition": formData.engine_condition,
        "Chassis_Condition": formData.chassis_condition,
        "City": formData.city
    }

    try {
      // ارسال درخواست به بک‌اند
      const response = await axios.post("/api/predict/", payload)
      
      // اگر موفقیت آمیز بود، کاربر را هدایت کن یا قیمت را نمایش بده
      console.log("Prediction Result:", response.data)
      alert(`قیمت تخمینی: ${response.data.predicted_price}`) // نمایش موقت
      navigate("/dashboard")
      
    } catch (err) {
      console.error(err)
      setError(err.response?.data?.message || "خطا در ارتباط با سرور. لطفا ورودی‌ها را چک کنید.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />

      <div className="max-w-2xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">تخمین قیمت خودرو</h1>
        </div>

        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8">
          {error && (
            <div className="mb-6 p-3 bg-red-50 border border-red-200 text-red-700 rounded-md text-sm">{error}</div>
          )}

          <form onSubmit={handleSubmit} className="space-y-6">
            
            {/* Brand */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">برند</label>
              <select name="brand" value={formData.brand} onChange={handleChange} required className="w-full px-4 py-2 border rounded-md">
                <option value="">انتخاب کنید</option>
                {brands.map(b => <option key={b} value={b}>{b}</option>)}
              </select>
            </div>

            {/* Model Year */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">سال ساخت</label>
              <input type="number" name="model_year" value={formData.model_year} onChange={handleChange} required className="w-full px-4 py-2 border rounded-md" placeholder="مثلا 1396" />
            </div>

            {/* Mileage */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">کارکرد (کیلومتر)</label>
              <input type="number" name="mileage" value={formData.mileage} onChange={handleChange} required className="w-full px-4 py-2 border rounded-md" />
            </div>

            {/* Gearbox */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">گیربکس</label>
              <select name="gearbox" value={formData.gearbox} onChange={handleChange} className="w-full px-4 py-2 border rounded-md">
                {gearboxes.map(g => <option key={g} value={g}>{g}</option>)}
              </select>
            </div>

             {/* Fuel Type */}
             <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">نوع سوخت</label>
              <select name="fuel_type" value={formData.fuel_type} onChange={handleChange} className="w-full px-4 py-2 border rounded-md">
                {fuelTypes.map(f => <option key={f} value={f}>{f}</option>)}
              </select>
            </div>

            {/* City */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">شهر</label>
              <select name="city" value={formData.city} onChange={handleChange} required className="w-full px-4 py-2 border rounded-md">
                <option value="">انتخاب کنید</option>
                {cities.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>

            {/* دکمه ارسال */}
            <div className="flex gap-4 pt-4">
              <button type="submit" disabled={loading} className="flex-1 bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 transition">
                {loading ? "در حال محاسبه..." : "تخمین قیمت"}
              </button>
            </div>

          </form>
        </div>
      </div>
    </div>
  )
}