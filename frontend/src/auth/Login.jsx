"use client"

import { useState } from "react"
import { useNavigate, Link } from "react-router-dom"
import axios from "../api/axios"

export default function Login() {
  const navigate = useNavigate()
  const [formData, setFormData] = useState({ email: "", password: "" })
  const [error, setError] = useState("")
  const [loading, setLoading] = useState(false)

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value })
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError("")
    setLoading(true)

    try {
      // تغییر مهم: اضافه کردن اسلش (/) به انتهای آدرس
      const { data } = await axios.post("/api/auth/login/", formData)
      localStorage.setItem("token", data.token)
      navigate("/dashboard", { replace: true })
    } catch (err) {
      setError(err.response?.data?.message || "ایمیل یا رمز عبور اشتباه است")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 px-4">
      <div className="max-w-md w-full bg-white rounded-lg shadow-md p-8">
        <h2 className="text-3xl font-bold text-center mb-8">Login</h2>

        {error && <div className="mb-4 p-3 bg-red-50 text-red-700 rounded">{error}</div>}

        <form onSubmit={handleSubmit} className="space-y-6">
          <input
            type="email"
            name="email"
            placeholder="Email"
            value={formData.email}
            onChange={handleChange}
            required
            className="w-full input"
          />

          <input
            type="password"
            name="password"
            placeholder="Password"
            value={formData.password}
            onChange={handleChange}
            required
            className="w-full input"
          />

          <button disabled={loading} className="btn-primary w-full">
            {loading ? "Logging in..." : "Login"}
          </button>
        </form>

        <p className="mt-6 text-center text-sm">
          حساب نداری؟ <Link to="/signup" className="text-blue-600">ثبت‌نام</Link>
        </p>
      </div>
    </div>
  )
}