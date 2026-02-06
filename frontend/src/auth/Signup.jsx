"use client"

import { useState } from "react"
import { useNavigate, Link } from "react-router-dom"
import axios from "../api/axios"

export default function Signup() {
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
      await axios.post("/api/auth/register", formData)
      navigate("/login")
    } catch (err) {
      setError(err.response?.data?.message || "ثبت‌نام ناموفق بود")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 px-4">
      <div className="max-w-md w-full bg-white rounded-lg shadow-md p-8">
        <h2 className="text-3xl font-bold text-center mb-8">Sign Up</h2>

        {error && <div className="mb-4 p-3 bg-red-50 text-red-700 rounded">{error}</div>}

        <form onSubmit={handleSubmit} className="space-y-6">
          <input
            type="email"
            name="email"
            placeholder="Email"
            onChange={handleChange}
            required
            className="w-full input"
          />

          <input
            type="password"
            name="password"
            placeholder="Password"
            onChange={handleChange}
            required
            className="w-full input"
          />

          <button disabled={loading} className="btn-primary w-full">
            {loading ? "Signing up..." : "Sign Up"}
          </button>
        </form>

        <p className="mt-6 text-center text-sm">
          حساب داری؟ <Link to="/login" className="text-blue-600">ورود</Link>
        </p>
      </div>
    </div>
  )
}
