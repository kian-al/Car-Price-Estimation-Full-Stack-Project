import { useEffect, useState } from "react"
import Navbar from "../components/Navbar"
import EstimationCard from "../components/EstimationCard"
import axios from "../api/axios"

export default function Dashboard() {
  const [estimations, setEstimations] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState("")

  useEffect(() => {
    const controller = new AbortController()

    const fetchEstimations = async () => {
      try {
        const { data } = await axios.get("/api/estimations", {
          signal: controller.signal,
        })
        setEstimations(data.slice(0, 10))
      } catch (err) {
        if (!controller.signal.aborted) {
          setError("خطا در دریافت داده‌ها")
        }
      } finally {
        setLoading(false)
      }
    }

    fetchEstimations()
    return () => controller.abort()
  }, [])

  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />

      <div className="max-w-7xl mx-auto px-6 py-8">
        <h1 className="text-3xl font-bold mb-6">Dashboard</h1>

        {loading && <p>در حال بارگذاری...</p>}
        {error && <p className="text-red-600">{error}</p>}

        {!loading && estimations.length === 0 && (
          <p className="text-gray-500">هیچ تخمینی ثبت نشده</p>
        )}

        <div className="grid md:grid-cols-3 gap-6">
          {estimations.map((item) => (
            <EstimationCard key={item.id} estimation={item} />
          ))}
        </div>
      </div>
    </div>
  )
}
