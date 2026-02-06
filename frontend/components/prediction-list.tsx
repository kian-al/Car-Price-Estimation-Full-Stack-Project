"use client"

import { useEffect, useState } from "react"

interface Prediction {
  id: number
  brand: string
  model: string
  year: number
  mileage: number
  city: string
  price: number
  date: string
}

// Mock data for initial display
const mockPredictions: Prediction[] = [
  {
    id: 1,
    brand: "Toyota",
    model: "Camry",
    year: 2020,
    mileage: 45000,
    city: "Los Angeles",
    price: 22500,
    date: new Date(Date.now() - 86400000 * 2).toISOString(),
  },
  {
    id: 2,
    brand: "Honda",
    model: "Civic",
    year: 2019,
    mileage: 60000,
    city: "San Francisco",
    price: 18900,
    date: new Date(Date.now() - 86400000 * 5).toISOString(),
  },
  {
    id: 3,
    brand: "Ford",
    model: "F-150",
    year: 2021,
    mileage: 30000,
    city: "Austin",
    price: 35800,
    date: new Date(Date.now() - 86400000 * 7).toISOString(),
  },
  {
    id: 4,
    brand: "Tesla",
    model: "Model 3",
    year: 2022,
    mileage: 15000,
    city: "Seattle",
    price: 42000,
    date: new Date(Date.now() - 86400000 * 10).toISOString(),
  },
  {
    id: 5,
    brand: "BMW",
    model: "3 Series",
    year: 2018,
    mileage: 75000,
    city: "Miami",
    price: 25600,
    date: new Date(Date.now() - 86400000 * 12).toISOString(),
  },
]

export function PredictionList() {
  const [predictions, setPredictions] = useState<Prediction[]>([])

  useEffect(() => {
    const storedPredictions = localStorage.getItem("predictions")
    if (storedPredictions) {
      const parsed = JSON.parse(storedPredictions)
      setPredictions(parsed.length > 0 ? parsed : mockPredictions)
    } else {
      setPredictions(mockPredictions)
    }
  }, [])

  if (predictions.length === 0) {
    return (
      <div className="text-center py-8 text-muted-foreground">
        No predictions yet. Create your first prediction to get started!
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {predictions.map((prediction) => (
        <div key={prediction.id} className="border border-border rounded-lg p-4 hover:bg-accent/50 transition-colors">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
            <div className="space-y-1">
              <h3 className="font-semibold text-foreground">
                {prediction.brand} {prediction.model}
              </h3>
              <div className="text-sm text-muted-foreground space-y-0.5">
                <div>
                  Year: {prediction.year} • Mileage: {prediction.mileage.toLocaleString()} km
                </div>
                <div>City: {prediction.city}</div>
              </div>
            </div>
            <div className="text-left sm:text-right">
              <div className="text-2xl font-bold text-primary">${prediction.price.toLocaleString()}</div>
              <div className="text-xs text-muted-foreground mt-1">{new Date(prediction.date).toLocaleDateString()}</div>
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}
