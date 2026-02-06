"use client"

import type React from "react"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"

export default function PredictPage() {
  const router = useRouter()
  const [brand, setBrand] = useState("")
  const [model, setModel] = useState("")
  const [year, setYear] = useState("")
  const [mileage, setMileage] = useState("")
  const [city, setCity] = useState("")
  const [predictedPrice, setPredictedPrice] = useState<number | null>(null)

  useEffect(() => {
    const storedUser = localStorage.getItem("user")
    if (!storedUser) {
      router.push("/login")
    }
  }, [router])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()

    // Mock prediction logic - generate a random price based on year and mileage
    const basePrice = 25000
    const yearFactor = (new Date().getFullYear() - Number.parseInt(year)) * 1200
    const mileageFactor = Number.parseInt(mileage) * 0.1
    const mockPrice = Math.max(5000, basePrice - yearFactor - mileageFactor)

    setPredictedPrice(Math.round(mockPrice))

    // Save prediction to localStorage
    const predictions = JSON.parse(localStorage.getItem("predictions") || "[]")
    const newPrediction = {
      id: Date.now(),
      brand,
      model,
      year: Number.parseInt(year),
      mileage: Number.parseInt(mileage),
      city,
      price: Math.round(mockPrice),
      date: new Date().toISOString(),
    }
    predictions.unshift(newPrediction)
    localStorage.setItem("predictions", JSON.stringify(predictions.slice(0, 10)))
  }

  const handleBackToDashboard = () => {
    router.push("/dashboard")
  }

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <h1 className="text-2xl font-bold text-foreground">Car Price Estimator</h1>
          <Button onClick={handleBackToDashboard} variant="outline" size="sm">
            Back to Dashboard
          </Button>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 max-w-2xl">
        <Card>
          <CardHeader>
            <CardTitle>New Price Prediction</CardTitle>
            <CardDescription>Enter the car details to estimate its market price</CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="brand">Brand</Label>
                  <Input
                    id="brand"
                    type="text"
                    placeholder="e.g., Toyota"
                    value={brand}
                    onChange={(e) => setBrand(e.target.value)}
                    required
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="model">Model</Label>
                  <Input
                    id="model"
                    type="text"
                    placeholder="e.g., Camry"
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                    required
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="year">Year</Label>
                  <Input
                    id="year"
                    type="number"
                    placeholder="e.g., 2020"
                    min="1990"
                    max={new Date().getFullYear()}
                    value={year}
                    onChange={(e) => setYear(e.target.value)}
                    required
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="mileage">Mileage (km)</Label>
                  <Input
                    id="mileage"
                    type="number"
                    placeholder="e.g., 50000"
                    min="0"
                    value={mileage}
                    onChange={(e) => setMileage(e.target.value)}
                    required
                  />
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="city">City</Label>
                <Input
                  id="city"
                  type="text"
                  placeholder="e.g., Los Angeles"
                  value={city}
                  onChange={(e) => setCity(e.target.value)}
                  required
                />
              </div>

              <Button type="submit" className="w-full">
                Estimate Price
              </Button>
            </form>

            {predictedPrice !== null && (
              <Alert className="mt-6 border-primary bg-primary/5">
                <AlertTitle className="text-lg font-bold">Predicted Price</AlertTitle>
                <AlertDescription className="text-3xl font-bold text-primary mt-2">
                  ${predictedPrice.toLocaleString()}
                </AlertDescription>
                <AlertDescription className="text-sm text-muted-foreground mt-2">
                  This is an estimated price based on the provided information.
                </AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>
      </main>
    </div>
  )
}
