export default function EstimationCard({ estimation }) {
  const { brand, model, year, mileage, estimatedPrice, createdAt } = estimation

  const formatPrice = (price) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(price)
  }

  const formatDate = (date) => {
    return new Date(date).toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
    })
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 hover:shadow-md transition">
      <div className="flex justify-between items-start mb-4">
        <div>
          <h3 className="text-xl font-bold text-gray-900">
            {brand} {model}
          </h3>
          <p className="text-sm text-gray-500 mt-1">{formatDate(createdAt)}</p>
        </div>
        <div className="text-right">
          <p className="text-2xl font-bold text-blue-600">{formatPrice(estimatedPrice)}</p>
          <p className="text-xs text-gray-500 mt-1">Estimated Price</p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 pt-4 border-t border-gray-100">
        <div>
          <p className="text-xs text-gray-500 mb-1">Year</p>
          <p className="text-sm font-medium text-gray-900">{year}</p>
        </div>
        <div>
          <p className="text-xs text-gray-500 mb-1">Mileage</p>
          <p className="text-sm font-medium text-gray-900">{mileage.toLocaleString()} km</p>
        </div>
      </div>
    </div>
  )
}
