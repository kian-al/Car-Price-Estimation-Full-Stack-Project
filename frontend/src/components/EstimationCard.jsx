export default function EstimationCard({ estimation }) {
  // اصلاح تطبیق نام فیلدها با خروجی بک‌اند جانگو
  const brand = estimation.brand || estimation.Brand;
  // نکته: در مدل شما نام مدل ماشین جداگانه ذخیره نشده، احتمالا منظور همان برند است یا باید اضافه شود
  // فعلا مدل را خالی یا همان برند نشان میدهیم
  const model = ""; 
  const year = estimation.model_year || estimation.Model_Year;
  const mileage = estimation.mileage || estimation.Mileage;
  const estimatedPrice = estimation.predicted_price;
  const createdAt = estimation.created_at;

  const formatPrice = (price) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "IRR", // تغییر به ریال چون قیمت‌ها به تومان/ریال هستند
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(price)
  }

  const formatDate = (date) => {
    if (!date) return "";
    return new Date(date).toLocaleDateString("fa-IR", { // تغییر به تاریخ شمسی (اختیاری)
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
          <p className="text-xs text-gray-500 mt-1">قیمت تخمینی</p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 pt-4 border-t border-gray-100">
        <div>
          <p className="text-xs text-gray-500 mb-1">سال ساخت</p>
          <p className="text-sm font-medium text-gray-900">{year}</p>
        </div>
        <div>
          <p className="text-xs text-gray-500 mb-1">کارکرد</p>
          <p className="text-sm font-medium text-gray-900">{Number(mileage).toLocaleString()} km</p>
        </div>
      </div>
    </div>
  )
}