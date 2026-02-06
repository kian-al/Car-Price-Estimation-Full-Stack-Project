import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom"
import Login from "./auth/Login"
import Signup from "./auth/Signup"
import Dashboard from "./pages/Dashboard"
import NewEstimation from "./pages/NewEstimation"

// کامپوننت محافظت شده (فقط برای کاربران لاگین شده)
function ProtectedRoute({ children }) {
  const token = localStorage.getItem("token")
  // اگر توکن نبود، برو به لاگین
  if (!token) {
    return <Navigate to="/login" replace />
  }
  return children
}

// کامپوننت عمومی (اگر کاربر لاگین بود، نباید این صفحات را ببیند)
function PublicRoute({ children }) {
  const token = localStorage.getItem("token")
  // اگر توکن بود، برو به داشبورد
  if (token) {
    return <Navigate to="/dashboard" replace />
  }
  return children
}

function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* مسیرهای عمومی (لاگین و ثبت نام) */}
        <Route
          path="/login"
          element={
            <PublicRoute>
              <Login />
            </PublicRoute>
          }
        />
        <Route
          path="/signup"
          element={
            <PublicRoute>
              <Signup />
            </PublicRoute>
          }
        />

        {/* مسیرهای محافظت شده (داشبورد و تخمین جدید) */}
        <Route
          path="/dashboard"
          element={
            <ProtectedRoute>
              <Dashboard />
            </ProtectedRoute>
          }
        />
        <Route
          path="/new-estimation"
          element={
            <ProtectedRoute>
              <NewEstimation />
            </ProtectedRoute>
          }
        />

        {/* مسیر پیش‌فرض: هدایت به داشبورد (که خودش چک میکنه لاگین هست یا نه) */}
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        
        {/* مدیریت آدرس‌های اشتباه */}
        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App