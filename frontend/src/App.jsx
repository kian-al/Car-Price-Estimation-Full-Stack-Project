import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom"
import Login from "./auth/Login"
import Signup from "./auth/Signup"
import Dashboard from "./pages/Dashboard"
import NewEstimation from "./pages/NewEstimation"

// Protected Route Component
function ProtectedRoute({ children }) {
  const token = localStorage.getItem("token")

  if (!token) {
    return <Navigate to="/login" replace />
  }

  return children
}

// Public Route Component (redirect to dashboard if already logged in)
function PublicRoute({ children }) {
  const token = localStorage.getItem("token")

  if (token) {
    return <Navigate to="/dashboard" replace />
  }

  return children
}

function App() {
  return (
    <BrowserRouter>
      <Routes>
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
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
