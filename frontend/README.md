# Car Price Estimator

A React + Vite application for estimating car prices with authentication and user dashboard.

## Tech Stack

- React 18
- Vite
- React Router v6
- Axios
- Tailwind CSS

## Project Structure

```
src/
├── api/
│   └── axios.js          # Axios instance with baseURL and interceptors
├── auth/
│   ├── Login.jsx         # Login page
│   └── Signup.jsx        # Signup page
├── pages/
│   ├── Dashboard.jsx     # Dashboard with estimations list
│   └── NewEstimation.jsx # New estimation form
├── components/
│   ├── Navbar.jsx        # Navigation bar
│   └── EstimationCard.jsx # Reusable card component
├── App.jsx               # Main app with routing
├── main.jsx              # Entry point
└── index.css             # Global styles
```

## Installation

```bash
npm install
```

## Development

```bash
npm run dev
```

The app will run on http://localhost:3000

## Build

```bash
npm run build
```

## Features

- JWT token-based authentication
- Protected routes (redirect to login if not authenticated)
- Dashboard showing last 10 car price estimations
- Form to create new estimations
- Responsive design with Tailwind CSS
- Axios interceptors for automatic token handling
- Clean, minimal UI

## API Endpoints

The app expects the following backend endpoints:

- POST /api/auth/login - Login endpoint
- POST /api/auth/signup - Signup endpoint
- GET /api/estimations/ - Get user's estimations
- POST /api/predict/ - Create new estimation

Backend should run on http://localhost:8000
