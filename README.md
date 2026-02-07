# 🚗 Car Price Estimator

A modern Full-Stack platform for estimating used car prices using Machine Learning. This project features a robust Django backend and a responsive React frontend, fully containerized with Docker.

![Project Status](https://img.shields.io/badge/Status-Active-success)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## ✨ Features

* 🔮 **Smart Price Estimation:** Predicts car prices based on brand, year, mileage, and technical condition using a trained ML model.
* 🔐 **Secure Authentication:** Full sign-up and login system powered by JWT (JSON Web Tokens).
* 📊 **User Dashboard:** Users can view and manage their history of previous price estimations.
* 📱 **Responsive UI:** Modern and mobile-friendly interface built with React and Tailwind CSS.
* 🐳 **Dockerized:** Easy deployment and setup using Docker Compose.

## 🛠 Tech Stack

### Backend
* **Python & Django:** Core web framework.
* **Django REST Framework (DRF):** For building RESTful APIs.
* **Scikit-Learn:** Machine Learning library for price prediction model.
* **JWT (Simple JWT):** Secure authentication mechanism.
* **SQLite:** Default database (swappable with PostgreSQL).

### Frontend
* **React.js:** UI library.
* **Vite:** Next Generation Frontend Tooling.
* **Tailwind CSS:** Utility-first CSS framework.
* **Axios:** Promise based HTTP client.
* **Shadcn/UI:** Reusable components.

### DevOps
* **Docker & Docker Compose:** Containerization and orchestration.

---

## 🚀 Installation & Setup

You can run the project using **Docker** (Recommended) or **Manually**.

### Option 1: Using Docker (Recommended)

Simply run the following command in the root directory:

```bash
docker-compose up --build
Access the services at:Frontend: http://localhost:3000Backend: http://localhost:8000Option 2: Manual Setup1. Backend SetupBashcd backend
python -m venv venv
# Activate venv:
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate 

pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
2. Frontend SetupOpen a new terminal:Bashcd frontend
npm install
npm run dev
📡 API DocumentationBase URL: http://localhost:8000
👤 AuthenticationMethodEndpointDescriptionPOST/api/auth/register/Register a new userPOST/api/auth/login/Login and obtain JWT token
🚗 Car ServicesMethodEndpointDescriptionPOST/api/predict/Submit car details for price prediction (Auth required)GET/api/estimations/Get user's estimation history (Auth required)
📂 Project StructurePlaintextroot/
├── backend/            # Django Server Code
│   ├── accounts/       # User management
│   ├── prediction/     # ML Logic & Views
│   ├── estimations/    # History management
│   └── Dockerfile
├── frontend/           # React Client Code
│   ├── src/
│   │   ├── auth/       # Login/Signup pages
│   │   ├── pages/      # Dashboard & Predict pages
│   │   └── api/        # Axios setup
│   └── Dockerfile
└── docker-compose.yml  # Docker orchestration
🤝 ContributingContributions are welcome! Please feel free to submit a Pull Request.Developed with  by Kian Almasi
