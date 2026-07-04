<div align="center">

# 📈 Web Regression Prediction

### Machine Learning Regression Web Application using Flask & XGBoost

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-Web_App-black?style=for-the-badge&logo=flask)
![XGBoost](https://img.shields.io/badge/XGBoost-Regression-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

A modern web application for predicting regression values using a trained **XGBoost Regression Model** with an interactive Flask interface.

</div>

---

# ✨ Features

- 📊 Regression prediction using XGBoost
- 🎨 Modern and responsive UI
- 📝 Input form with default values
- 📂 Model loaded automatically
- ⚡ Fast prediction process
- 🔄 Multiple predictions without resetting inputs
- 📈 Prediction confidence (if available)
- 💻 Built with Flask

---

# 🖥️ Demo

Example Interface

| Home | Prediction |
|------|------------|
| Modern UI | Prediction Result |

> *(Add screenshots here after uploading them to GitHub)*

```
images/
├── home.png
└── result.png
```

---

# 📁 Project Structure

```text
web_5_regresion/
│
├── app.py
├── model.pkl
├── requirements.txt
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
├── templates/
│   ├── index.html
│   └── result.html
│
└── README.md
```

---

# 🚀 Installation

## 1. Clone Repository

```bash
git clone https://github.com/Zekken23/web_5_regresion.git
```

Move into the project directory

```bash
cd web_5_regresion
```

---

## 2. Create Virtual Environment

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Run Application

```bash
python app.py
```

Open your browser

```
http://127.0.0.1:5000
```

---

# 📊 Machine Learning Model

This project uses:

- **Algorithm:** XGBoost Regressor
- **Problem Type:** Regression
- **Framework:** Scikit-Learn + XGBoost

Prediction flow

```
User Input
      │
      ▼
Preprocessing
      │
      ▼
Trained XGBoost Model
      │
      ▼
Prediction Result
```

---

# 🎯 Example Prediction

| Feature | Value |
|----------|------:|
| Feature 1 | 25 |
| Feature 2 | 78 |
| Feature 3 | 4 |
| Feature 4 | 150 |

Prediction

```
Predicted Value : 84.27
```

---

# 🛠️ Tech Stack

- Python
- Flask
- HTML5
- CSS3
- JavaScript
- XGBoost
- Pandas
- NumPy
- Scikit-Learn

---

# 📦 Requirements

Example

```
Flask
xgboost
numpy
pandas
scikit-learn
joblib
```

or

```bash
pip install -r requirements.txt
```

---

# 📸 Screenshots

## Home Page

<img src="images/home.png" width="900">

## Prediction Result

<img src="images/result.png" width="900">

---

#  Future Improvements

- User Authentication
- Prediction History
- Data Visualization
- Model Performance Dashboard
- REST API
- Dark Mode
- Docker Deployment

---

# Author

**Muhammad Yusron AL Ghoni Rizqullah**
**Dino Alfian Zamri**
**M Deanova Whisal**

GitHub

https://github.com/Zekken23

---

# Support

If you find this project useful, consider giving it a ⭐ on GitHub!

```
⭐ Star this repository
🍴 Fork it
🐛 Report issues
```

---

<div align="center">

Made with ❤️ using Flask & XGBoost

</div>
