# MedCheck

MedCheck is an AI-powered healthcare screening web app that helps users assess risk for several diseases using machine learning models and medical image analysis.

## Overview

This project demonstrates a full-stack healthcare prediction workflow: the frontend is built with HTML templates, the backend uses Flask, and ML models are loaded to detect patterns from tabular health metrics and medical images.

## Diseases covered

- Heart Disease
- Diabetes
- Breast Cancer
- Alzheimer
- Pneumonia
- Brain Tumour
- Covid-19

## Tech stack

- Python
- Flask
- NumPy
- scikit-learn
- TensorFlow / Keras
- Pillow
- HTML / CSS

## Project structure

- `app.py` — Flask application and prediction routes
- `constants.py` — disease identifiers and model configuration
- `static/models/` — trained model files
- `static/scalers/` — preprocessing scalers
- `templates/` — HTML templates for each disease page
- `uploads/` — uploaded medical images

## Local setup

1. Create and activate a virtual environment. (python --version 3.9.25)
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python app.py
```

4. Open the app in the browser at `http://localhost:5000`.

## Notes

- This project is designed for educational and portfolio use.
- For production deployment, use environment variables, secure file handling, and model validation checks.
- The app is best suited for local environment setup or a lightweight deployment workflow.

## Portfolio positioning

MedCheck shows practical experience across machine learning, healthcare AI, web development, and model deployment. It is a strong example of a full-stack AI project that connects real-world domain data with a user-facing interface.
