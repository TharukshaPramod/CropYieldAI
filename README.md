# README.md
# Crop Yield Prediction AI

## Setup
1. poetry install
2. poetry run python -m spacy download en_core_web_sm
3. Add .env with keys

## Run UI
poetry run streamlit run app.py

## Run API
poetry run uvicorn main_api:app --reload

## Test
poetry run pytest