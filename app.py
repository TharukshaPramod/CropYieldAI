import streamlit as st
import requests
import jwt

st.title("Crop Yield Prediction AI")
query = st.text_input("Enter query (e.g., Predict wheat yield in California)")
if st.button("Predict"):
    token = jwt.encode({"user": "test"}, "supersecretkey", algorithm="HS256")
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post("http://localhost:8000/predict", json={"query": query}, headers=headers)
    if response.status_code == 200:
        st.write(response.json()["result"])
    else:
        st.error("Error: Check API")