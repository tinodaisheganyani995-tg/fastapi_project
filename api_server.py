from fastapi import FastAPI
import joblib

app = FastAPI()

model = joblib.load("model.pkl")

@app.get("/")
def home():
    return {"message": "Hello AI Engineer"}

@app.get("/square/{num}")
def square(num: int):
    return {"number": num, "square": num * num}

@app.get("/predict/{value}")
def predict(value: int):
    result = model.predict([[value]])

    return {
        "input": value,
        "prediction": float(result[0])
    }