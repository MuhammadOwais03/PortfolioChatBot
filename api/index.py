from fastapi import FastAPI
from mangum import Mangum

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hello from FastAPI"}

handler = Mangum(app)  # Only used on Vercel
