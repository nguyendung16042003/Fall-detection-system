from fastapi import FastAPI

app = FastAPI(title="Fall Detection Cloud API", version="1.0.0")

@app.get("/")
async def root():
    return {"message": "Fall Detection System API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}