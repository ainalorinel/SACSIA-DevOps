from fastapi import FastAPI

app = FastAPI(
    title="SACSIA Deployment Check",
    description="Endpoint de vérification du déploiement",
    version="1.0.0",
)

@app.get("/")
async def root():
    return {
        "project": "SACSIA",
        "status": "online",
        "message": "L'API FastAPI fonctionne sur PythonAnywhere."
    }

@app.get("/health")
async def health():
    return {"status": "ok"}