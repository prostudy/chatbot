from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "¡Hola desde FastAPI en Vercel!"}

@app.post("/chat")
async def chat(data: dict):
    message = data.get("message", "Hola")
    return {"response": f"Procesé tu mensaje: {message}"}
