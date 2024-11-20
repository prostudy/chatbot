from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import openai
import os
from difflib import get_close_matches
import numpy as np
from dotenv import load_dotenv
import os

app = FastAPI()


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


# Configuración del middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://pagos.agilpm.com"],  # Origen permitido (frontend)
    allow_credentials=True,
    allow_methods=["*"],  # Métodos HTTP permitidos
    allow_headers=["*"],  # Encabezados permitidos
)


# Diccionario de preguntas y respuestas
faq = {
    "¿Cómo puedo proteger mi información personal en Internet?": {
        "respuesta": "Para proteger tu información en línea, utiliza contraseñas seguras y únicas para cada cuenta...",
        "sticker": "https://mexicodesconocido.com.mx/bardal.jpeg"
    },
    "¿Qué es la inteligencia artificial y cómo afecta mi vida diaria?": {
        "respuesta": "La inteligencia artificial es una rama de la informática que permite a las máquinas aprender...",
        "sticker": "https://mexicodesconocido.com.mx/bardal.jpeg"
    },
    # Agrega más preguntas con respuestas y stickers asociados
}

## Función para generar embeddings de preguntas
def obtener_embedding(texto):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=texto
    )
    return np.array(response["data"][0]["embedding"])

# Precalcular embeddings para el FAQ
faq_embeddings = {pregunta: obtener_embedding(pregunta) for pregunta in faq}

# Buscar la pregunta más similar
def encontrar_pregunta_mas_similar(pregunta_usuario):
    embedding_usuario = obtener_embedding(pregunta_usuario)
    similitudes = {
        pregunta: np.dot(embedding_usuario, embedding) / (
            np.linalg.norm(embedding_usuario) * np.linalg.norm(embedding)
        )
        for pregunta, embedding in faq_embeddings.items()
    }
    pregunta_mas_similar = max(similitudes, key=similitudes.get)
    mayor_similitud = similitudes[pregunta_mas_similar]
    if mayor_similitud > 0.85:
        return pregunta_mas_similar
    return None

# Endpoint principal del chatbot
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    pregunta_usuario = data.get("message", "")

    # Buscar respuesta en el FAQ
    pregunta_mas_similar = encontrar_pregunta_mas_similar(pregunta_usuario)
    if pregunta_mas_similar:
        respuesta = faq[pregunta_mas_similar]
        return {"response": respuesta["respuesta"], "sticker": respuesta["sticker"]}

    # Si no hay coincidencias, usar GPT
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Eres un asistente basado en preguntas frecuentes. Solo responde eso y nada mas"},
            {"role": "user", "content": pregunta_usuario},
        ],
    )
    return {
        "response": response.choices[0].message["content"],
        "sticker": "https://mexicodesconocido.com.mx/bardal.jpeg"  # Sticker genérico
    }