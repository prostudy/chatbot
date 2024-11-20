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
    #allow_origins=["https://pagos.agilpm.com"],  # Origen permitido (frontend)
    allow_origins=["*"],  # Origen permitido (frontend)
    allow_credentials=True,
    allow_methods=["*"],  # Métodos HTTP permitidos
    allow_headers=["*"],  # Encabezados permitidos
)


# Diccionario de preguntas y respuestas
faq = {
    "¿Qué tipos de aceites ofrece Quaker State?": {
        "respuesta": "Quaker State ofrece aceites sintéticos, semisintéticos y minerales para satisfacer diferentes necesidades de motores.",
        "sticker": "https://mexicodesconocido.com.mx/bardal.jpeg"
    },
    "¿Cuándo debo cambiar el aceite de mi vehículo?": {
        "respuesta": "Se recomienda cambiar el aceite cada 5,000 a 10,000 kilómetros dependiendo del tipo de aceite y las condiciones de manejo.",
        "sticker": "https://mexicodesconocido.com.mx/bardal.jpeg"
    },
    "¿Qué diferencia hay entre un aceite sintético y un mineral?": {
        "respuesta": "El aceite sintético ofrece mayor protección y rendimiento, especialmente en condiciones extremas, mientras que el mineral es más económico.",
        "sticker": "https://mexicodesconocido.com.mx/bardal.jpeg"
    },
    "¿Cómo elijo el mejor aceite para mi motor?": {
        "respuesta": "Consulta el manual del propietario de tu vehículo y selecciona un aceite con las especificaciones recomendadas.",
        "sticker": "https://mexicodesconocido.com.mx/bardal.jpeg"
    },
    "¿Qué significa la viscosidad del aceite?": {
        "respuesta": "La viscosidad indica qué tan espeso o fluido es el aceite. Por ejemplo, un 5W-30 es más fluido a bajas temperaturas.",
        "sticker": "https://mexicodesconocido.com.mx/bardal.jpeg"
    },
    "¿Es seguro usar aceite sintético en motores antiguos?": {
        "respuesta": "Sí, los aceites sintéticos modernos están diseñados para ser compatibles con la mayoría de los motores, incluso los antiguos.",
        "sticker": "https://mexicodesconocido.com.mx/bardal.jpeg"
    },
    "¿Cuáles son los beneficios del aceite sintético Quaker State?": {
        "respuesta": "El aceite sintético Quaker State proporciona mayor protección contra el desgaste, mejor limpieza del motor y mayor duración entre cambios.",
        "sticker": "https://mexicodesconocido.com.mx/bardal.jpeg"
    },
    "¿Puedo mezclar diferentes tipos de aceites?": {
        "respuesta": "No es recomendable, ya que podría afectar el rendimiento y la protección del motor.",
        "sticker": "https://mexicodesconocido.com.mx/bardal.jpeg"
    },
    "¿Qué es el grado API en el aceite de motor?": {
        "respuesta": "El grado API es una clasificación de calidad del aceite. Un aceite con API SN es adecuado para motores modernos.",
        "sticker": "https://mexicodesconocido.com.mx/bardal.jpeg"
    },
    "¿Por qué mi motor consume más aceite de lo normal?": {
        "respuesta": "El consumo excesivo de aceite puede deberse a fugas, desgaste del motor o uso de un aceite inadecuado.",
        "sticker": "https://mexicodesconocido.com.mx/bardal.jpeg"
    },
    "¿Qué debo hacer si mi aceite se ve oscuro?": {
        "respuesta": "El aceite oscuro no siempre es un problema; indica que está cumpliendo su función de limpiar el motor. Sin embargo, revísalo si está cerca del cambio programado.",
        "sticker": "https://mexicodesconocido.com.mx/bardal.jpeg"
    },
    "¿Cuáles son las consecuencias de no cambiar el aceite a tiempo?": {
        "respuesta": "El aceite sucio puede dañar el motor, aumentar el desgaste y reducir la eficiencia del combustible.",
        "sticker": "https://mexicodesconocido.com.mx/bardal.jpeg"
    },
    "¿Quaker State tiene aceites para motores diésel?": {
        "respuesta": "Sí, Quaker State ofrece aceites diseñados específicamente para motores diésel, proporcionando mayor protección y rendimiento.",
        "sticker": "https://mexicodesconocido.com.mx/bardal.jpeg"
    },
    "¿Cómo sé si mi motor necesita un aceite de alta viscosidad?": {
        "respuesta": "Un motor con desgaste elevado puede beneficiarse de un aceite de mayor viscosidad para reducir fugas y consumo.",
        "sticker": "https://mexicodesconocido.com.mx/bardal.jpeg"
    },
    "¿Qué pasa si uso un aceite con viscosidad incorrecta?": {
        "respuesta": "Usar un aceite con viscosidad inadecuada puede afectar la lubricación y el rendimiento del motor, especialmente en condiciones extremas.",
        "sticker": "https://mexicodesconocido.com.mx/bardal.jpeg"
    },
    "¿Qué es el aceite multigrado?": {
        "respuesta": "El aceite multigrado funciona bien en un rango amplio de temperaturas, como 5W-30, que fluye bien en frío y mantiene su espesor en calor.",
        "sticker": "https://mexicodesconocido.com.mx/bardal.jpeg"
    },
    "¿Qué tan ecológicos son los aceites Quaker State?": {
        "respuesta": "Quaker State trabaja continuamente para mejorar la sostenibilidad de sus productos, ofreciendo aceites de larga duración que reducen los desechos.",
        "sticker": "https://mexicodesconocido.com.mx/bardal.jpeg"
    },
    "¿Cómo se recicla el aceite usado?": {
        "respuesta": "El aceite usado debe llevarse a centros de reciclaje autorizados. Nunca lo viertas en el suelo o el drenaje.",
        "sticker": "https://mexicodesconocido.com.mx/bardal.jpeg"
    },
    "¿Quaker State ofrece aceites para climas extremos?": {
        "respuesta": "Sí, los aceites sintéticos de Quaker State están diseñados para proteger el motor en climas muy fríos o muy calurosos.",
        "sticker": "https://mexicodesconocido.com.mx/bardal.jpeg"
    },
    "¿Qué debo hacer si el nivel de aceite está bajo?": {
        "respuesta": "Añade aceite de inmediato con las mismas especificaciones que ya está en el motor y verifica si hay fugas.",
        "sticker": "https://mexicodesconocido.com.mx/bardal.jpeg"
    }
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
            {"role": "system", "content": "Eres un asistente basado en preguntas frecuentes. Enfoca a los usuarios a preguntas solo de la dinámica"},
            {"role": "user", "content": pregunta_usuario},
        ],
    )
    return {
        "response": response.choices[0].message["content"],
        "sticker": "https://mexicodesconocido.com.mx/bardal.jpeg"  # Sticker genérico
    }