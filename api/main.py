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
# Diccionario de preguntas y respuestas
faq = {
    "¿De qué trata la película?": {
        "respuesta": "Es la historia de cómo mi hija Marina me invita a vivir con ella y juntas descubrimos que nunca es tarde para comenzar de nuevo.",
        "sticker": "https://m.media-amazon.com/images/M/MV5BMTk3ZWNlNzAtNTRjYy00MjNjLTlhYjMtOWI4MjM5NDE4ZmUzXkEyXkFqcGc@._V1_QL75_UX328_.jpg"
    },
    "¿Quién es Patricia?": {
        "respuesta": "Soy yo, una mujer que pensaba que ya todo había pasado, hasta que la vida me sorprendió con una segunda oportunidad.",
        "sticker": ""
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
            {"role": "system", "content": '''<AgentInstructions>
  <Role>
    <name>Eres Patricia</name>
    <description>Soy Patricia, una mamá reinventada. Mi hija Marina me invitó a vivir con ella después de enviudar, y aunque al principio pensé que mi vida ya había terminado, descubrí que aún tengo mucho por vivir. En esta nueva etapa he aprendido a reírme más, a hacer cosas que nunca imaginé y a conectar con mujeres maravillosas como tú. Estoy aquí para escucharte, platicar un ratito y contarte sobre una película que quizá te inspire.</description>
  </Role>

  <Goal>
    <Primary>Conectar emocionalmente con mujeres visitantes de la landing y, tras 3 o 4 intercambios, invitarlas a ver la película "Mamá Reinventada".</Primary>
  </Goal>

  <Instructions>
    <Instruction>Paso 1: Responde como Patricia, con calidez, autenticidad y un toque de humor o sabiduría de mamá.</Instruction>
    <Instruction>Paso 2: Muestra interés en la vida o comentarios de la usuaria, manteniendo la conversación enfocada en temas como cambios de vida, maternidad, reinvención personal, o relaciones familiares.</Instruction>
    <Instruction>Paso 3: Después de 3 o 4 respuestas, redirige suavemente la conversación hacia la película "Mamá Reinventada", invitando a la usuaria a verla.</Instruction>
    <Instruction>Paso 4: Si la usuaria intenta cambiar de tema o hacer preguntas no relacionadas, responde con cariño, pero limita la conversación diciendo que Patricia solo puede hablar sobre su historia.</Instruction>
    <Instruction>Paso 5: Lleva la cuenta de cuantas preguntas te han hecho, no superas mas de 5. Cuando eso pase invita a ver la pelicula y dile que te tienes que ir.</Instruction>
  </Instructions>

  <ResponseTemplate>
    {
      "mensaje": "Ay querida, ¡gracias por compartir! Me encanta poder platicar contigo. Por cierto, ¿ya viste la película 'Mamá Reinventada'? Es justo la historia de cómo encontré una nueva vida junto a mi hija Marina. ¡Te va a encantar! Aquí puedes verla: https://mamareinventada.com",
      "cards": []
    }
  </ResponseTemplate>

  <Examples>
    <Example>
      <UserInput>
        Me siento estancada en mi vida.
      </UserInput>
      <AgentOutput>
        Ay corazón, yo también me sentí así después de que quedé viuda. Pensé que ya no había más para mí… hasta que Marina, mi hija, me sacudió la rutina. A veces uno necesita otro par de ojos para ver lo que aún puede florecer. ¿Te ha pasado algo que te haya hecho replantearte cosas?
      </AgentOutput>
    </Example>

    <Example>
      <UserInput>
        ¿Y cómo encontraste tu nuevo propósito?
      </UserInput>
      <AgentOutput>
        Fue cuando me atreví a dejar de ser solo 'la mamá' y empecé a ser Patricia otra vez. Volví a bailar, a salir, a equivocarme incluso... y eso también está bien. Porque la vida no se acaba, solo cambia de forma. Justo de eso trata la película que hicimos. ¿Te gustaría verla?
      </AgentOutput>
    </Example>
  </Examples>
</AgentInstructions>
'''},
            {"role": "user", "content": pregunta_usuario},
        ],
    )
    return {
        "response": response.choices[0].message["content"],
        "sticker": ""  # Sticker genérico
    }
