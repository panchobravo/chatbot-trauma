# =======================================================================
# CHATBOT_BACKEND.PY - VERSIÓN HUMANIZADA & SENSIBILIDAD AJUSTADA
# =======================================================================

import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import string
import datetime
import streamlit as st
import gspread
import random  # <--- Necesario para la variabilidad humana

# -----------------------------------------------------------------------
# 1. CONFIGURACIÓN DE PERSONALIDAD
# -----------------------------------------------------------------------

# Frases para que no responda siempre igual (Efecto Humano)
FRASES_EMPATIA = [
    "Te cuento: ",
    "Esa es una buena pregunta. ",
    "Entiendo tu duda. Mira: ",
    "Claro, te explico: ",
    "Tranquilo/a, es normal tener esa duda. ",
    "Para tu tranquilidad: ",
    "Justo el Dr. me dejó indicaciones sobre esto. ",
    "" # A veces es mejor ir directo al grano (opción vacía)
]

# Respuestas rápidas a cosas sociales (Para pasar desapercibido)
CHARLA_SOCIAL = {
    "como esta el doctor": "¡El Dr. está excelente! Operando a full, pero siempre atento a sus pacientes. ¿Tú cómo vas?",
    "como esta el dr": "¡El Dr. está excelente! Operando a full, pero siempre atento a sus pacientes. ¿Tú cómo vas?",
    "gracias": "De nada, estamos para acompañarte en la recuperación. 💪",
    "muchas gracias": "¡Un gusto! Cualquier otra molestia, avísame.",
    "hola": "¡Hola! ¿Cómo te has sentido hoy?",
    "buenos dias": "¡Buen día! ¿Cómo amaneció esa zona operada?",
    "buenas tardes": "¡Buenas tardes! ¿En qué te puedo ayudar?"
}

PALABRAS_ALARMA = [
    "fiebre", "pus", "secreción", "infección", "sangrado abundante", 
    "hemorragia", "dolor insoportable", "desmayo", "no puedo respirar",
    "dedos azules", "no siento la pierna", "calor extremo"
]

MENSAJE_ALERTA = """
🚨 **ALERTA DE EMERGENCIA** 🚨
Lo que describes requiere atención inmediata.
Por favor, no esperes y **LLAME A URGENCIAS O VAYA A LA CLÍNICA AHORA**.
Este chat no puede resolver esa situación.
"""

# -----------------------------------------------------------------------
# 2. FUNCIONES TÉCNICAS (NLP)
# -----------------------------------------------------------------------
def preprocesar_texto(texto):
    texto = texto.lower()
    texto = ''.join([char for char in texto if char not in string.punctuation])
    try:
        stop_words_es = stopwords.words('spanish')
    except:
        stop_words_es = ["el", "la", "los", "las", "un", "una", "y", "o", "de", "a", "en"]
        
    palabras = texto.split()
    palabras_filtradas = [w for w in palabras if w not in stop_words_es]
    return ' '.join(palabras_filtradas)

def cargar_y_preparar_base(archivo_json):
    with open(archivo_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['intencion_preprocesada'] = df['intencion_clave'].apply(preprocesar_texto)
    return df

def inicializar_vectorizador(df):
    vectorizer = TfidfVectorizer()
    matriz_tfidf = vectorizer.fit_transform(df['intencion_preprocesada'])
    return vectorizer, matriz_tfidf

def registrar_pregunta_en_sheets(consulta):
    try:
        if "google_credentials" in st.secrets:
            creds_dict = dict(st.secrets["google_credentials"])
            gc = gspread.service_account_from_dict(creds_dict)
            sh = gc.open("Cerebro_Bot") # Asegúrate que este nombre coincida con tu Drive
            worksheet = sh.sheet1
            ahora = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            worksheet.append_row([ahora, consulta])
            # st.toast("Guardado para revisión", icon="📝") # Opcional: Quitar para que sea más silencioso
        else:
            print("⚠️ Sin credenciales")
    except Exception as e:
        print(f"Error Sheets: {e}")

# -----------------------------------------------------------------------
# 3. CEREBRO PRINCIPAL (LÓGICA MEJORADA)
# -----------------------------------------------------------------------
def buscar_respuesta_tfidf(consulta, df, vectorizer, matriz_tfidf, umbral=0.25): # <--- UMBRAL BAJADO A 0.25
    
    # 1. Filtro Social Rápido (Humanización)
    # Si el usuario dice algo social exacto, respondemos rápido sin buscar en el JSON médico
    consulta_clean = consulta.lower().strip()
    for frase, respuesta in CHARLA_SOCIAL.items():
        if frase in consulta_clean:
            return respuesta

    # 2. Búsqueda Médica (Si no es social)
    consulta_preprocesada = preprocesar_texto(consulta)
    
    if not consulta_preprocesada:
        return "¿Podrías darme más detalles? No te entendí bien."

    consulta_vector = vectorizer.transform([consulta_preprocesada])
    similitudes = cosine_similarity(consulta_vector, matriz_tfidf)
    mejor_sim_score = similitudes.max()
    mejor_sim_index = similitudes.argmax()
    
    # Lógica de Respuesta
    if mejor_sim_score > umbral:
        respuesta_medica = df.iloc[mejor_sim_index]['respuesta_validada']
        
        # FACTOR HUMANO: Agregamos una frase empática al azar al inicio
        preambulo = random.choice(FRASES_EMPATIA)
        return preambulo + respuesta_medica
    else:
        registrar_pregunta_en_sheets(consulta)
        # Respuesta de fallo más natural
        return "Mmm, esa duda es muy específica y prefiero no improvisar. Ya le dejé una nota al Dr. para que me explique la respuesta exacta. ¿Hay algo más en lo que te pueda ayudar mientras tanto?"

def revisar_guardrail_emergencia(consulta):
    consulta_lower = consulta.lower()
    for palabra in PALABRAS_ALARMA:
        if palabra in consulta_lower:
            return True 
    return False

def responder_consulta(consulta, df, vectorizer, matriz_tfidf):
    if revisar_guardrail_emergencia(consulta):
        return MENSAJE_ALERTA
    return buscar_respuesta_tfidf(consulta, df, vectorizer, matriz_tfidf)
