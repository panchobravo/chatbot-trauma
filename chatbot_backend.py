# =======================================================================
# CHATBOT_BACKEND.PY - ALGORITMO DE BÚSQUEDA VECTORIAL MEJORADO (V4.0)
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
import random

# -----------------------------------------------------------------------
# 1. CONSTANTES Y CONFIGURACIÓN
# -----------------------------------------------------------------------

PALABRAS_ALARMA = [
    "fiebre", "pus", "secreción", "infección", "sangrado abundante", 
    "hemorragia", "dolor insoportable", "desmayo", "no puedo respirar",
    "dedos azules", "no siento la pierna", "calor extremo",
    "se abrió", "abierta", "herida abierta", "veo la placa", "veo el hueso",
    "hueso expuesto", "tornillo", "supurando", "mal olor", "negro", "necrosis"
]

MENSAJE_ALERTA = """
🚨 **ALERTA DE EMERGENCIA** 🚨
Lo que describes NO es normal y requiere evaluación médica presencial inmediata.
Si la herida se abrió, ves material (placas/hueso) o hay infección, **NO toques nada**.
**Dirígete a Urgencias ahora mismo.**
"""

CHARLA_SOCIAL = {
    "como esta el doctor": "¡El Dr. está excelente! Operando, pero atento a ustedes.",
    "gracias": "No hay de qué. Vamos paso a paso con esa recuperación. 💪",
    "hola": "¡Hola! ¿Cómo va esa recuperación hoy?",
    "chao": "¡Descansa! Recuerda mantener la pierna en alto.",
    "adios": "¡Que tengas buen descanso!",
    "eres un robot": "Soy un asistente virtual entrenado por el Dr., pero mi objetivo es ayudarte de verdad.",
    "eres humano": "Soy una IA asistente del equipo médico. Estoy aquí para que no te sientas solo/a con tus dudas."
}

RESPUESTAS_EMOCIONALES = {
    "mal": "Siento escuchar eso. La recuperación tiene días difíciles. ¿Tienes mucho dolor o es algo más?",
    "mas o menos": "Entiendo, hay días mejores y peores. ¿Qué es lo que más te molesta hoy?",
    "asustado": "Es normal sentir miedo después de una cirugía. Estoy aquí para orientarte. ¿Qué síntomas te preocupan?",
    "triste": "El ánimo afecta la recuperación. ¡Ánimo! Esto es temporal. ¿Te duele algo físicamente?",
    "bien": "¡Qué buena noticia! Me alegra mucho. ¿Tienes alguna duda puntual hoy?",
    "mejor": "¡Excelente! Vamos progresando. Sigue cuidándote igual."
}

FRASES_EMPATIA = [
    "Comprendo tu inquietud. ",
    "Es una duda muy frecuente. ",
    "Para tu tranquilidad: ",
    "Te explico lo que indica el Dr.: ",
    "Mira, lo importante es esto: ",
    "" 
]

# -----------------------------------------------------------------------
# 2. PROCESAMIENTO NLP (CORE)
# -----------------------------------------------------------------------

def preprocesar_texto(texto):
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = ''.join([char for char in texto if char not in string.punctuation])
    try:
        stop_words_es = stopwords.words('spanish')
    except:
        # Fallback si NLTK no descarga las stopwords
        stop_words_es = ["el", "la", "los", "las", "un", "una", "y", "o", "de", "a", "en", "que", "me", "mi", "mis"]
    
    palabras = texto.split()
    palabras_filtradas = [w for w in palabras if w not in stop_words_es]
    return ' '.join(palabras_filtradas)

def cargar_y_preparar_base(archivo_json):
    """
    Carga el JSON y crea un campo 'texto_busqueda' que concatena
    la intención clave + las palabras clave. Esto corrige el error
    de que el bot no encuentre términos que solo están en los tags.
    """
    with open(archivo_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    
    # FUSIONAMOS INTENCIÓN + PALABRAS CLAVE (SOLUCIÓN AL BUG)
    df['texto_busqueda'] = df.apply(
        lambda row: row['intencion_clave'] + " " + " ".join(row['palabras_clave']), 
        axis=1
    )
    
    # Vectorizamos sobre el campo fusionado
    df['intencion_preprocesada'] = df['texto_busqueda'].apply(preprocesar_texto)
    return df

def inicializar_vectorizador(df):
    vectorizer = TfidfVectorizer()
    matriz_tfidf = vectorizer.fit_transform(df['intencion_preprocesada'])
    return vectorizer, matriz_tfidf

# -----------------------------------------------------------------------
# 3. CONEXIÓN A GOOGLE SHEETS
# -----------------------------------------------------------------------

def registrar_pregunta_en_sheets(consulta):
    try:
        if "google_credentials" in st.secrets:
            creds_dict = dict(st.secrets["google_credentials"])
            gc = gspread.service_account_from_dict(creds_dict)
            sh = gc.open("Cerebro_Bot") 
            worksheet = sh.sheet1
            ahora = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            worksheet.append_row([ahora, consulta])
    except Exception as e:
        print(f"Error Sheets: {e}")

def guardar_paciente_en_sheets(nombre, apellidos, rut, telefono, email):
    try:
        if "google_credentials" in st.secrets:
            creds_dict = dict(st.secrets["google_credentials"])
            gc = gspread.service_account_from_dict(creds_dict)
            sh = gc.open("Cerebro_Bot")
            try:
                worksheet = sh.worksheet("Usuarios")
            except:
                worksheet = sh.sheet1 
            ahora = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            worksheet.append_row([ahora, nombre, apellidos, rut, telefono, email])
            return True
    except Exception as e:
        st.error(f"Error guardando paciente: {e}")
        return False

# -----------------------------------------------------------------------
# 4. LÓGICA DE RESPUESTA
# -----------------------------------------------------------------------

def buscar_respuesta_tfidf(consulta, df, vectorizer, matriz_tfidf, umbral=0.20): # Umbral optimizado
    
    consulta_clean = consulta.lower().strip()
    palabras_usuario = consulta_clean.split()

    # 1. FILTRO SOCIAL (Solo si la frase es corta < 5 palabras)
    if len(palabras_usuario) < 5: 
        for frase, respuesta in CHARLA_SOCIAL.items():
            if frase in consulta_clean:
                return respuesta

    # 2. FILTRO EMOCIONAL
    for emocion, respuesta in RESPUESTAS_EMOCIONALES.items():
        if emocion in consulta_clean:
            return respuesta

    # 3. BÚSQUEDA MÉDICA (VECTORIAL)
    consulta_preprocesada = preprocesar_texto(consulta)
    
    if not consulta_preprocesada:
        return "Disculpa, no entendí bien. ¿Podrías explicármelo con otras palabras?"

    consulta_vector = vectorizer.transform([consulta_preprocesada])
    similitudes = cosine_similarity(consulta_vector, matriz_tfidf)
    mejor_sim_score = similitudes.max()
    mejor_sim_index = similitudes.argmax()
    
    if mejor_sim_score > umbral:
        respuesta_medica = df.iloc[mejor_sim_index]['respuesta_validada']
        preambulo = random.choice(FRASES_EMPATIA)
        return preambulo + respuesta_medica
    else:
        registrar_pregunta_en_sheets(consulta)
        return "Entiendo tu pregunta, pero como es un tema delicado y no tengo la respuesta exacta validada por el Dr., prefiero no arriesgarme. Ya dejé anotada tu duda para preguntarle."

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
