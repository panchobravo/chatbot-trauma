# =======================================================================
# CHATBOT_BACKEND.PY - V6.0 "TOLERANCIA A ERRORES Y TIPOS"
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
# 1. PERSONALIDAD Y DATOS
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
    # Saludos
    "hola": "¡Hola! ¿Cómo amaneció esa pierna hoy?",
    "buenos dias": "¡Buen día! ¿Cómo pasaste la noche?",
    "buenas tardes": "¡Buenas tardes! ¿En qué te puedo ayudar?",
    "chao": "¡Descansa! Intenta mantener la pierna en alto.",
    "adios": "¡Que tengas buen descanso! Cuídate.",
    
    # Estado del Dr.
    "como esta el doctor": "¡El Dr. está a mil por hora operando! Pero me dejó encargado de cuidarlos. ¿Tú cómo sigues?",
    "donde esta el doctor": "Probablemente en pabellón, pero yo tengo acceso a sus protocolos.",

    # Agradecimientos
    "gracias": "¡De nada! Estamos remando juntos en esto. 💪",
    "muchas gracias": "Un placer. Cualquier cosa chica que te preocupe, escríbeme.",
    
    # Identidad
    "eres un robot": "Soy una IA entrenada por el equipo médico, pero créeme que me preocupo por tu recuperación.",
    "eres humano": "Soy tu asistente virtual, pero detrás de mis respuestas está la experiencia de todo el equipo médico.",
    
    # Errores
    "te equivocaste": "¡Ups! Tienes razón, a veces aprendo lento. Gracias por la paciencia.",
    
    # PREGUNTAS DE APERTURA
    "tengo una duda": "Para eso estoy. Cuéntame, ¿qué te preocupa?",
    "quiero hacer una consulta": "Adelante, soy todo oídos. ¿Qué pasó?",
    "puedo hacer una pregunta": "¡Claro que sí! Pregunta con confianza.",
    "necesito ayuda": "Aquí estoy. ¿Es algo urgente o una duda sobre el tratamiento?"
}

RESPUESTAS_EMOCIONALES = {
    "mal": "Uhh, siento escuchar eso. La recuperación es una montaña rusa. ¿Es mucho dolor físico?",
    "pésimo": "Lo siento mucho. Hay días muy duros. ¿Necesitas revisar tu medicación?",
    "regular": "Te entiendo, esos días 'ni fu ni fa' cansan mucho. ¿Te duele algo puntual?",
    "mas o menos": "Ánimo. Es normal no estar al 100% todavía. ¿Cómo va el dolor del 1 al 10?",
    "asustado": "El miedo es normal post-cirugía. No estás solo/a. ¿Qué síntoma te preocupa?",
    "tengo miedo": "Tranquilo/a. Cuéntame qué sientes exactamente y lo revisamos juntos.",
    "triste": "Ánimo... Sé que es difícil estar quieto/a, pero cada día falta menos. 💪",
    "bien": "¡Qué alegría! Esas noticias nos dan energía a todo el equipo.",
    "mejor": "¡Excelente! Significa que vamos por buen camino. Sigue cuidándote."
}

FRASES_EMPATIA = [
    "Te entiendo perfecto. Mira, sobre eso el protocolo es: ",
    "Buena pregunta. Para tu tranquilidad, te cuento: ",
    "Es súper común esa duda. Lo que indicamos siempre es: ",
    "Claro, déjame aclararte ese punto importante: ",
    "Entiendo que eso te preocupe. La indicación médica es: ",
    "Justo el Dr. siempre recalca esto: ",
    "Mira, para que no corras riesgos innecesarios: ",
    "Aquí la regla de oro es la siguiente: ",
    "" 
]

# -----------------------------------------------------------------------
# 2. FUNCIONES DE PROCESAMIENTO
# -----------------------------------------------------------------------

def preprocesar_texto(texto):
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    # Mantenemos solo letras y números, eliminamos puntuación
    texto = ''.join([char for char in texto if char not in string.punctuation])
    return texto

def combinar_columnas(row):
    parte1 = str(row['intencion_clave'])
    parte2 = " ".join(row['palabras_clave'])
    tags = row.get('tags', [])
    if isinstance(tags, list):
        parte3 = " ".join(tags)
    else:
        parte3 = ""
    return parte1 + " " + parte2 + " " + parte3

def cargar_y_preparar_base(archivo_json):
    with open(archivo_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['texto_busqueda'] = df.apply(combinar_columnas, axis=1)
    # Preprocesamos, pero OJO: el vectorizador hará el trabajo pesado de los typos
    df['intencion_preprocesada'] = df['texto_busqueda'].apply(preprocesar_texto)
    return df

def inicializar_vectorizador(df):
    # --- LA MAGIA CONTRA LOS TYPOS ---
    # analyzer='char_wb': Analiza grupos de letras, no palabras enteras.
    # ngram_range=(3, 5): Busca coincidencias de 3, 4 y 5 letras.
    # Esto permite que "funmar" coincida con "fumar" porque comparten "fumar", "uma", "mar".
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))
    matriz_tfidf = vectorizer.fit_transform(df['intencion_preprocesada'])
    return vectorizer, matriz_tfidf

# -----------------------------------------------------------------------
# 3. GOOGLE SHEETS
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

def buscar_respuesta_tfidf(consulta, df, vectorizer, matriz_tfidf, umbral=0.15): 
    # Bajamos umbral a 0.15 porque la búsqueda por caracteres da scores más bajos pero más precisos
    
    consulta_clean = consulta.lower().strip()
    # Quitamos puntuación para la lógica social también
    consulta_limpia_social = ''.join([c for c in consulta_clean if c not in string.punctuation])
    palabras_usuario = consulta_limpia_social.split()

    # 1. FILTRO SOCIAL (Tolerante)
    # Subimos el límite a 12 palabras para aguantar frases como "mmm otra vez eres un robot"
    if len(palabras_usuario) < 12: 
        for frase, respuesta in CHARLA_SOCIAL.items():
            if frase in consulta_limpia_social:
                return respuesta

    # 2. FILTRO EMOCIONAL (Exacto)
    # Buscamos la palabra EXACTA en la lista de palabras del usuario
    # Así "animal" no activa "mal".
    for emocion, respuesta in RESPUESTAS_EMOCIONALES.items():
        if emocion in palabras_usuario: # <--- CAMBIO CLAVE: Búsqueda exacta en lista
            return respuesta

    # 3. BÚSQUEDA MÉDICA (Fuzzy / Typos)
    consulta_preprocesada = preprocesar_texto(consulta)
    
    if not consulta_preprocesada:
        return "Disculpa, no te capté bien. ¿Me lo podrías explicar con otras palabras? 🤔"

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
        return (
            "Sabes, tu pregunta es súper específica y prefiero no 'carrilearme' (improvisar). "
            "Como es un tema médico delicado, mejor dejé anotada tu duda para que el Dr. la revise. "
            "Mientras tanto, ¿hay algo más estándar en lo que te pueda orientar?"
        )

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
