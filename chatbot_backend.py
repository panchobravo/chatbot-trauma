# =======================================================================
# CHATBOT_BACKEND.PY - V9.1 (TONO PROFESIONAL & PRIORIDAD CLÍNICA)
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
import re

# -----------------------------------------------------------------------
# 1. CONFIGURACIÓN DE DATOS
# -----------------------------------------------------------------------

# Mapeo de Jerga -> Español Neutro (Para que el bot entienda, pero no hable mal)
CHILENISMOS_MAP = {
    r"\bcaleta\b": "mucho", r"\bmas o menos\b": "regular", r"\bmaoma\b": "regular",
    r"\bpal gato\b": "mal", r"\bbrigido\b": "intenso", r"\bcuatico\b": "grave", 
    r"\bpata\b": "pierna", r"\bguata\b": "estomago", r"\balharaco\b": "exagerado", 
    r"\bcachai\b": "entiendes", r"\bpesca\b": "atencion", r"\bseco\b": "experto", 
    r"\bpololo\b": "pareja", r"\bpucho\b": "cigarro", r"\bchao\b": "adios", 
    r"\bharto\b": "mucho", r"\bsipo\b": "si", r"\byapo\b": "ya", 
    r"\bal tiro\b": "inmediatamente", r"\bjoya\b": "excelente", r"\bbacan\b": "excelente", 
    r"\bfome\b": "aburrido", r"\bcharcha\b": "malo", r"\bfumar\b": "tabaco",
    r"\bfumo\b": "tabaco", r"\bfumas\b": "tabaco"
}

PALABRAS_ALARMA = [
    "fiebre", "pus", "secreción", "infección", "sangrado abundante", 
    "hemorragia", "dolor insoportable", "desmayo", "no puedo respirar",
    "dedos azules", "no siento la pierna", "calor extremo",
    "se abrió", "abierta", "herida abierta", "veo la placa", "veo el hueso",
    "hueso expuesto", "tornillo", "supurando", "mal olor", "negro", "necrosis"
]

MENSAJE_ALERTA = """
🚨 **ALERTA DE SEGURIDAD** 🚨
Lo que describes requiere evaluación médica inmediata.
Si presentas herida abierta, exposición de material o signos de infección, **NO manipules la zona**.
**Dirígete al Servicio de Urgencia más cercano ahora mismo.**
"""

# Respuestas Sociales (Fallback) - Tono Profesional y Empático
DICCIONARIO_SOCIAL_FALLBACK = {
    "si": "Comprendo. Si el síntoma persiste, por favor sigue las indicaciones de reposo y elevación.",
    "sipo": "Entendido. Si tienes más antecedentes que agregar, estoy atento.",
    "obvio": "Claro. ¿En qué más puedo orientarte?",
    "ya": "Perfecto. ¿Alguna otra consulta?",
    "no": "Bien. Recuerda que el reposo es fundamental para tu evolución.",
    "nopo": "De acuerdo. Cualquier cambio nos avisas.",
    "nada": "Me alegro. A seguir con los cuidados indicados.",
    "hola": "¡Hola! Bienvenido al asistente virtual de Traumatología. ¿Cómo te has sentido?",
    "wena": "¡Hola! ¿Cómo va esa recuperación?",
    "buenos dias": "¡Buen día! ¿Cómo pasaste la noche?",
    "chao": "Hasta luego. Recuerda mantener la extremidad elevada.",
    "gracias": "No hay de qué. Estamos comprometidos con tu recuperación. 💪",
    "vale": "De nada.",
    "eres un robot": "Soy un asistente virtual basado en inteligencia artificial, diseñado para apoyar al equipo médico.",
    "ayuda": "Estoy aquí para orientarte. Cuéntame qué síntoma tienes o qué duda necesitas resolver.",
    "mal": "Lamento escuchar eso. El postoperatorio puede ser difícil. ¿El malestar es por dolor intenso?",
    "pesimo": "Lo siento mucho. Si el dolor o el malestar no ceden con los medicamentos indicados, avísanos.",
    "regular": "Entiendo, hay días de evolución más lenta. Ten paciencia, es parte del proceso cicatrizal.",
    "bien": "¡Qué buena noticia! Una buena evolución nos alegra a todos. Sigue cuidándote.",
    "mejor": "¡Excelente! Significa que el tratamiento está funcionando."
}

FRASES_EMPATIA = [
    "Según nuestro protocolo clínico: ",
    "Es una consulta frecuente. Te explico: ",
    "Para tu tranquilidad, la indicación médica es: ",
    "Respecto a eso, lo importante es: ",
    "Entiendo tu preocupación. La pauta indica: "
]

# -----------------------------------------------------------------------
# 2. PROCESAMIENTO NLP
# -----------------------------------------------------------------------

def normalizar_texto(texto):
    if not isinstance(texto, str): return ""
    texto = texto.lower()
    for slang, standard in CHILENISMOS_MAP.items():
        texto = re.sub(slang, standard, texto)
    # Stemming básico para diminutivos comunes
    texto = re.sub(r'(\w+)ito\b', r'\1', texto) 
    texto = ''.join([char for char in texto if char not in string.punctuation])
    return texto

def combinar_columnas(row):
    tags = " ".join(row.get('tags', [])) if isinstance(row.get('tags'), list) else ""
    return f"{row['intencion_clave']} {' '.join(row['palabras_clave'])} {tags}"

def cargar_y_preparar_base(archivo_json):
    with open(archivo_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['texto_busqueda'] = df.apply(combinar_columnas, axis=1)
    df['intencion_preprocesada'] = df['texto_busqueda'].apply(normalizar_texto)
    return df

def inicializar_vectorizador(df):
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))
    matriz_tfidf = vectorizer.fit_transform(df['intencion_preprocesada'])
    return vectorizer, matriz_tfidf

# -----------------------------------------------------------------------
# 3. CONECTIVIDAD SHEETS
# -----------------------------------------------------------------------

def conectar_sheets():
    if "google_credentials" in st.secrets:
        creds_dict = dict(st.secrets["google_credentials"])
        gc = gspread.service_account_from_dict(creds_dict)
        return gc.open("Cerebro_Bot")
    return None

def registrar_pregunta_en_sheets(consulta):
    try:
        sh = conectar_sheets()
        if sh: sh.sheet1.append_row([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), consulta])
    except: pass

def guardar_paciente_en_sheets(nombre, apellidos, rut, telefono, email):
    try:
        sh = conectar_sheets()
        if sh:
            try: ws = sh.worksheet("Usuarios")
            except: ws = sh.sheet1
            ws.append_row([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), nombre, apellidos, rut, telefono, email])
            return True
    except: return False

def registrar_feedback(consulta, respuesta, calificacion):
    try:
        sh = conectar_sheets()
        if sh:
            try: ws = sh.worksheet("Feedback")
            except: ws = sh.add_worksheet(title="Feedback", rows=1000, cols=4)
            ws.append_row([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), consulta, respuesta[:50], calificacion])
    except: pass

# -----------------------------------------------------------------------
# 4. LÓGICA DE NEGOCIO (PRIORIDAD CLÍNICA)
# -----------------------------------------------------------------------

def buscar_respuesta_tfidf(consulta, df, vectorizer, matriz_tfidf, umbral=0.15):
    
    # 1. Normalización
    texto_norm = normalizar_texto(consulta)
    
    # 2. BÚSQUEDA MÉDICA (PRIORIDAD 1)
    # Ejecutamos la búsqueda vectorial PRIMERO. Si es un tema médico, manda esto.
    consulta_vec = vectorizer.transform([texto_norm])
    similitudes = cosine_similarity(consulta_vec, matriz_tfidf)
    mejor_score = similitudes.max()
    idx = similitudes.argmax()
    
    if mejor_score > umbral:
        respuesta_base = df.iloc[idx]['respuesta_validada']
        tags = df.iloc[idx].get('tags', [])
        preambulo = random.choice(FRASES_EMPATIA)
        return preambulo + respuesta_base, tags

    # 3. FALLBACK SOCIAL (PRIORIDAD 2)
    # Solo si NO es médico, vemos si es un saludo o emoción simple.
    palabras = texto_norm.split()
    for palabra in palabras:
        if palabra in DICCIONARIO_SOCIAL_FALLBACK:
            return DICCIONARIO_SOCIAL_FALLBACK[palabra], []

    # 4. SIN RESPUESTA (FALLBACK PROFESIONAL)
    registrar_pregunta_en_sheets(consulta)
    return (
        "Entiendo tu consulta, pero al ser un caso clínico específico que escapa a mis protocolos generales, "
        "por seguridad prefiero no dar una respuesta automática. "
        "He dejado registrada tu inquietud para que el equipo médico la revise. "
        "¿Hay algo más sobre lo que te pueda orientar mientras tanto?", []
    )

def revisar_guardrail_emergencia(consulta):
    for p in PALABRAS_ALARMA:
        if p in consulta.lower(): return True
    return False

def responder_consulta(consulta, df, vectorizer, matriz_tfidf, contexto_previo=""):
    # Fusión de Contexto
    if len(consulta.split()) < 4 and contexto_previo:
        consulta_aumentada = f"{consulta} {contexto_previo}"
    else:
        consulta_aumentada = consulta

    if revisar_guardrail_emergencia(consulta):
        return MENSAJE_ALERTA, []
    
    return buscar_respuesta_tfidf(consulta_aumentada, df, vectorizer, matriz_tfidf)
