# =======================================================================
# CHATBOT_BACKEND.PY - V8.1 HOTFIX (CORRECCIÓN DE EMERGENCIA)
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
# 1. CONFIGURACIÓN Y DICCIONARIOS
# -----------------------------------------------------------------------

# Diccionario de Modismos (Traductor)
CHILENISMOS_MAP = {
    r"\bcaleta\b": "mucho", r"\bmas o menos\b": "regular", r"\bmaoma\b": "regular",
    r"\bpal gato\b": "mal", r"\bbrigido\b": "intenso", r"\bcuatico\b": "grave", 
    r"\bpata\b": "pierna", r"\bguata\b": "estomago", r"\balharaco\b": "exagerado", 
    r"\bcachai\b": "entiendes", r"\bpesca\b": "atencion", r"\bseco\b": "experto", 
    r"\bpololo\b": "pareja", r"\bpucho\b": "cigarro", r"\bchao\b": "adios", 
    r"\bharto\b": "mucho", r"\bsipo\b": "si", r"\byapo\b": "ya", 
    r"\bal tiro\b": "inmediatamente", r"\bjoya\b": "excelente", r"\bbacan\b": "excelente", 
    r"\bfome\b": "aburrido", r"\bcharcha\b": "malo"
}

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

# Diccionario Mixto: Social + Emocional (Unificado para que no fallen)
DICCIONARIO_RAPIDO = {
    "si": "Entiendo. Si el síntoma persiste, revisa las indicaciones anteriores.",
    "sipo": "Vale. Si eso te preocupa, cuéntame más detalles.",
    "obvio": "Claro. ¿En qué más te puedo ayudar?",
    "ya": "Perfecto. ¿Alguna otra duda?",
    "no": "Entendido. Recuerda mantener reposo.",
    "nopo": "Ok. Avísame si cambia algo.",
    "nada": "Me alegro entonces. ¡A seguir cuidándose!",
    "hola": "¡Hola! ¿Cómo amaneció esa pierna hoy?",
    "wena": "¡Wena! ¿Cómo va la recuperación?",
    "buenos dias": "¡Buen día! ¿Cómo pasaste la noche?",
    "chao": "¡Cuídate! Pata arriba y a descansar.",
    "gracias": "¡De nada! A ponerle empeño a esa recuperación. 💪",
    "vale": "¡De nada!",
    "eres un robot": "Soy una IA asistente del equipo médico, lista para ayudarte.",
    "ayuda": "Estoy aquí. Cuéntame qué te pasa o qué duda tienes.",
    
    # Emociones
    "mal": "Pucha, qué lata. La recuperación tiene días pesados. ¿Es mucho dolor físico?",
    "pesimo": "Lo siento mucho. A veces dan ganas de tirar la toalla, pero falta poco. ¿Revisamos tus remedios?",
    "regular": "Ya veo, esos días 'ni fu ni fa'. Paciencia, es parte del proceso.",
    "bien": "¡Buena! Esas noticias nos alegran el día. Sigue así.",
    "mejor": "¡Excelente! Significa que vamos impeque. A no descuidarse eso sí."
}

FRASES_EMPATIA = [
    "Te explico lo que indica el protocolo: ",
    "Buena pregunta. Mira: ",
    "Es una duda común. Lo que hacemos es: ",
    "Claro, déjame aclararte este punto: ",
    "Para tu tranquilidad, te cuento: "
]

# -----------------------------------------------------------------------
# 2. MOTOR NLP (SIMPLIFICADO Y SEGURO)
# -----------------------------------------------------------------------

def normalizar_texto(texto):
    if not isinstance(texto, str): return ""
    texto = texto.lower()
    
    # 1. Chilenismos
    for slang, standard in CHILENISMOS_MAP.items():
        texto = re.sub(slang, standard, texto)
    
    # 2. Diminutivos
    texto = re.sub(r'(\w+)ito\b', r'\1', texto) 
    
    # 3. Limpieza de puntuación
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
    # Usamos char_wb con rango 3-5 para tolerancia a typos
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))
    matriz_tfidf = vectorizer.fit_transform(df['intencion_preprocesada'])
    return vectorizer, matriz_tfidf

# -----------------------------------------------------------------------
# 3. SHEETS & UTILS
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
# 4. LÓGICA CENTRAL (ARREGLADA)
# -----------------------------------------------------------------------

def buscar_respuesta_tfidf(consulta, df, vectorizer, matriz_tfidf, umbral=0.15): # Umbral bajado a 0.15 para captar mejor "fumar"
    
    # 1. Normalización
    texto_norm = normalizar_texto(consulta)
    palabras = texto_norm.split()

    # 2. FILTRO RÁPIDO (SOCIAL / EMOCIONAL) - ¡AQUÍ ESTABA EL ERROR!
    # Ahora revisamos si ALGUNA palabra de la frase está en nuestro diccionario rápido
    # Esto asegura que "hola ayuda" o "estoy mal" funcionen.
    for palabra in palabras:
        if palabra in DICCIONARIO_RAPIDO:
            return DICCIONARIO_RAPIDO[palabra], []

    # 3. BÚSQUEDA MÉDICA (VECTORIAL)
    consulta_vec = vectorizer.transform([texto_norm])
    similitudes = cosine_similarity(consulta_vec, matriz_tfidf)
    mejor_score = similitudes.max()
    idx = similitudes.argmax()
    
    if mejor_score > umbral:
        respuesta_base = df.iloc[idx]['respuesta_validada']
        tags = df.iloc[idx].get('tags', [])
        preambulo = random.choice(FRASES_EMPATIA)
        return preambulo + respuesta_base, tags
    else:
        registrar_pregunta_en_sheets(consulta)
        return (
            "Sabes, esa pregunta es súper específica y prefiero no 'carrilearme'. "
            "Dejé anotada tu duda para el Dr. ¿Hay algo más estándar en lo que te pueda orientar?", []
        )

def revisar_guardrail_emergencia(consulta):
    for p in PALABRAS_ALARMA:
        if p in consulta.lower(): return True
    return False

def responder_consulta(consulta, df, vectorizer, matriz_tfidf, contexto_previo=""):
    # Fusión de Contexto simple
    if len(consulta.split()) < 4 and contexto_previo:
        consulta_aumentada = f"{consulta} {contexto_previo}"
    else:
        consulta_aumentada = consulta

    if revisar_guardrail_emergencia(consulta):
        return MENSAJE_ALERTA, []
    
    return buscar_respuesta_tfidf(consulta_aumentada, df, vectorizer, matriz_tfidf)
