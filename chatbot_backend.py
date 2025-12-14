# =======================================================================
# CHATBOT_BACKEND.PY - V10.0 (CEREBRO CONTEXTUAL & ANTI-LOOP)
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
# 1. MAPAS DE LENGUAJE Y SEGURIDAD
# -----------------------------------------------------------------------

# Corrector de "Chilenismos" y Typos Frecuentes
# Esto traduce lo que el paciente escribe a lo que el bot entiende.
CHILENISMOS_MAP = {
    # Typos y Errores comunes detectados
    r"\bquiero la pata\b": "quebre la pierna", # Corrección específica para tu error
    r"\bme quiero\b": "me quebre", 
    
    # Anatomía y Jerga
    r"\bpata\b": "pierna", 
    r"\bguata\b": "estomago", 
    r"\bpucho\b": "cigarro",
    r"\bcago\b": "daño", 
    
    # Intensificadores
    r"\bcaleta\b": "mucho", 
    r"\bbrigido\b": "intenso", 
    r"\bpal gato\b": "mal",
    r"\bmas o menos\b": "regular",
    
    # Verbos/Acciones
    r"\bcachai\b": "entiendes", 
    r"\bpesca\b": "atencion", 
    r"\bal tiro\b": "inmediatamente",
    r"\bsipo\b": "si", 
    r"\byapo\b": "ya",
    r"\bnopo\b": "no"
}

# ALERTA ROJA: Palabras que disparan envío a URGENCIAS inmediatamente.
# Agregamos variantes de fractura.
PALABRAS_ALARMA = [
    "fiebre", "pus", "secreción", "infección", "sangrado abundante", 
    "hemorragia", "dolor insoportable", "desmayo", "no puedo respirar",
    "dedos azules", "no siento la pierna", "calor extremo",
    "se abrió", "abierta", "herida abierta", "hueso expuesto", 
    "tornillo", "supurando", "mal olor", "negro", "necrosis",
    "quebre", "quiebro", "rompi", "fractura", "sono un crack", "cague la operacion"
]

MENSAJE_ALERTA = """
🚨 **ALERTA DE EMERGENCIA** 🚨
Lo que describes parece una complicación grave (posible infección o fractura).
**NO es algo para resolver por chat.**
Por favor, dirígete al **Servicio de Urgencia** más cercano de inmediato.
"""

# CEREBRO SOCIAL (Prioridad Alta para frases cortas)
DICCIONARIO_SOCIAL = {
    "si": "Perfecto. Si te surge otra duda mientras lees, aquí estoy.",
    "bueno": "Quedamos en eso.",
    "ya": "Súper. ¿Algo más?",
    "ok": "Vale, seguimos.",
    "no": "Entendido. A descansar entonces.",
    "gracias": "¡De nada! Vamos paso a paso. 💪",
    "hola": "¡Hola! Soy tu asistente de traumatología. ¿Cómo te sientes hoy?",
    "wena": "¡Wena! ¿Cómo va esa recuperación?",
    "chao": "¡Cuídate! Pata arriba y a descansar.",
    "ayuda": "Dime qué te pasa, estoy aquí para orientarte.",
    "mal": "Pucha, lo siento. La recuperación tiene días difíciles. ¿Es dolor o incomodidad?",
    "bien": "¡Qué buena noticia! Me alegra que vayas bien.",
    "regular": "Paciencia, hay días lentos. Sigue las indicaciones y mejorará."
}

FRASES_EMPATIA = [
    "Es una duda muy frecuente. Te cuento: ",
    "Mira, según el protocolo médico: ",
    "Para tu tranquilidad: ",
    "Entiendo tu preocupación. La indicación es: ",
    "Claro, déjame explicarte este punto: "
]

# -----------------------------------------------------------------------
# 2. MOTOR DE PROCESAMIENTO (NLP)
# -----------------------------------------------------------------------

def normalizar_texto(texto):
    if not isinstance(texto, str): return ""
    texto = texto.lower()
    
    # 1. Aplicar correcciones de mapa (Typos y Chilenismos)
    for slang, standard in CHILENISMOS_MAP.items():
        texto = re.sub(slang, standard, texto)
    
    # 2. Limpieza básica
    texto = ''.join([char for char in texto if char not in string.punctuation])
    return texto

def combinar_columnas(row):
    # Fusionamos Intención + Palabras Clave + Tags para búsqueda amplia
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
    # char_wb + ngram 3-5 permite detectar palabras aunque estén mal escritas
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))
    matriz_tfidf = vectorizer.fit_transform(df['intencion_preprocesada'])
    return vectorizer, matriz_tfidf

# -----------------------------------------------------------------------
# 3. LÓGICA DE DECISIÓN (EL NÚCLEO ARREGLADO)
# -----------------------------------------------------------------------

def revisar_guardrail_emergencia(consulta):
    consulta_norm = normalizar_texto(consulta)
    for p in PALABRAS_ALARMA:
        if p in consulta_norm: 
            return True
    return False

def buscar_respuesta_inteligente(consulta, df, vectorizer, matriz_tfidf, umbral=0.16):
    
    # PASO 1: NORMALIZACIÓN
    texto_norm = normalizar_texto(consulta)
    palabras = texto_norm.split()
    
    # PASO 2: FILTRO SOCIAL ANTICIPADO (Anti-Loop)
    # Si la frase es corta (menos de 3 palabras) Y está en el diccionario social,
    # respondemos eso INMEDIATAMENTE y cortamos el flujo.
    # Esto evita que "si" busque en la base médica.
    if len(palabras) <= 3:
        for palabra in palabras:
            if palabra in DICCIONARIO_SOCIAL:
                return DICCIONARIO_SOCIAL[palabra], []

    # PASO 3: BÚSQUEDA MÉDICA (Si no fue charla social corta)
    consulta_vec = vectorizer.transform([texto_norm])
    similitudes = cosine_similarity(consulta_vec, matriz_tfidf)
    mejor_score = similitudes.max()
    idx = similitudes.argmax()
    
    if mejor_score > umbral:
        respuesta_base = df.iloc[idx]['respuesta_validada']
        tags = df.iloc[idx].get('tags', [])
        preambulo = random.choice(FRASES_EMPATIA)
        return preambulo + respuesta_base, tags

    # PASO 4: FALLBACK (Si no entendió nada)
    return (
        "Esa pregunta es muy específica. Para no arriesgarnos, prefiero dejarla anotada para el Dr. "
        "¿Tienes alguna otra duda sobre cuidados generales, herida o medicamentos?", []
    )

def responder_consulta(consulta, df, vectorizer, matriz_tfidf, contexto_previo=""):
    # 1. Chequeo de Seguridad PRIMERO (Prioridad Absoluta)
    if revisar_guardrail_emergencia(consulta):
        return MENSAJE_ALERTA, []
    
    # 2. Manejo de Contexto (Solo para frases médicas cortas, no para "si" o "gracias")
    texto_norm = normalizar_texto(consulta)
    es_social = any(p in DICCIONARIO_SOCIAL for p in texto_norm.split() if len(texto_norm.split()) <= 3)
    
    if not es_social and len(consulta.split()) < 5 and contexto_previo:
        consulta_aumentada = f"{consulta} {contexto_previo}"
    else:
        consulta_aumentada = consulta

    # 3. Ejecutar búsqueda
    return buscar_respuesta_inteligente(consulta_aumentada, df, vectorizer, matriz_tfidf)

# -----------------------------------------------------------------------
# 4. HERRAMIENTAS DE REGISTRO (GOOGLE SHEETS)
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
