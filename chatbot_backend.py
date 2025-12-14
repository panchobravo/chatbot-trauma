# =======================================================================
# CHATBOT_BACKEND.PY - V11.1 (INTELIGENCIA + SHEETS RESTAURADO)
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

CHILENISMOS_MAP = {
    r"\bquiero la pata\b": "quebre la pierna", 
    r"\bme quiero\b": "me quebre", 
    r"\bme saque la cresta\b": "caida grave",
    r"\bpata\b": "pierna", r"\bguata\b": "estomago", r"\bpucho\b": "cigarro",
    r"\bcago\b": "daño", r"\bcaleta\b": "mucho", r"\bbrigido\b": "intenso", 
    r"\bpal gato\b": "mal", r"\bmas o menos\b": "regular", r"\bcachai\b": "entiendes", 
    r"\bpesca\b": "atencion", r"\bal tiro\b": "inmediatamente", r"\bsipo\b": "si", 
    r"\byapo\b": "ya", r"\bnopo\b": "no", r"\bctm\b": "dolor terrible",
    r"\bmierda\b": "dolor", r"\bconchetumare\b": "dolor terrible"
}

# Palabras que indican que el paciente está desesperado o insultando del dolor
GROSERIAS_DOLOR = ["ctm", "conchetumare", "mierda", "pico", "puta", "recontra"]

PALABRAS_ALARMA = [
    "fiebre", "pus", "secreción", "infección", "sangrado abundante", 
    "hemorragia", "dolor insoportable", "desmayo", "no puedo respirar",
    "dedos azules", "no siento la pierna", "calor extremo",
    "se abrió", "abierta", "herida abierta", "hueso expuesto", 
    "tornillo", "supurando", "mal olor", "negro", "necrosis",
    "quebre", "quiebro", "rompi", "fractura", "sono un crack"
]

MENSAJE_ALERTA = """
🚨 **ALERTA DE EMERGENCIA** 🚨
Lo que describes parece una complicación grave.
**NO es algo para resolver por chat.**
Por favor, dirígete al **Servicio de Urgencia** más cercano de inmediato.
"""

DICCIONARIO_SOCIAL = {
    "si": "Bien. Si tienes otra duda, dímela.",
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

# Frases suaves para situaciones normales
FRASES_EMPATIA = [
    "Es una duda muy frecuente. Te cuento: ",
    "Mira, según el protocolo médico: ",
    "Para tu tranquilidad: ",
    "Entiendo tu preocupación. La indicación es: ",
    "Claro, déjame explicarte este punto: "
]

# Frases directas para cuando hay groserías/dolor intenso (Sin rodeos)
FRASES_URGENCIA = [
    "Entiendo que el dolor es fuerte. Ojo con esto: ",
    "Mantén la calma. Mira: ",
    "Vale, vamos al grano: ",
    "" # A veces mejor no decir nada y dar la info
]

# -----------------------------------------------------------------------
# 2. MOTOR NLP
# -----------------------------------------------------------------------

def normalizar_texto(texto):
    if not isinstance(texto, str): return ""
    texto = texto.lower()
    for slang, standard in CHILENISMOS_MAP.items():
        texto = re.sub(slang, standard, texto)
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
# 3. CONEXIÓN A GOOGLE SHEETS (RESTAURADA)
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
# 4. LÓGICA DE DECISIÓN
# -----------------------------------------------------------------------

def revisar_guardrail_emergencia(consulta):
    consulta_norm = normalizar_texto(consulta)
    for p in PALABRAS_ALARMA:
        if p in consulta_norm: return True
    return False

def detectar_groseria(texto):
    for g in GROSERIAS_DOLOR:
        if g in texto.lower(): return True
    return False

def buscar_respuesta_inteligente(consulta, df, vectorizer, matriz_tfidf, umbral=0.16):
    texto_norm = normalizar_texto(consulta)
    palabras = texto_norm.split()
    
    # 1. Filtro Social (Anti-Loop)
    if len(palabras) <= 3:
        for palabra in palabras:
            if palabra in DICCIONARIO_SOCIAL:
                return DICCIONARIO_SOCIAL[palabra], [] 

    # 2. Búsqueda Médica
    consulta_vec = vectorizer.transform([texto_norm])
    similitudes = cosine_similarity(consulta_vec, matriz_tfidf)
    mejor_score = similitudes.max()
    idx = similitudes.argmax()
    
    if mejor_score > umbral:
        respuesta_base = df.iloc[idx]['respuesta_validada']
        tags = df.iloc[idx].get('tags', [])
        
        # Preámbulo según tono
        if detectar_groseria(consulta):
            preambulo = random.choice(FRASES_URGENCIA)
        else:
            preambulo = random.choice(FRASES_EMPATIA)
            
        return preambulo + respuesta_base, tags

    # 3. Fallback
    return (
        "Esa pregunta es muy específica y no quiero improvisar. "
        "La dejaré anotada para el Dr. ¿Tienes alguna duda sobre herida, dolor o reposo?", []
    )

def responder_consulta(consulta, df, vectorizer, matriz_tfidf, contexto_previo=""):
    
    # ALERTA
    if revisar_guardrail_emergencia(consulta):
        return MENSAJE_ALERTA, [] 
    
    # CONTEXTO ESTRICTO (<3 palabras y existe contexto)
    texto_split = consulta.split()
    if len(texto_split) < 3 and contexto_previo:
        consulta_aumentada = f"{consulta} {contexto_previo}"
    else:
        consulta_aumentada = consulta

    return buscar_respuesta_inteligente(consulta_aumentada, df, vectorizer, matriz_tfidf)
