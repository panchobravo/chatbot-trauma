# =======================================================================
# CHATBOT_BACKEND.PY - V12.0 (CEREBRO HÍBRIDO: LÓGICA + PERSONALIDAD)
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
# 1. BASE CULTURAL, EMOCIONAL Y DE SEGURIDAD
# -----------------------------------------------------------------------

# MAPA DE TRADUCCIÓN (Restaurado completo)
CHILENISMOS_MAP = {
    # Typos y Errores graves
    r"\bquiero la pata\b": "quebre la pierna", r"\bme quiero\b": "me quebre", 
    r"\bme saque la cresta\b": "caida grave",
    
    # Anatomía y Objetos
    r"\bpata\b": "pierna", r"\bguata\b": "estomago", r"\bpucho\b": "cigarro",
    r"\bfumo\b": "tabaco", r"\bfumas\b": "tabaco", r"\bfumar\b": "tabaco",
    
    # Intensidad y Estado
    r"\bcago\b": "daño", r"\bcaleta\b": "mucho", r"\bbrigido\b": "intenso", 
    r"\bpal gato\b": "mal", r"\bmas o menos\b": "regular", r"\bmaoma\b": "regular",
    r"\bhecho bolsa\b": "muy mal", r"\bcuatico\b": "grave",
    
    # Modismos Positivos (Restaurados)
    r"\bjoya\b": "excelente", r"\bfilete\b": "excelente", r"\bbacan\b": "excelente",
    r"\bseco\b": "experto",
    
    # Verbos y Conectores
    r"\bcachai\b": "entiendes", r"\bpesca\b": "atencion", r"\bal tiro\b": "inmediatamente", 
    r"\bsipo\b": "si", r"\byapo\b": "ya", r"\bnopo\b": "no",
    
    # Groserías (para normalizar detección)
    r"\bctm\b": "dolor terrible", r"\bmierda\b": "dolor", r"\bconchetumare\b": "dolor terrible"
}

GROSERIAS_DOLOR = ["ctm", "conchetumare", "mierda", "pico", "puta", "recontra", "chucha"]

PALABRAS_ALARMA = [
    "fiebre", "pus", "secreción", "infección", "sangrado abundante", 
    "hemorragia", "dolor insoportable", "desmayo", "no puedo respirar",
    "dedos azules", "no siento la pierna", "calor extremo",
    "se abrió", "abierta", "herida abierta", "hueso expuesto", 
    "tornillo", "supurando", "mal olor", "negro", "necrosis",
    "quebre", "quiebro", "rompi", "fractura", "sono un crack"
]

MENSAJE_ALERTA = """
🚨 **ALERTA DE SEGURIDAD** 🚨
Lo que describes NO es normal y requiere evaluación médica inmediata.
**NO manipules la zona.**
Por favor, dirígete al **Servicio de Urgencia** más cercano ahora mismo.
"""

# DICCIONARIO "PERSONALIDAD + ANTI-LOOP"
# Si la frase es corta y está aquí, responde esto directo.
RESPUESTAS_RAPIDAS = {
    # Loop Breakers (Afirmaciones/Negaciones)
    "si": "Perfecto. Si tienes otra duda, dímela.",
    "bueno": "Quedamos en eso.",
    "ya": "Súper. ¿Algo más?",
    "ok": "Vale, seguimos.",
    "no": "Entendido. A descansar entonces.",
    "nada": "Me alegro. ¡A seguir cuidándose!",
    
    # Saludos y Cortesía (Restaurados)
    "hola": "¡Hola! Bienvenido al asistente virtual del Equipo de Tobillo y Pie. ¿Cómo amaneciste hoy?",
    "wena": "¡Wena! ¿Cómo va esa recuperación?",
    "buenos dias": "¡Buen día! Espero que hayas descansado bien.",
    "buenas tardes": "¡Buenas tardes! Aquí estoy atento a tus dudas.",
    "chao": "¡Cuídate mucho! Pata arriba y a descansar.",
    "gracias": "¡De nada! Estamos comprometidos contigo. 💪",
    "vale": "De nada.",
    
    # Identidad y Doctor (Restaurado)
    "eres un robot": "Soy una IA entrenada por el equipo médico para acompañarte 24/7.",
    "quien eres": "Soy tu asistente virtual de Traumatología.",
    "como esta el doctor": "¡El Dr. está a mil operando! Pero me dejó todos sus protocolos para ayudarte.",
    "donde esta el doctor": "Probablemente en pabellón salvando tobillos, pero yo te ayudo por mientras.",
    
    # Emociones (Restaurado)
    "ayuda": "Estoy aquí. Cuéntame qué sientes o qué duda tienes.",
    "mal": "Pucha, lo siento. La recuperación es una montaña rusa. ¿Es mucho dolor físico?",
    "pesimo": "Lo siento mucho. Si el dolor no cede con los remedios, avísanos.",
    "regular": "Paciencia, hay días lentos. Es parte del proceso de sanar.",
    "bien": "¡Qué buena noticia! Me alegra mucho leer eso.",
    "mejor": "¡Excelente! Vamos por buen camino.",
    "tengo miedo": "Es súper normal sentir miedo, no te angusties. Cuéntame qué te asusta y lo revisamos.",
    "estoy triste": "Ánimo... Sé que aburre estar quieto, pero piensa que el hueso se está pegando ahora mismo. 💪"
}

# Frases suaves para situaciones normales
FRASES_EMPATIA = [
    "Es una duda muy frecuente. Te cuento: ",
    "Mira, según el protocolo médico: ",
    "Para tu tranquilidad: ",
    "Entiendo tu preocupación. La indicación es: ",
    "Claro, déjame explicarte este punto: "
]

# Frases directas para urgencia/dolor
FRASES_URGENCIA = [
    "Entiendo que el dolor es fuerte. Ojo con esto: ",
    "Mantén la calma. Mira: ",
    "Vale, vamos al grano: ",
    ""
]

# -----------------------------------------------------------------------
# 2. MOTOR NLP
# -----------------------------------------------------------------------

def normalizar_texto(texto):
    if not isinstance(texto, str): return ""
    texto = texto.lower()
    for slang, standard in CHILENISMOS_MAP.items():
        texto = re.sub(slang, standard, texto)
    # Stemming diminutivos
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
# 4. LÓGICA DE DECISIÓN (CEREBRO PRINCIPAL)
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
    
    # 1. FILTRO RÁPIDO (PERSONALIDAD & ANTI-LOOP)
    # Si la frase es corta (<= 4 palabras), revisamos si es social/emocional antes de buscar enfermedades.
    # Esto recupera respuestas como "tengo miedo" o "donde esta el doctor".
    if len(palabras) <= 4:
        # Buscamos frases exactas o palabras clave en el diccionario
        for frase, resp in RESPUESTAS_RAPIDAS.items():
            if frase in texto_norm: 
                return resp, [] # Retorna respuesta directa y limpia contexto

    # 2. BÚSQUEDA MÉDICA (Si no fue social corta)
    consulta_vec = vectorizer.transform([texto_norm])
    similitudes = cosine_similarity(consulta_vec, matriz_tfidf)
    mejor_score = similitudes.max()
    idx = similitudes.argmax()
    
    if mejor_score > umbral:
        respuesta_base = df.iloc[idx]['respuesta_validada']
        tags = df.iloc[idx].get('tags', [])
        
        # Ajuste de Tono (Si hay groserías, no usamos empatía suave)
        if detectar_groseria(consulta):
            preambulo = random.choice(FRASES_URGENCIA)
        else:
            preambulo = random.choice(FRASES_EMPATIA)
            
        return preambulo + respuesta_base, tags

    # 3. FALLBACK (Si no entendió nada)
    return (
        "Esa pregunta es muy específica y prefiero no improvisar por tu seguridad. "
        "La dejaré registrada para el Dr. ¿Tienes alguna duda sobre cuidados, heridas o medicamentos?", []
    )

def responder_consulta(consulta, df, vectorizer, matriz_tfidf, contexto_previo=""):
    
    # ALERTA DE SEGURIDAD (Siempre primero)
    if revisar_guardrail_emergencia(consulta):
        return MENSAJE_ALERTA, [] 
    
    # GESTIÓN DE CONTEXTO ESTRICTA (V11)
    # Solo agregamos contexto si la consulta es MUY corta (< 3 palabras)
    texto_split = consulta.split()
    if len(texto_split) < 3 and contexto_previo:
        consulta_aumentada = f"{consulta} {contexto_previo}"
    else:
        consulta_aumentada = consulta

    return buscar_respuesta_inteligente(consulta_aumentada, df, vectorizer, matriz_tfidf)
