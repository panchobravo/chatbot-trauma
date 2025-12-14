# =======================================================================
# CHATBOT_BACKEND.PY - V8.0 "ULTIMATE CHILEAN CONTEXT AWARE"
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
# 1. BASE CULTURAL Y EMOCIONAL
# -----------------------------------------------------------------------

# INTENSIDAD: Palabras que suben el nivel de urgencia emocional
INTENSIFICADORES_CHILENOS = [
    "caleta", "harto", "brigido", "cuatico", "la cago", "horrible", 
    "insoportable", "muchisimo", "desesperado", "furioso", "ctm", 
    "pal pico", "pal gato", "chacal", "tremendo", "urgente"
]

# TRADUCTOR: Jerga -> Español Neutro
CHILENISMOS_MAP = {
    r"\bcaleta\b": "mucho", r"\bmas o menos\b": "regular", r"\bmaoma\b": "regular",
    r"\breguleque\b": "regular", r"\bpal gato\b": "mal", r"\bhecho bolsa\b": "muy mal",
    r"\bbrigido\b": "intenso", r"\bcuatico\b": "grave", r"\bpata\b": "pierna",
    r"\bguata\b": "estomago", r"\balharaco\b": "exagerado", r"\bcolor\b": "exageracion",
    r"\bcachai\b": "entiendes", r"\bpesca\b": "atencion", r"\bpescar\b": "atender",
    r"\bseco\b": "experto", r"\bpololo\b": "pareja", r"\bpolola\b": "pareja",
    r"\bmarido\b": "esposo", r"\bseñora\b": "esposa", r"\bpucho\b": "cigarro",
    r"\bcaña\b": "resaca", r"\bquedo la escoba\b": "problema grave", r"\btincada\b": "corazonada",
    r"\bchao\b": "adios", r"\bharto\b": "mucho", r"\bsipo\b": "si",
    r"\byapo\b": "ya", r"\bal tiro\b": "inmediatamente", r"\bjoya\b": "excelente",
    r"\bbacan\b": "excelente", r"\bfilete\b": "excelente", r"\bfome\b": "aburrido",
    r"\bcharcha\b": "malo"
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

CHARLA_SOCIAL = {
    "si": "Entiendo. Si el síntoma persiste, revisa las indicaciones anteriores.",
    "sipo": "Vale. Si eso te preocupa, cuéntame más detalles.",
    "obvio": "Claro. ¿En qué más te puedo ayudar?",
    "ya": "Perfecto. ¿Alguna otra duda?",
    "bueno": "Quedamos en eso. ¿Otra consulta?",
    "no": "Entendido. Recuerda mantener reposo.",
    "nopo": "Ok. Avísame si cambia algo.",
    "nada": "Me alegro entonces. ¡A seguir cuidándose!",
    "hola": "¡Hola! ¿Cómo amaneció esa pierna hoy?",
    "wena": "¡Wena! ¿Cómo va la recuperación?",
    "quiubo": "¡Hola! ¿En qué te ayudo?",
    "buenos dias": "¡Buen día! ¿Cómo pasaste la noche?",
    "chao": "¡Cuídate! Pata arriba y a descansar.",
    "gracias": "¡De nada! A ponerle empeño a esa recuperación. 💪",
    "vale": "¡De nada!",
    "eres un robot": "Soy una IA asistente del equipo médico, lista para ayudarte."
}

# --- VARIABILIDAD: FRASES PARA COMBINAR ---
FRASES_EMPATIA_NORMAL = [
    "Te explico lo que indica el protocolo: ",
    "Buena pregunta. Mira: ",
    "Es una duda común. Lo que hacemos es: ",
    "Claro, déjame aclararte este punto: ",
    "Para tu tranquilidad, te cuento: "
]

FRASES_EMPATIA_INTENSA = [
    "Uff, entiendo que es difícil. Pero mira, lo importante es: ",
    "Tranquilo/a, respira. Vamos a resolver esto: ",
    "Veo que te preocupa harto. Déjame explicarte bien: ",
    "Oye, calma. Es normal sentirse así, pero ojo con esto: ",
    "No te angusties. Aquí está la indicación precisa: "
]

CIERRES_HUMANOS = [
    " ¿Te queda más claro así?",
    " ¡Vamos que se puede!",
    " ¿Te parece?",
    " Cualquier otra cosa, me dices.",
    " ¡Ánimo con eso!"
]

# -----------------------------------------------------------------------
# 2. MOTOR NLP (INTELIGENTE)
# -----------------------------------------------------------------------

def normalizar_texto_avanzado(texto):
    if not isinstance(texto, str): return ""
    texto = texto.lower()
    
    # 1. Chilenismos
    for slang, standard in CHILENISMOS_MAP.items():
        texto = re.sub(slang, standard, texto)
    
    # 2. Diminutivos (Stemming casero)
    # Convierte "piecito" -> "pie", "dolorcito" -> "dolor"
    texto = re.sub(r'(\w+)ito\b', r'\1', texto) 
    texto = re.sub(r'(\w+)ita\b', r'\1', texto)
    texto = re.sub(r'(\w+)illos\b', r'\1', texto)
    
    # 3. Limpieza final
    texto = ''.join([char for char in texto if char not in string.punctuation])
    return texto

def detectar_intensidad(texto):
    """Devuelve True si el usuario está muy emocional/intenso"""
    texto = texto.lower()
    for palabra in INTENSIFICADORES_CHILENOS:
        if palabra in texto:
            return True
    return False

def combinar_columnas(row):
    return f"{row['intencion_clave']} {' '.join(row['palabras_clave'])} {' '.join(row.get('tags', []))}"

def cargar_y_preparar_base(archivo_json):
    with open(archivo_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['texto_busqueda'] = df.apply(combinar_columnas, axis=1)
    df['intencion_preprocesada'] = df['texto_busqueda'].apply(normalizar_texto_avanzado)
    return df

def inicializar_vectorizador(df):
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))
    matriz_tfidf = vectorizer.fit_transform(df['intencion_preprocesada'])
    return vectorizer, matriz_tfidf

# -----------------------------------------------------------------------
# 3. CONEXIONES Y MEMORIA (SHEETS)
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
        if sh:
            sh.sheet1.append_row([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), consulta])
    except Exception as e: print(f"Error Log: {e}")

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
    """NUEVO: Guarda qué respuesta fue útil y cuál no"""
    try:
        sh = conectar_sheets()
        if sh:
            try: ws = sh.worksheet("Feedback") # Crea esta hoja en tu Excel
            except: ws = sh.add_worksheet(title="Feedback", rows=1000, cols=4)
            ws.append_row([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), consulta, respuesta[:50], calificacion])
    except Exception as e: print(f"Error Feedback: {e}")

# -----------------------------------------------------------------------
# 4. CEREBRO CENTRAL
# -----------------------------------------------------------------------

def buscar_respuesta_tfidf(consulta, df, vectorizer, matriz_tfidf, umbral=0.18):
    
    # 1. Análisis de Contexto
    texto_normalizado = normalizar_texto_avanzado(consulta)
    es_intenso = detectar_intensidad(consulta)
    
    # 2. Filtro Social Rápido
    palabras = texto_normalizado.split()
    if len(palabras) < 10:
        for frase, resp in CHARLA_SOCIAL.items():
            if frase in texto_normalizado:
                return resp, [] # Retorna respuesta y tags vacíos

    # 3. Búsqueda Vectorial
    consulta_vec = vectorizer.transform([texto_normalizado])
    similitudes = cosine_similarity(consulta_vec, matriz_tfidf)
    mejor_score = similitudes.max()
    idx = similitudes.argmax()
    
    if mejor_score > umbral:
        respuesta_base = df.iloc[idx]['respuesta_validada']
        tags_detectados = df.iloc[idx].get('tags', [])
        
        # Variabilidad: Elegimos prefijo según intensidad
        if es_intenso:
            prefijo = random.choice(FRASES_EMPATIA_INTENSA)
        else:
            prefijo = random.choice(FRASES_EMPATIA_NORMAL)
            
        sufijo = random.choice(CIERRES_HUMANOS)
        
        respuesta_final = prefijo + respuesta_base + sufijo
        return respuesta_final, tags_detectados
    else:
        registrar_pregunta_en_sheets(consulta)
        return "Sabes, esa pregunta es súper específica y prefiero no 'carrilearme'. Dejé anotada tu duda para el Dr. ¿Te ayudo con otra cosa?", []

def revisar_guardrail_emergencia(consulta):
    for p in PALABRAS_ALARMA:
        if p in consulta.lower(): return True
    return False

def responder_consulta(consulta, df, vectorizer, matriz_tfidf, contexto_previo=""):
    # Fusión de Contexto: Si hay contexto previo y la consulta es corta, los unimos
    if len(consulta.split()) < 3 and contexto_previo:
        consulta_aumentada = f"{consulta} {contexto_previo}"
    else:
        consulta_aumentada = consulta

    if revisar_guardrail_emergencia(consulta):
        return MENSAJE_ALERTA, []
    
    return buscar_respuesta_tfidf(consulta_aumentada, df, vectorizer, matriz_tfidf)
