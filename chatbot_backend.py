# =======================================================================
# CHATBOT_BACKEND.PY - V7.0 "CHILEAN EDITION & CONTEXT AWARE"
# =======================================================================
# Autor: Arquitectura de Software - Nivel Senior
# Descripción: Backend robusto con normalización de modismos locales (CL)
#              y manejo de interacciones cortas (Afirmaciones/Negaciones).
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
# 1. BASE DE CONOCIMIENTO & CULTURA (CONFIGURACIÓN)
# -----------------------------------------------------------------------

# Diccionario de traducción de "Chileno" a "Español Clínico"
CHILENISMOS_MAP = {
    r"\bcaleta\b": "mucho",
    r"\bmas o menos\b": "regular",
    r"\bmaoma\b": "regular",
    r"\breguleque\b": "regular",
    r"\bpal gato\b": "mal",
    r"\bhecho bolsa\b": "muy mal",
    r"\bbrigido\b": "intenso",
    r"\bcuatico\b": "grave",
    r"\bpata\b": "pierna",
    r"\bguata\b": "estomago",
    r"\balharaco\b": "exagerado",
    r"\bcolor\b": "exageracion", # Ej: "le pones color"
    r"\bcachai\b": "entiendes",
    r"\bpesca\b": "atencion", # Ej: "no me tomas pesca"
    r"\bpescar\b": "atender",
    r"\bseco\b": "experto",
    r"\bpololo\b": "pareja",
    r"\bpolola\b": "pareja",
    r"\bmarido\b": "esposo",
    r"\bseñora\b": "esposa",
    r"\bpucho\b": "cigarro",
    r"\bcaña\b": "resaca",
    r"\bquedo la escoba\b": "problema grave",
    r"\btincada\b": "corazonada",
    r"\bchao\b": "adios",
    r"\bharto\b": "mucho",
    r"\bsipo\b": "si",
    r"\byapo\b": "ya",
    r"\bal tiro\b": "inmediatamente",
    r"\bjoya\b": "excelente",
    r"\bbacan\b": "excelente",
    r"\bfilete\b": "excelente",
    r"\bfome\b": "aburrido",
    r"\bcharcha\b": "malo"
}

PALABRAS_ALARMA = [
    "fiebre", "pus", "secreción", "infección", "sangrado abundante", 
    "hemorragia", "dolor insoportable", "desmayo", "no puedo respirar",
    "dedos azules", "no siento la pierna", "calor extremo",
    "se abrió", "abierta", "herida abierta", "veo la placa", "veo el hueso",
    "hueso expuesto", "tornillo", "supurando", "mal olor", "negro", "necrosis",
    "se me abrieron", "sangre viva"
]

MENSAJE_ALERTA = """
🚨 **ALERTA DE EMERGENCIA** 🚨
Lo que describes NO es normal y requiere evaluación médica presencial inmediata.
Si la herida se abrió, ves material (placas/hueso) o hay infección, **NO toques nada**.
**Dirígete a Urgencias ahora mismo.**
"""

# Respuestas para interacciones cortas (Si/No/Saludos)
CHARLA_SOCIAL = {
    # Afirmaciones (El parche para tu error de "Si")
    "si": "Entiendo. Si el síntoma persiste, revisa las indicaciones que te di. ¿Hay algo más específico que quieras saber?",
    "sipo": "Vale. Si eso te preocupa, cuéntame más detalles para buscar en mis archivos médicos.",
    "obvio": "Claro. ¿En qué más te puedo ayudar?",
    "ya": "Perfecto. ¿Alguna otra duda?",
    "bueno": "Quedamos en eso. ¿Otra consulta?",
    
    # Negaciones
    "no": "Entendido. Si no tienes más dudas por ahora, recuerda mantener reposo.",
    "nopo": "Ok. Avísame si cambia algo.",
    "nada": "Me alegro entonces. ¡A seguir cuidándose!",
    
    # Saludos y Modismos
    "hola": "¡Hola! ¿Cómo amaneció esa pierna hoy?",
    "wena": "¡Wena! ¿Cómo va la recuperación?",
    "quiubo": "¡Hola! ¿En qué te ayudo?",
    "buenos dias": "¡Buen día! ¿Cómo pasaste la noche?",
    "buenas tardes": "¡Buenas tardes! Aquí atento a tus dudas.",
    "chao": "¡Cuídate! Pata arriba y a descansar.",
    
    # Identidad
    "eres un robot": "Soy una IA asistente del equipo médico. No tomo café, pero me sé todos los protocolos.",
    "quien eres": "Soy el asistente virtual de Traumatología. Estoy aquí para resolver dudas rápidas.",
    
    # Gratitud
    "gracias": "¡De nada! A ponerle empeño a esa recuperación. 💪",
    "vale": "¡De nada!",
    "te pasaste": "¡Gracias a ti por la paciencia! Estamos para ayudar."
}

RESPUESTAS_EMOCIONALES = {
    "mal": "Pucha, qué lata escuchar eso. La recuperación tiene días bien pesados. ¿Es mucho dolor físico?",
    "pesimo": "Lo siento mucho. A veces dan ganas de tirar la toalla, pero falta poco. ¿Necesitas revisar tus remedios?",
    "regular": "Ya veo, esos días 'ni fu ni fa'. Paciencia, es parte del proceso. ¿Te duele algo puntual?",
    "mas o menos": "Ánimo. Es normal no estar al 100% todavía. ¿Del 1 al 10, cuánto te duele?",
    "asustado": "Es súper normal tener susto, sobre todo si es tu primera cirugía. Pero aquí estamos. ¿Qué sientes raro?",
    "tengo miedo": "Tranquilo. Cuéntame qué sientes exactamente y lo revisamos juntos para que te quedes tranquilo.",
    "triste": "Arriba ese ánimo. Sé que aburre estar quieto, pero piensa que el hueso se está pegando ahora mismo. 💪",
    "bien": "¡Buena! Esas noticias nos alegran el día. Sigue así.",
    "mejor": "¡Excelente! Significa que vamos impeque. A no descuidarse eso sí."
}

FRASES_EMPATIA = [
    "Te cacho perfecto. Mira, el protocolo dice: ",
    "Buena pregunta. Para que te quedes tranquilo: ",
    "Es típica esa duda. Lo que indicamos siempre es: ",
    "Claro, déjame explicarte eso: ",
    "Entiendo que te urgalla eso. La indicación médica es: ",
    "Justo el Dr. siempre recalca esto: ",
    "Mira, para que no corras riesgos innecesarios: ",
    "Aquí la regla de oro es: ",
    "" 
]

# -----------------------------------------------------------------------
# 2. MOTOR DE PROCESAMIENTO (NLP AVANZADO)
# -----------------------------------------------------------------------

def normalizar_chilenismos(texto):
    """Reemplaza jerga chilena por español neutro para mejorar la búsqueda"""
    texto = texto.lower()
    for slang, standard in CHILENISMOS_MAP.items():
        # Usamos regex para reemplazar solo palabras completas
        texto = re.sub(slang, standard, texto)
    return texto

def preprocesar_texto(texto):
    if not isinstance(texto, str):
        return ""
    
    # 1. Normalización cultural (Chilenismos)
    texto = normalizar_chilenismos(texto)
    
    # 2. Limpieza estándar
    texto = ''.join([char for char in texto if char not in string.punctuation])
    
    return texto

def combinar_columnas(row):
    """Crea el 'Documento' de búsqueda unificando intención + palabras clave + tags"""
    parte1 = str(row['intencion_clave'])
    parte2 = " ".join(row['palabras_clave'])
    tags = row.get('tags', [])
    parte3 = " ".join(tags) if isinstance(tags, list) else ""
    return parte1 + " " + parte2 + " " + parte3

def cargar_y_preparar_base(archivo_json):
    with open(archivo_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    
    # Creamos el campo de búsqueda enriquecido
    df['texto_busqueda'] = df.apply(combinar_columnas, axis=1)
    
    # Preprocesamos la base de datos también (para que 'pucho' coincida con 'cigarro' si está mapeado)
    df['intencion_preprocesada'] = df['texto_busqueda'].apply(preprocesar_texto)
    return df

def inicializar_vectorizador(df):
    # Usamos char_wb con rango 3-5 para tolerancia a typos (ej: "dolr" -> "dolor")
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))
    matriz_tfidf = vectorizer.fit_transform(df['intencion_preprocesada'])
    return vectorizer, matriz_tfidf

# -----------------------------------------------------------------------
# 3. INTERFAZ DE DATOS (GOOGLE SHEETS)
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
# 4. LÓGICA CENTRAL DEL CHATBOT
# -----------------------------------------------------------------------

def buscar_respuesta_tfidf(consulta, df, vectorizer, matriz_tfidf, umbral=0.18):
    
    # 1. Preprocesamiento Cultural
    # Si el usuario dice "me duele la pata", internamente buscamos "me duele la pierna"
    consulta_normalizada = normalizar_chilenismos(consulta)
    consulta_clean = consulta_normalizada.lower().strip()
    
    # Quitamos puntuación para comparaciones exactas de listas
    consulta_sin_puntuacion = ''.join([c for c in consulta_clean if c not in string.punctuation])
    palabras_usuario = consulta_sin_puntuacion.split()

    # 2. FILTRO SOCIAL Y AFIRMACIONES (Prioridad Alta, Tolerante)
    # Aceptamos frases de hasta 10 palabras. Si dice "sipo", entra aquí.
    if len(palabras_usuario) < 10: 
        # Búsqueda exacta de frase en diccionario
        for frase, respuesta in CHARLA_SOCIAL.items():
            if frase == consulta_sin_puntuacion: # Coincidencia exacta (ej: "si")
                return respuesta
            if frase in consulta_sin_puntuacion and len(frase) > 3: # Coincidencia parcial para frases largas
                return respuesta

    # 3. FILTRO EMOCIONAL (Búsqueda de palabras clave)
    for emocion, respuesta in RESPUESTAS_EMOCIONALES.items():
        if emocion in palabras_usuario:
            return respuesta

    # 4. BÚSQUEDA MÉDICA (Vectorial TF-IDF)
    # Usamos la consulta normalizada (sin chilenismos)
    consulta_final = preprocesar_texto(consulta)
    
    if not consulta_final:
        return "Disculpa, no te capté. ¿Me lo podrías explicar de nuevo? 🤔"

    consulta_vector = vectorizer.transform([consulta_final])
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
            "Sabes, esa pregunta es súper específica y prefiero no 'carrilearme' (improvisar). "
            "Como es un tema médico, dejé anotada tu duda para preguntarle al Dr. "
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
