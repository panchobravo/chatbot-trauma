# =======================================================================
# CHATBOT_BACKEND.PY - V5.1 "BLINDADA" (Sintaxis Simplificada)
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
    "como esta el doctor": "¡El Dr. está a mil por hora operando! Pero me dejó encargado de cuidarlos. ¿Tú cómo sigues?",
    "gracias": "¡De nada! Estamos remando juntos en esto. 💪",
    "muchas gracias": "Un placer. Cualquier cosa chica que te preocupe, escríbeme.",
    "hola": "¡Hola! ¿Cómo amaneció esa pierna hoy?",
    "chao": "¡Descansa! Intenta mantener la pierna en alto un ratito.",
    "adios": "¡Que tengas buen descanso! Cuídate.",
    "eres un robot": "Soy una IA entrenada por el equipo médico, pero créeme que me preocupo por tu recuperación.",
    "eres humano": "Soy tu asistente virtual, pero detrás de mis respuestas está la experiencia de todo el equipo médico.",
    "te equivocaste": "¡Ups! Tienes razón, a veces aprendo lento. Gracias por la paciencia.",
    "buenos dias": "¡Buen día! ¿Cómo pasaste la noche?",
    "buenas tardes": "¡Buenas tardes! ¿En qué te puedo ayudar en este momento?"
}

RESPUESTAS_EMOCIONALES = {
    "mal": "Uhh, siento escuchar eso. La recuperación es una montaña rusa, hay días malos. ¿Es mucho dolor o es el encierro?",
    "mas o menos": "Te entiendo. Esos días 'ni fu ni fa' son pesados. ¿Te duele algo puntual o es cansancio general?",
    "asustado": "Es súper normal tener miedo, sobre todo después de una cirugía. Aquí estamos para darte seguridad. ¿Qué síntoma te asusta?",
    "tengo miedo": "Tranquilo/a. El miedo es normal, pero no dejes que te paralice. Cuéntame qué sientes y lo revisamos.",
    "triste": "Ánimo... Sé que es difícil estar quieto/a tanto tiempo, pero piensa que cada día es uno menos para el alta. 💪",
    "bien": "¡Qué alegría leer eso! Esas son las noticias que nos gusta recibir. Sigue así.",
    "mejor": "¡Excelente! Significa que el cuerpo está haciendo su trabajo. No bajemos la guardia eso sí."
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
    texto = ''.join([char for char in texto if char not in string.punctuation])
    try:
        stop_words_es = stopwords.words('spanish')
    except:
        stop_words_es = ["el", "la", "los", "las", "un", "una", "y", "o", "de", "a", "en", "que", "me", "mi", "mis", "con", "por", "para"]
    
    palabras = texto.split()
    palabras_filtradas = [w for w in palabras if w not in stop_words_es]
    return ' '.join(palabras_filtradas)

def combinar_columnas(row):
    """Función auxiliar para evitar errores de sintaxis en lambdas complejas"""
    parte1 = str(row['intencion_clave'])
    parte2 = " ".join(row['palabras_clave'])
    # Manejo seguro de tags
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
    
    # Usamos la función auxiliar en lugar de lambda
    df['texto_busqueda'] = df.apply(combinar_columnas, axis=1)
    
    df['intencion_preprocesada'] = df['texto_busqueda'].apply(preprocesar_texto)
    return df

def inicializar_vectorizador(df):
    vectorizer = TfidfVectorizer()
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

def buscar_respuesta_tfidf(consulta, df, vectorizer, matriz_tfidf, umbral=0.20):
    
    consulta_clean = consulta.lower().strip()
    palabras_usuario = consulta_clean.split()

    # Filtro Social (Solo si es corto)
    if len(palabras_usuario) < 5: 
        for frase, respuesta in CHARLA_SOCIAL.items():
            if frase in consulta_clean:
                return respuesta

    # Filtro Emocional
    for emocion, respuesta in RESPUESTAS_EMOCIONALES.items():
        if emocion in consulta_clean:
            return respuesta

    # Búsqueda Médica
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
