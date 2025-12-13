# =======================================================================
# CHATBOT_BACKEND.PY - CONEXIÓN A GOOGLE SHEETS
# =======================================================================

import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import string
import datetime
import streamlit as st
import gspread # El cartero de Google

# -----------------------------------------------------------------------
# CONFIGURACIÓN DE SEGURIDAD
# -----------------------------------------------------------------------
PALABRAS_ALARMA = [
    "fiebre", "pus", "secreción", "infección", "sangrado abundante", 
    "hemorragia", "dolor insoportable", "desmayo", "no puedo respirar",
    "dedos azules", "no siento la pierna", "calor extremo"
]

MENSAJE_ALERTA = """
🚨 **ALERTA DE EMERGENCIA** 🚨
Su consulta contiene síntomas que requieren atención médica inmediata.
Este chatbot NO PUEDE diagnosticar emergencias.
Por favor, **LLAME INMEDIATAMENTE** a nuestra línea de emergencia.
"""

# -----------------------------------------------------------------------
# FUNCIONES DE NLP
# -----------------------------------------------------------------------
def preprocesar_texto(texto):
    texto = texto.lower()
    texto = ''.join([char for char in texto if char not in string.punctuation])
    stop_words_es = stopwords.words('spanish')
    palabras = texto.split()
    palabras_filtradas = [w for w in palabras if w not in stop_words_es]
    return ' '.join(palabras_filtradas)

def cargar_y_preparar_base(archivo_json):
    with open(archivo_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['intencion_preprocesada'] = df['intencion_clave'].apply(preprocesar_texto)
    return df

def inicializar_vectorizador(df):
    vectorizer = TfidfVectorizer()
    matriz_tfidf = vectorizer.fit_transform(df['intencion_preprocesada'])
    return vectorizer, matriz_tfidf

# -----------------------------------------------------------------------
# 🔌 CONEXIÓN A GOOGLE SHEETS (NUEVO)
# -----------------------------------------------------------------------
def registrar_pregunta_en_sheets(consulta):
    """Conecta con Google Sheets y guarda la pregunta sin respuesta"""
    try:
        # 1. Recuperar la llave desde los Secretos de Streamlit
        if "google_credentials" in st.secrets:
            creds_dict = json.loads(st.secrets["google_credentials"])
            
            # 2. Autenticar con Google
            gc = gspread.service_account_from_dict(creds_dict)
            
            # 3. Abrir la hoja (Asegúrate de que se llame EXACTAMENTE así)
            sh = gc.open("Cerebro_Bot") 
            worksheet = sh.sheet1
            
            # 4. Escribir (Append)
            ahora = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            worksheet.append_row([ahora, consulta])
        else:
            print("⚠️ No se encontraron credenciales en st.secrets")
            
    except Exception as e:
        # Imprimimos el error en la consola de Streamlit para depurar si falla
        print(f"❌ Error al guardar en Sheets: {e}")

# -----------------------------------------------------------------------
# LÓGICA PRINCIPAL
# -----------------------------------------------------------------------
def buscar_respuesta_tfidf(consulta, df, vectorizer, matriz_tfidf, umbral=0.4):
    consulta_preprocesada = preprocesar_texto(consulta)
    
    if not consulta_preprocesada:
        return "Por favor, formule una pregunta más clara."

    consulta_vector = vectorizer.transform([consulta_preprocesada])
    similitudes = cosine_similarity(consulta_vector, matriz_tfidf)
    mejor_sim_score = similitudes.max()
    mejor_sim_index = similitudes.argmax()
    
    if mejor_sim_score > umbral:
        return df.iloc[mejor_sim_index]['respuesta_validada']
    else:
        # --- AQUÍ GUARDAMOS EN LA NUBE ---
        def registrar_pregunta_en_sheets(consulta):
    """Conecta con Google Sheets y guarda la pregunta sin respuesta"""
    try:
        # 1. Verificamos si existen las llaves
        if "google_credentials" in st.secrets:
            creds_dict = json.loads(st.secrets["google_credentials"])
            
            # 2. Autenticar
            gc = gspread.service_account_from_dict(creds_dict)
            
            # 3. Abrir hoja
            sh = gc.open("Cerebro_Bot") 
            worksheet = sh.sheet1
            
            # 4. Escribir
            ahora = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            worksheet.append_row([ahora, consulta])
            
            # ÉXITO: Mostramos un aviso pequeño de que funcionó
            st.toast("✅ Pregunta guardada en Google Sheets", icon="📝")
            
        else:
            # FALLO 1: No hay secretos
            st.error("⚠️ ERROR: No encontré 'google_credentials' en los Secrets de Streamlit.")
            
    except Exception as e:
        # FALLO 2: Error técnico (Aquí saldrá el culpable)
        st.error(f"❌ ERROR DE CONEXIÓN: {e}")
        return "Lo siento, aún no tengo esa información específica validada. **He guardado tu pregunta** para que el Dr. la revise y me enseñe pronto la respuesta."

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
