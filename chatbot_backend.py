# =======================================================================
# CHATBOT_BACKEND.PY - LÓGICA CON SISTEMA DE APRENDIZAJE (LOGGING)
# =======================================================================

import json
import pandas as pd
# Herramientas de NLP/IA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import string
# NUEVO: Librerías para el sistema de aprendizaje (Logging)
import csv
import datetime
import os

# -----------------------------------------------------------------------
# 2. CONFIGURACIÓN DE SEGURIDAD (GUARDRAILS)
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
Por favor, **LLAME INMEDIATAMENTE** a nuestra línea de emergencia: **+56 9 XXXXXXXX**
"""

# -----------------------------------------------------------------------
# 3. FUNCIONES DE NLP
# -----------------------------------------------------------------------
def preprocesar_texto(texto):
    """Limpia y normaliza el texto."""
    texto = texto.lower()
    texto = ''.join([char for char in texto if char not in string.punctuation])
    stop_words_es = stopwords.words('spanish')
    palabras = texto.split()
    palabras_filtradas = [w for w in palabras if w not in stop_words_es]
    return ' '.join(palabras_filtradas)

def cargar_y_preparar_base(archivo_json):
    """Carga el JSON y preprocesa."""
    with open(archivo_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['intencion_preprocesada'] = df['intencion_clave'].apply(preprocesar_texto)
    return df

# -----------------------------------------------------------------------
# 4. FUNCIONES DE IA Y APRENDIZAJE
# -----------------------------------------------------------------------
def inicializar_vectorizador(df):
    vectorizer = TfidfVectorizer()
    matriz_tfidf = vectorizer.fit_transform(df['intencion_preprocesada'])
    return vectorizer, matriz_tfidf

def registrar_pregunta_sin_respuesta(consulta):
    """
    Guarda la pregunta no entendida en un archivo CSV.
    """
    nombre_archivo = 'preguntas_sin_respuesta.csv'
    existe = os.path.isfile(nombre_archivo)
    
    try:
        with open(nombre_archivo, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Si es archivo nuevo, escribir encabezados
            if not existe:
                writer.writerow(['Fecha_Hora', 'Pregunta_Paciente'])
            
            # Escribir el registro
            ahora = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([ahora, consulta])
    except Exception as e:
        print(f"Error al registrar log: {e}")

def buscar_respuesta_tfidf(consulta, df, vectorizer, matriz_tfidf, umbral=0.4):
    """
    Busca respuesta. Si no supera el umbral de confianza, REGISTRA la duda.
    """
    consulta_preprocesada = preprocesar_texto(consulta)
    
    if not consulta_preprocesada:
        return "Por favor, formule una pregunta más clara."

    # 1. Vectorizar y buscar
    consulta_vector = vectorizer.transform([consulta_preprocesada])
    similitudes = cosine_similarity(consulta_vector, matriz_tfidf)
    mejor_sim_score = similitudes.max()
    mejor_sim_index = similitudes.argmax()
    
    # 2. Decisión Inteligente
    if mejor_sim_score > umbral:
        return df.iloc[mejor_sim_index]['respuesta_validada']
    else:
        # --- APRENDIZAJE ACTIVADO ---
        registrar_pregunta_sin_respuesta(consulta)
        return "Lo siento, aún no he aprendido la respuesta específica para eso. **He anotado tu pregunta** para que el equipo médico la revise y actualice mi base de conocimiento pronto. Si es urgente, llama a la clínica."

# -----------------------------------------------------------------------
# 5. FUNCIÓN CENTRAL
# -----------------------------------------------------------------------
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
