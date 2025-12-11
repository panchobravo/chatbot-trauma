# =======================================================================
# APP.PY - INTERFAZ DE USUARIO (FRONT-END STREAMLIT)
# =======================================================================

import streamlit as st
import nltk
nltk.download('stopwords')
# Importamos las funciones necesarias de nuestro archivo de lógica central
from chatbot_backend import (
    cargar_y_preparar_base, 
    inicializar_vectorizador, 
    responder_consulta
)

# -----------------------------------------------------------------------
# 1. INICIALIZACIÓN Y CARGA DE MODELO (CACHÉ)
# -----------------------------------------------------------------------

@st.cache_resource
def inicializar_chatbot():
    """
    Carga la Base de Conocimiento y entrena el modelo TF-IDF. 
    @st.cache_resource asegura que esto solo se ejecute una vez al inicio, 
    ahorrando tiempo y recursos (clave para la solución low cost).
    """
    try:
        # 1. Cargar y preprocesar los datos desde el JSON
        base_datos = cargar_y_preparar_base('knowledge_base.json')
        
        # 2. Inicializar el vectorizador TF-IDF con los datos cargados
        tfidf_vectorizer, tfidf_matriz = inicializar_vectorizador(base_datos)
        
        return base_datos, tfidf_vectorizer, tfidf_matriz
    
    except FileNotFoundError:
        st.error("Error: Archivo 'knowledge_base.json' no encontrado. Asegúrese de que esté en el directorio correcto.")
        return None, None, None
    except Exception as e:
        st.error(f"Error al inicializar el chatbot: {e}")
        return None, None, None


# Llamada a la función de inicialización
df_base, vectorizer, matriz_tfidf = inicializar_chatbot()

# -----------------------------------------------------------------------
# 2. CONFIGURACIÓN DEL DISEÑO DE LA APLICACIÓN
# -----------------------------------------------------------------------

st.set_page_config(page_title="Asistente de Traumatología", layout="wide")
st.title("👨‍⚕️ Asistente Digital de Traumatología")
st.markdown("---")
st.warning("⚠️ **Importante:** Este prototipo proporciona información validada por su especialista. **NO REEMPLAZA UNA CONSULTA MÉDICA**. Pruebe la seguridad escribiendo 'fiebre'.")

# -----------------------------------------------------------------------
# 3. LÓGICA DEL CHAT (INTERACCIÓN)
# -----------------------------------------------------------------------

if df_base is not None:
    # 3.1. Inicializar el Historial de Chat
    # Usamos st.session_state para guardar la conversación entre interacciones
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Mensaje de bienvenida inicial del asistente
        st.session_state.messages.append(
            {"role": "assistant", "content": "¿Hola! Soy tu asistente. ¿Sobre qué quieres consultar hoy? (ej. Cuidado de herida, cuándo caminar)"}
        )

    # 3.2. Mostrar Mensajes Previos en Pantalla
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 3.3. Manejar la Entrada del Usuario
    if prompt := st.chat_input("Escribe tu pregunta aquí..."):
        
        # 1. Agregar la consulta del usuario al historial y mostrarla
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Llamar a la función de respuesta del backend (que incluye el Guardrail y TF-IDF)
        respuesta = responder_consulta(prompt, df_base, vectorizer, matriz_tfidf)
        
        # 3. Mostrar respuesta del asistente en pantalla y guardarla en el historial
        with st.chat_message("assistant"):
            st.markdown(respuesta)
        
        st.session_state.messages.append({"role": "assistant", "content": respuesta})
