import streamlit as st
from chatbot_backend import (
    cargar_y_preparar_base, 
    inicializar_vectorizador, 
    responder_consulta, 
    registrar_pregunta_en_sheets
)
import time

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA (ESTÉTICA PRO)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Asistente Traumatología",  # Título en la pestaña del navegador
    page_icon="🏥",                       # Ícono en la pestaña
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# 2. ESTILO CSS (OCULTAR MARCAS DE STREAMLIT)
# -----------------------------------------------------------------------------
# Esto oculta el menú de hamburguesa, el pie de página y ajusta colores
st.markdown("""
<style>
    /* Ocultar menú de hamburguesa superior derecho */
    #MainMenu {visibility: hidden;}
    /* Ocultar pie de página "Made with Streamlit" */
    footer {visibility: hidden;}
    /* Ocultar barra de decoración superior */
    header {visibility: hidden;}
    
    /* Estilo del chat */
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. BARRA LATERAL (INFORMACIÓN FIJA)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("🏥 Consulta Virtual")
    st.markdown("---")
    st.write("**Dr. [Equipo de Traumatologos de Tobillo y Pie]**")
    st.write("Traumatología y Ortopedia")
    st.markdown("---")
    st.info(
        "ℹ️ **Nota:** Este asistente responde dudas frecuentes post-operatorias. "
        "No reemplaza una consulta de urgencia."
    )
    st.error("🚨 **Emergencias:** Si tienes fiebre alta, dolor incontrolable o sangrado, acude a Urgencias inmediatamente.")

# -----------------------------------------------------------------------------
# 4. INICIALIZACIÓN DEL CEREBRO
# -----------------------------------------------------------------------------
@st.cache_resource
def iniciar_cerebro():
    df = cargar_y_preparar_base('knowledge_base.json')
    vec, mat = inicializar_vectorizador(df)
    return df, vec, mat

try:
    df, vectorizer, matriz_tfidf = iniciar_cerebro()
except Exception as e:
    st.error(f"Error cargando el cerebro: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 5. INTERFAZ DE CHAT (TIPO WHATSAPP)
# -----------------------------------------------------------------------------

# Título Principal
st.title("👨‍⚕️ Asistente Dr. [Equipo de Traumatologos de Tobillo y Pie]")
st.markdown("Hola, soy tu asistente virtual. ¿En qué puedo orientarte hoy sobre tu recuperación?")

# Historial de Chat
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

# Mostrar mensajes anteriores
for mensaje in st.session_state.mensajes:
    with st.chat_message(mensaje["rol"]):
        st.markdown(mensaje["contenido"])

# Input del usuario
prompt = st.chat_input("Escribe tu duda aquí...")

if prompt:
    # 1. Guardar y mostrar mensaje del usuario
    st.session_state.mensajes.append({"rol": "user", "contenido": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Pensando... (Efecto visual)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("🩺 *Analizando consulta...*")
        time.sleep(0.5) # Pequeña pausa para naturalidad

        # 3. Obtener respuesta
        respuesta = responder_consulta(prompt, df, vectorizer, matriz_tfidf)
        
        # 4. Mostrar respuesta
        placeholder.markdown(respuesta)
    
    # 5. Guardar en historial
    st.session_state.mensajes.append({"rol": "assistant", "contenido": respuesta})
