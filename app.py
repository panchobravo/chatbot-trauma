import streamlit as st
from chatbot_backend import (
    cargar_y_preparar_base, 
    inicializar_vectorizador, 
    responder_consulta, 
    registrar_pregunta_en_sheets,
    guardar_paciente_en_sheets,
    registrar_feedback
)
import time

# CONFIGURACIÓN DE PÁGINA
st.set_page_config(page_title="Asistente Traumatología", page_icon="🏥", layout="centered", initial_sidebar_state="collapsed")
st.markdown("""<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;} .stChatMessage {border-radius: 15px; padding: 10px;}</style>""", unsafe_allow_html=True)

# INICIALIZACIÓN DEL CEREBRO
@st.cache_resource
def iniciar_cerebro():
    df = cargar_y_preparar_base('knowledge_base.json')
    vec, mat = inicializar_vectorizador(df)
    return df, vec, mat

try:
    df, vectorizer, matriz_tfidf = iniciar_cerebro()
except Exception as e:
    st.error(f"Error cargando sistema: {e}")
    st.stop()

# ESTADO DE SESIÓN
if "usuario_registrado" not in st.session_state: st.session_state.usuario_registrado = False
if "mensajes" not in st.session_state: st.session_state.mensajes = []
if "ultimo_contexto" not in st.session_state: st.session_state.ultimo_contexto = ""
if "nombre_usuario" not in st.session_state: st.session_state.nombre_usuario = "" # Nuevo para personalizar el saludo

# --- ESCENA A: REGISTRO DE PACIENTE ---
if not st.session_state.usuario_registrado:
    st.title("🏥 Bienvenido/a")
    st.markdown("Por favor, ingresa tus datos para acceder al chat de soporte.")
    with st.form("registro"):
        col1, col2 = st.columns(2)
        nombre = col1.text_input("Nombre")
        apellidos = col2.text_input("Apellidos")
        rut = st.text_input("RUT")
        col3, col4 = st.columns(2)
        telefono = col3.text_input("Teléfono", max_chars=9)
        email = col4.text_input("Email")
        
        if st.form_submit_button("Ingresar"):
            if len(telefono) >= 8 and "@" in email:
                if guardar_paciente_en_sheets(nombre, apellidos, rut, telefono, email):
                    st.session_state.usuario_registrado = True
                    st.session_state.nombre_usuario = nombre # Guardamos el nombre
                    st.rerun()
            else:
                st.error("Por favor completa los datos correctamente.")

# --- ESCENA B: CHAT ASISTENTE ---
else:
    # MENSAJE DE BIENVENIDA AUTOMÁTICO (LA MAGIA NUEVA ✨)
    if not st.session_state.mensajes:
        saludo_inicial = f"¡Hola {st.session_state.nombre_usuario}! 👋 Soy el asistente virtual del equipo médico. Estoy aquí para acompañarte en tu recuperación. ¿Tienes alguna duda sobre tu dolor, la herida o los medicamentos? Pregúntame con confianza."
        st.session_state.mensajes.append({"rol": "assistant", "contenido": saludo_inicial})

    with st.sidebar:
        st.header(f"Hola, {st.session_state.nombre_usuario}")
        if st.button("Cerrar Sesión"):
            st.session_state.usuario_registrado = False
            st.session_state.mensajes = []
            st.rerun()

    st.title("👨‍⚕️ Asistente Traumatología")
    
    # Renderizar historial
    for i, mensaje in enumerate(st.session_state.mensajes):
        with st.chat_message(mensaje["rol"]):
            st.markdown(mensaje["contenido"])

    # Input del usuario
    prompt = st.chat_input("Escribe tu duda aquí...")

    if prompt:
        # 1. Mostrar mensaje usuario
        st.session_state.mensajes.append({"rol": "user", "contenido": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Procesar respuesta
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("🩺 *Consultando protocolos...*")
            time.sleep(0.6) # Pequeña pausa para que se sienta natural
            
            respuesta, tags = responder_consulta(
                prompt, df, vectorizer, matriz_tfidf, 
                st.session_state.ultimo_contexto
            )
            
            # Actualizar memoria de contexto
            if tags: st.session_state.ultimo_contexto = " ".join(tags)
            
            placeholder.markdown(respuesta)
            
            # Botones de Feedback
            col_a, col_b = st.columns([1, 10])
            with col_a:
                if st.button("👍", key=f"like_{len(st.session_state.mensajes)}"):
                    registrar_feedback(prompt, respuesta, "POSITIVO")
                    st.toast("¡Gracias! Nos ayuda a mejorar.", icon="🎓")
            with col_b:
                if st.button("👎", key=f"dislike_{len(st.session_state.mensajes)}"):
                    registrar_feedback(prompt, respuesta, "NEGATIVO")
                    st.toast("Revisaremos esta respuesta.", icon="📝")

        st.session_state.mensajes.append({"rol": "assistant", "contenido": respuesta})
