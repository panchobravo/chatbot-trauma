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

# CONFIGURACIÓN
st.set_page_config(page_title="Asistente Traumatología", page_icon="🏥", layout="centered", initial_sidebar_state="collapsed")
st.markdown("""<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;} .stChatMessage {border-radius: 15px; padding: 10px;}</style>""", unsafe_allow_html=True)

# INICIALIZACIÓN
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

# ESTADO DE SESIÓN (MEMORIA)
if "usuario_registrado" not in st.session_state: st.session_state.usuario_registrado = False
if "mensajes" not in st.session_state: st.session_state.mensajes = []
if "ultimo_contexto" not in st.session_state: st.session_state.ultimo_contexto = "" # <--- MEMORIA CONTEXTUAL

# --- ESCENA A: REGISTRO ---
if not st.session_state.usuario_registrado:
    st.title("🏥 Bienvenido/a")
    st.markdown("Ingresa tus datos para comenzar.")
    with st.form("registro"):
        col1, col2 = st.columns(2)
        nombre = col1.text_input("Nombre")
        apellidos = col2.text_input("Apellidos")
        rut = st.text_input("RUT")
        col3, col4 = st.columns(2)
        telefono = col3.text_input("Teléfono (9 dígitos)", max_chars=9)
        email = col4.text_input("Email")
        if st.form_submit_button("Ingresar"):
            if len(telefono) == 9 and "@" in email:
                if guardar_paciente_en_sheets(nombre, apellidos, rut, telefono, email):
                    st.session_state.usuario_registrado = True
                    st.rerun()
            else:
                st.error("Datos inválidos.")

# --- ESCENA B: CHAT INTELIGENTE ---
else:
    with st.sidebar:
        st.header("Dr. Virtual")
        if st.button("Cerrar Sesión"):
            st.session_state.usuario_registrado = False
            st.rerun()

    st.title("👨‍⚕️ Asistente Traumatología")
    
    # Historial
    for i, mensaje in enumerate(st.session_state.mensajes):
        with st.chat_message(mensaje["rol"]):
            st.markdown(mensaje["contenido"])

    # Input Usuario
    prompt = st.chat_input("Escribe aquí...")

    if prompt:
        # 1. Mostrar mensaje usuario
        st.session_state.mensajes.append({"rol": "user", "contenido": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Procesar con Contexto
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("🩺 *Pensando...*")
            time.sleep(0.5)
            
            # LLAMADA AL CEREBRO CON TEXTO + CONTEXTO ANTERIOR
            respuesta, tags = responder_consulta(
                prompt, df, vectorizer, matriz_tfidf, 
                st.session_state.ultimo_contexto
            )
            
            # ACTUALIZAR MEMORIA CONTEXTUAL (Si hay tags nuevos, los guardamos)
            if tags:
                st.session_state.ultimo_contexto = " ".join(tags)
            
            placeholder.markdown(respuesta)
            
            # BOTONES DE FEEDBACK (Exclusivo V8.0)
            col_a, col_b = st.columns([1, 10])
            with col_a:
                if st.button("👍", key=f"like_{len(st.session_state.mensajes)}"):
                    registrar_feedback(prompt, respuesta, "POSITIVO")
                    st.toast("¡Gracias por enseñarme!", icon="🎓")
            with col_b:
                if st.button("👎", key=f"dislike_{len(st.session_state.mensajes)}"):
                    registrar_feedback(prompt, respuesta, "NEGATIVO")
                    st.toast("Anotado para mejorar.", icon="📝")

        st.session_state.mensajes.append({"rol": "assistant", "contenido": respuesta})
