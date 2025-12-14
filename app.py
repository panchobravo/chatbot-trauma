import streamlit as st
from chatbot_backend import (
    cargar_y_preparar_base, 
    inicializar_vectorizador, 
    responder_consulta, 
    registrar_pregunta_en_sheets,
    guardar_paciente_en_sheets,
    registrar_feedback,
    conectar_sheets # Importamos esto para el test
)
import time
import datetime

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

# ESTADO
if "usuario_registrado" not in st.session_state: st.session_state.usuario_registrado = False
if "mensajes" not in st.session_state: st.session_state.mensajes = []
if "ultimo_contexto" not in st.session_state: st.session_state.ultimo_contexto = ""
if "nombre_usuario" not in st.session_state: st.session_state.nombre_usuario = ""

# --- ESCENA A: REGISTRO ---
if not st.session_state.usuario_registrado:
    st.title("🏥 Bienvenido/a")
    st.markdown("Por favor, ingresa tus datos para acceder.")
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
                exito = guardar_paciente_en_sheets(nombre, apellidos, rut, telefono, email)
                if exito:
                    st.session_state.usuario_registrado = True
                    st.session_state.nombre_usuario = nombre
                    st.rerun()
                else:
                    st.error("Error al conectar con la base de datos de usuarios.")
            else:
                st.error("Datos incompletos.")

# --- ESCENA B: CHAT ---
else:
    # 1. SALUDO AUTOMÁTICO
    if not st.session_state.mensajes:
        saludo = f"¡Hola {st.session_state.nombre_usuario}! 👋 Soy el asistente virtual del equipo médico. Estoy aquí para acompañarte. ¿Tienes dudas sobre dolor, heridas o medicamentos?"
        st.session_state.mensajes.append({"rol": "assistant", "contenido": saludo})

    # 2. SIDEBAR CON DIAGNÓSTICO
    with st.sidebar:
        st.header(f"Paciente: {st.session_state.nombre_usuario}")
        
        st.divider()
        st.subheader("🔧 Zona Técnica")
        # BOTÓN PARA PROBAR CONEXIÓN MANUALMENTE
        if st.button("Test Conexión Sheets"):
            sh = conectar_sheets()
            if sh:
                st.success("✅ Conexión al Archivo: OK")
                try:
                    sh.worksheet("Hoja1").append_row(["TEST", "Hoja1 OK", datetime.datetime.now().strftime("%H:%M")])
                    st.write("✅ Escritura en Hoja1: OK")
                except Exception as e: st.error(f"❌ Fallo Hoja1: {e}")
                
                try:
                    sh.worksheet("Feedback").append_row(["TEST", "Feedback OK", datetime.datetime.now().strftime("%H:%M")])
                    st.write("✅ Escritura en Feedback: OK")
                except Exception as e: st.error(f"❌ Fallo Feedback: {e}")
            else:
                st.error("❌ Fallo Conexión General")

        if st.button("Cerrar Sesión"):
            st.session_state.usuario_registrado = False
            st.session_state.mensajes = []
            st.rerun()

    st.title("👨‍⚕️ Asistente Traumatología")
    
    # Renderizar mensajes
    for mensaje in st.session_state.mensajes:
        with st.chat_message(mensaje["rol"]):
            st.markdown(mensaje["contenido"])

    # Input Usuario
    prompt = st.chat_input("Escribe tu duda aquí...")

    if prompt:
        # A. GUARDAR PREGUNTA EN GOOGLE SHEETS (Siempre)
        registrar_pregunta_en_sheets(prompt)

        # B. MOSTRAR Y PROCESAR
        st.session_state.mensajes.append({"rol": "user", "contenido": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("🩺 *Consultando...*")
            time.sleep(0.5)
            
            respuesta, tags = responder_consulta(
                prompt, df, vectorizer, matriz_tfidf, 
                st.session_state.ultimo_contexto
            )
            
            if tags: st.session_state.ultimo_contexto = " ".join(tags)
            placeholder.markdown(respuesta)
            
            # C. FEEDBACK
            col_a, col_b = st.columns([1, 10])
            with col_a:
                if st.button("👍", key=f"like_{len(st.session_state.mensajes)}"):
                    registrar_feedback(prompt, respuesta, "POSITIVO")
                    st.toast("¡Gracias!", icon="✅")
            with col_b:
                if st.button("👎", key=f"dislike_{len(st.session_state.mensajes)}"):
                    registrar_feedback(prompt, respuesta, "NEGATIVO")
                    st.toast("Reportado.", icon="🚩")

        st.session_state.mensajes.append({"rol": "assistant", "contenido": respuesta})
