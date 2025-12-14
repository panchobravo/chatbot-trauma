import streamlit as st
from chatbot_backend import (
    cargar_y_preparar_base, 
    inicializar_vectorizador, 
    responder_consulta, 
    registrar_pregunta_en_sheets,
    guardar_paciente_en_sheets # <--- IMPORTANTE: Importamos la nueva función
)
import time

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Asistente Traumatología",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Estilos CSS
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stChatMessage {border-radius: 15px; padding: 10px;}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. INICIALIZACIÓN (CEREBRO)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# 3. LÓGICA DE LOGIN / REGISTRO
# -----------------------------------------------------------------------------

# Verificamos si el usuario ya ingresó sus datos en esta sesión
if "usuario_registrado" not in st.session_state:
    st.session_state.usuario_registrado = False

# ==========================================
# ESCENA A: EL FORMULARIO DE INGRESO
# ==========================================
if not st.session_state.usuario_registrado:
    st.title("🏥 Bienvenido/a")
    st.markdown("Para brindarte una mejor atención y que el Dr. pueda contactarte si es necesario, por favor ingresa tus datos.")
    st.info("🔒 Tus datos son confidenciales y de uso exclusivo del equipo médico.")

    with st.form("registro_paciente"):
        col1, col2 = st.columns(2)
        nombre = col1.text_input("Nombre Completo")
        rut = col2.text_input("RUT (ej: 12.345.678-9)")
        
        telefono = st.text_input("Teléfono Móvil (9 dígitos)", placeholder="Ej: 987654321", max_chars=9)
        email = st.text_input("Correo Electrónico")
        
        submit_btn = st.form_submit_button("Ingresar a la Consulta")

        if submit_btn:
            # --- VALIDACIONES ---
            errores = []
            
            # 1. Validar campos vacíos
            if not nombre or not rut or not telefono or not email:
                errores.append("⚠️ Por favor completa todos los campos.")
            
            # 2. Validar teléfono (Solo números y largo 9)
            if len(telefono) != 9 or not telefono.isdigit():
                errores.append("⚠️ El teléfono debe tener exactamente 9 dígitos numéricos (sin +56).")
            
            # 3. Validar Email básico
            if "@" not in email or "." not in email:
                errores.append("⚠️ Ingresa un correo electrónico válido.")

            if errores:
                for err in errores:
                    st.error(err)
            else:
                # --- SI TODO ESTÁ BIEN ---
                with st.spinner("Registrando sus datos..."):
                    guardado = guardar_paciente_en_sheets(nombre, rut, telefono, email)
                    
                if guardado:
                    st.session_state.usuario_registrado = True
                    st.success("¡Datos registrados correctamente!")
                    time.sleep(1) # Pequeña pausa para que lea
                    st.rerun() # Recargamos la página para mostrar el chat
                else:
                    st.error("Hubo un problema de conexión. Intente nuevamente.")

# ==========================================
# ESCENA B: EL CHAT (Solo se ve si está registrado)
# ==========================================
else:
    # BARRA LATERAL (Solo visible en el chat)
    with st.sidebar:
        st.header("🏥 Consulta Virtual")
        st.write("**Dr. [TU APELLIDO]**")
        st.info("Recuerda: Si tienes síntomas graves, acude a Urgencias.")
        if st.button("Cerrar Sesión"):
            st.session_state.usuario_registrado = False
            st.rerun()

    # CHAT UI
    st.title("👨‍⚕️ Asistente Dr. [TU APELLIDO]")
    st.markdown("Hola, soy el asistente del Dr. ¿En qué te puedo ayudar hoy?")

    if "mensajes" not in st.session_state:
        st.session_state.mensajes = []

    for mensaje in st.session_state.mensajes:
        with st.chat_message(mensaje["rol"]):
            st.markdown(mensaje["contenido"])

    prompt = st.chat_input("Escribe tu duda aquí...")

    if prompt:
        st.session_state.mensajes.append({"rol": "user", "contenido": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("🩺 *Analizando...*")
            time.sleep(0.5)
            respuesta = responder_consulta(prompt, df, vectorizer, matriz_tfidf)
            placeholder.markdown(respuesta)
        
        st.session_state.mensajes.append({"rol": "assistant", "contenido": respuesta})
