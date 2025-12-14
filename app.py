import streamlit as st
from chatbot_backend import (
    cargar_y_preparar_base, 
    inicializar_vectorizador, 
    responder_consulta, 
    registrar_pregunta_en_sheets,
    guardar_paciente_en_sheets
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

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stChatMessage {border-radius: 15px; padding: 10px;}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. INICIALIZACIÓN
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
# 3. LÓGICA DE REGISTRO
# -----------------------------------------------------------------------------
if "usuario_registrado" not in st.session_state:
    st.session_state.usuario_registrado = False

# ==========================================
# ESCENA A: EL FORMULARIO DE INGRESO
# ==========================================
if not st.session_state.usuario_registrado:
    st.title("🏥 Bienvenido/a")
    st.markdown("Para ingresar a la consulta virtual del Dr., por favor complete sus datos.")
    st.info("🔒 Sus datos son confidenciales.")

    with st.form("registro_paciente"):
        # Fila 1: Nombre y Apellidos
        col1, col2 = st.columns(2)
        nombre = col1.text_input("Nombre")
        apellidos = col2.text_input("Apellidos")
        
        # Fila 2: RUT
        rut = st.text_input("RUT (ej: 12.345.678-9)")

        # Fila 3: Contacto
        col3, col4 = st.columns(2)
        telefono = col3.text_input("Teléfono (9 dígitos)", placeholder="987654321", max_chars=9)
        email = col4.text_input("Email")
        
        submit_btn = st.form_submit_button("Ingresar")

        if submit_btn:
            errores = []
            
            # --- VALIDACIONES ESTRICTAS ---
            
            # 1. Campos vacíos
            if not nombre or not apellidos or not rut or not telefono or not email:
                errores.append("⚠️ Debe completar TODOS los campos.")
            
            # 2. Validación de Teléfono (Exactamente 9 dígitos numéricos)
            if len(telefono) != 9:
                errores.append("⚠️ El teléfono debe tener exactamente 9 dígitos.")
            elif not telefono.isdigit():
                errores.append("⚠️ El teléfono solo debe contener números.")
            
            # 3. Validación de Email simple
            if "@" not in email or "." not in email:
                errores.append("⚠️ El correo electrónico no parece válido.")

            # --- RESULTADO ---
            if errores:
                for err in errores:
                    st.error(err)
            else:
                with st.spinner("Verificando datos..."):
                    # Pasamos nombre y apellido por separado
                    guardado = guardar_paciente_en_sheets(nombre, apellidos, rut, telefono, email)
                    
                if guardado:
                    st.session_state.usuario_registrado = True
                    st.success("✅ Registro exitoso.")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Error de conexión. Intente nuevamente.")

# ==========================================
# ESCENA B: EL CHAT
# ==========================================
else:
    with st.sidebar:
        st.header("🏥 Consulta Virtual")
        # Aquí podrías poner el apellido del Dr. real
        st.write("**Traumatología y Ortopedia**")
        st.info("Si tiene síntomas graves (fiebre, dolor extremo), acuda a Urgencias.")
        if st.button("Salir"):
            st.session_state.usuario_registrado = False
            st.rerun()

    st.title("👨‍⚕️ Asistente Virtual")
    st.markdown("Hola, soy el asistente del Dr. ¿En qué te puedo ayudar hoy?")

    if "mensajes" not in st.session_state:
        st.session_state.mensajes = []

    for mensaje in st.session_state.mensajes:
        with st.chat_message(mensaje["rol"]):
            st.markdown(mensaje["contenido"])

    prompt = st.chat_input("Escribe tu consulta aquí...")

    if prompt:
        st.session_state.mensajes.append({"rol": "user", "contenido": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("🩺 *Escribiendo...*")
            time.sleep(0.5)
            respuesta = responder_consulta(prompt, df, vectorizer, matriz_tfidf)
            placeholder.markdown(respuesta)
        
        st.session_state.mensajes.append({"rol": "assistant", "contenido": respuesta})
