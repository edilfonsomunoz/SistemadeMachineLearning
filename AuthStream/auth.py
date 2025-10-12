import streamlit as st
import bcrypt
import re
import os
import psycopg2
from datetime import datetime

def get_db_connection():
    """Obtener conexión a la base de datos PostgreSQL"""
    try:
        conn = psycopg2.connect(os.environ['DATABASE_URL'])
        return conn
    except Exception as e:
        st.error(f"Error al conectar a la base de datos: {str(e)}")
        return None

def hash_password(password):
    """Hashear contraseña usando bcrypt con salt"""
    salt = bcrypt.gensalt()
    password_hash = bcrypt.hashpw(password.encode('utf-8'), salt)
    return password_hash.decode('utf-8')

def verify_password(password, password_hash):
    """Verificar contraseña contra hash bcrypt"""
    return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))

def validar_email(email):
    """Validar formato de email"""
    patron = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(patron, email) is not None

def validar_codigo_estudiante(codigo):
    """Validar que el código de estudiante sea de 6 dígitos"""
    return codigo.isdigit() and len(codigo) == 6

def registrar_usuario(email, nombre, codigo_estudiante, password):
    """Registrar un nuevo usuario en la base de datos"""
    conn = get_db_connection()
    if conn is None:
        return False, "Error de conexión a la base de datos"
    
    try:
        cur = conn.cursor()
        
        # Verificar si el usuario ya existe
        cur.execute("SELECT email FROM users WHERE email = %s", (email,))
        if cur.fetchone():
            cur.close()
            conn.close()
            return False, "Ya existe un usuario registrado con este correo electrónico"
        
        # Verificar si el código de estudiante ya existe
        cur.execute("SELECT codigo_estudiante FROM users WHERE codigo_estudiante = %s", (codigo_estudiante,))
        if cur.fetchone():
            cur.close()
            conn.close()
            return False, "Ya existe un usuario registrado con este código de estudiante"
        
        # Insertar nuevo usuario
        password_hash = hash_password(password)
        cur.execute(
            """
            INSERT INTO users (email, nombre, codigo_estudiante, password_hash)
            VALUES (%s, %s, %s, %s)
            """,
            (email, nombre, codigo_estudiante, password_hash)
        )
        
        conn.commit()
        cur.close()
        conn.close()
        
        return True, "Usuario registrado exitosamente"
        
    except Exception as e:
        if conn:
            conn.close()
        return False, f"Error al registrar usuario: {str(e)}"

def autenticar_usuario(email, password):
    """Autenticar usuario con email y contraseña"""
    conn = get_db_connection()
    if conn is None:
        return None
    
    try:
        cur = conn.cursor()
        
        # Obtener usuario por email
        cur.execute(
            """
            SELECT id, email, nombre, codigo_estudiante, password_hash
            FROM users
            WHERE email = %s
            """,
            (email,)
        )
        
        usuario = cur.fetchone()
        cur.close()
        conn.close()
        
        if usuario and verify_password(password, usuario[4]):
            return {
                'id': usuario[0],
                'email': usuario[1],
                'nombre': usuario[2],
                'codigo_estudiante': usuario[3]
            }
        return None
        
    except Exception as e:
        if conn:
            conn.close()
        st.error(f"Error al autenticar usuario: {str(e)}")
        return None

def mostrar_registro():
    """Mostrar formulario de registro centrado"""
    # Crear columnas para centrar el contenido
    col_left, col_center, col_right = st.columns([1, 2, 1])
    
    with col_center:
        # Contenedor con estilo de cuadro
        with st.container():
            st.markdown("<br>", unsafe_allow_html=True)
            st.header("📝 Registro de Usuario")
            
            with st.form("registro_form"):
                email = st.text_input("📧 Correo Electrónico", placeholder="usuario@ejemplo.com")
                nombre = st.text_input("👤 Nombre Completo", placeholder="Juan Pérez")
                codigo_estudiante = st.text_input("🎓 Código de Estudiante", placeholder="123456", max_chars=6)
                password = st.text_input("🔒 Contraseña", type="password", placeholder="Ingresa tu contraseña")
                
                submit_button = st.form_submit_button("✅ Registrarse", type="primary", use_container_width=True)
                
                if submit_button:
                    # Validaciones
                    errores = []
                    
                    if not email:
                        errores.append("El correo electrónico es obligatorio")
                    elif not validar_email(email):
                        errores.append("El formato del correo electrónico no es válido")
                    
                    if not nombre:
                        errores.append("El nombre es obligatorio")
                    elif len(nombre.strip()) < 2:
                        errores.append("El nombre debe tener al menos 2 caracteres")
                    
                    if not codigo_estudiante:
                        errores.append("El código de estudiante es obligatorio")
                    elif not validar_codigo_estudiante(codigo_estudiante):
                        errores.append("El código de estudiante debe ser de exactamente 6 dígitos")
                    
                    if not password:
                        errores.append("La contraseña es obligatoria")
                    elif len(password) < 6:
                        errores.append("La contraseña debe tener al menos 6 caracteres")
                    
                    if errores:
                        for error in errores:
                            st.error(f"❌ {error}")
                    else:
                        # Registrar usuario
                        exito, mensaje = registrar_usuario(email, nombre.strip(), codigo_estudiante, password)
                        
                        if exito:
                            st.success(f"✅ {mensaje}")
                            st.info("💡 Ahora puedes iniciar sesión con tus credenciales")
                        else:
                            st.error(f"❌ {mensaje}")

def mostrar_login():
    """Mostrar formulario de login centrado"""
    # Crear columnas para centrar el contenido
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Contenedor con estilo de cuadro
        with st.container():
            st.markdown("<br>", unsafe_allow_html=True)
            st.header("🔐 Iniciar Sesión")
            
            with st.form("login_form"):
                email = st.text_input("📧 Correo Electrónico", placeholder="usuario@ejemplo.com")
                password = st.text_input("🔒 Contraseña", type="password", placeholder="Ingresa tu contraseña")
                
                submit_button = st.form_submit_button("🚀 Iniciar Sesión", type="primary", use_container_width=True)
                
                if submit_button:
                    if not email or not password:
                        st.error("❌ Por favor, completa todos los campos")
                    else:
                        usuario = autenticar_usuario(email, password)
                        
                        if usuario:
                            # Login exitoso
                            st.session_state.authenticated = True
                            st.session_state.user_id = usuario['id']
                            st.session_state.user_email = usuario['email']
                            st.session_state.user_name = usuario['nombre']
                            st.session_state.codigo_estudiante = usuario['codigo_estudiante']
                            st.success("✅ Inicio de sesión exitoso")
                            st.rerun()
                        else:
                            st.error("❌ Credenciales incorrectas")

def verificar_sesion():
    """Verificar si el usuario está autenticado"""
    return st.session_state.get('authenticated', False)

def obtener_usuario_actual():
    """Obtener información del usuario actual"""
    if verificar_sesion():
        return {
            'id': st.session_state.get('user_id'),
            'email': st.session_state.user_email,
            'nombre': st.session_state.user_name,
            'codigo_estudiante': st.session_state.get('codigo_estudiante')
        }
    return None
