import streamlit as st
import psycopg2
import json
import os
from datetime import datetime
import pandas as pd

def get_db_connection():
    """Obtener conexión a la base de datos PostgreSQL"""
    try:
        conn = psycopg2.connect(os.environ['DATABASE_URL'])
        return conn
    except Exception as e:
        st.error(f"Error al conectar a la base de datos: {str(e)}")
        return None

def guardar_sesion_analisis(user_id, session_name, dataset_name, dataset_info, model_results):
    """Guardar sesión de análisis en la base de datos"""
    conn = get_db_connection()
    if conn is None:
        return False, "Error de conexión a la base de datos"
    
    try:
        cur = conn.cursor()
        
        # Convertir dataset_info y model_results a JSON
        dataset_info_json = json.dumps(dataset_info) if dataset_info else None
        model_results_json = json.dumps(model_results, default=str) if model_results else None
        
        cur.execute(
            """
            INSERT INTO analysis_sessions (user_id, session_name, dataset_name, dataset_info, model_results)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (user_id, session_name, dataset_name, dataset_info_json, model_results_json)
        )
        
        session_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        
        return True, session_id
        
    except Exception as e:
        if conn:
            conn.close()
        return False, f"Error al guardar sesión: {str(e)}"

def obtener_sesiones_usuario(user_id):
    """Obtener todas las sesiones de análisis de un usuario"""
    conn = get_db_connection()
    if conn is None:
        return []
    
    try:
        cur = conn.cursor()
        
        cur.execute(
            """
            SELECT id, session_name, dataset_name, created_at, dataset_info, model_results
            FROM analysis_sessions
            WHERE user_id = %s
            ORDER BY created_at DESC
            """,
            (user_id,)
        )
        
        sesiones = []
        for row in cur.fetchall():
            sesiones.append({
                'id': row[0],
                'session_name': row[1],
                'dataset_name': row[2],
                'created_at': row[3],
                'dataset_info': json.loads(row[4]) if row[4] else {},
                'model_results': json.loads(row[5]) if row[5] else {}
            })
        
        cur.close()
        conn.close()
        
        return sesiones
        
    except Exception as e:
        if conn:
            conn.close()
        st.error(f"Error al obtener sesiones: {str(e)}")
        return []

def cargar_sesion(session_id):
    """Cargar una sesión de análisis específica"""
    conn = get_db_connection()
    if conn is None:
        return None
    
    try:
        cur = conn.cursor()
        
        cur.execute(
            """
            SELECT session_name, dataset_name, created_at, dataset_info, model_results
            FROM analysis_sessions
            WHERE id = %s
            """,
            (session_id,)
        )
        
        row = cur.fetchone()
        cur.close()
        conn.close()
        
        if row:
            return {
                'session_name': row[0],
                'dataset_name': row[1],
                'created_at': row[2],
                'dataset_info': json.loads(row[3]) if row[3] else {},
                'model_results': json.loads(row[4]) if row[4] else {}
            }
        return None
        
    except Exception as e:
        if conn:
            conn.close()
        st.error(f"Error al cargar sesión: {str(e)}")
        return None

def eliminar_sesion(session_id, user_id):
    """Eliminar una sesión de análisis"""
    conn = get_db_connection()
    if conn is None:
        return False
    
    try:
        cur = conn.cursor()
        
        # Verificar que la sesión pertenece al usuario
        cur.execute(
            "DELETE FROM analysis_sessions WHERE id = %s AND user_id = %s",
            (session_id, user_id)
        )
        
        conn.commit()
        deleted = cur.rowcount > 0
        cur.close()
        conn.close()
        
        return deleted
        
    except Exception as e:
        if conn:
            conn.close()
        st.error(f"Error al eliminar sesión: {str(e)}")
        return False

def mostrar_historial_sesiones():
    """Mostrar interfaz de historial de sesiones"""
    st.header("📜 Historial de Análisis")
    
    user_id = st.session_state.get('user_id')
    if not user_id:
        st.error("❌ Usuario no autenticado")
        return
    
    # Obtener sesiones del usuario
    sesiones = obtener_sesiones_usuario(user_id)
    
    if not sesiones:
        st.info("📭 No tienes sesiones guardadas aún")
        st.info("💡 Las sesiones se guardarán automáticamente cuando ejecutes modelos")
        return
    
    st.success(f"✅ Se encontraron {len(sesiones)} sesiones guardadas")
    
    # Mostrar sesiones en una tabla
    st.subheader("📊 Tus Sesiones de Análisis")
    
    # Crear DataFrame para mostrar
    sesiones_df = pd.DataFrame([
        {
            'ID': s['id'],
            'Nombre': s['session_name'],
            'Dataset': s['dataset_name'],
            'Fecha': s['created_at'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(s['created_at'], datetime) else str(s['created_at']),
            'Modelos': ', '.join(s['model_results'].keys()) if s['model_results'] else 'Ninguno'
        }
        for s in sesiones
    ])
    
    st.dataframe(sesiones_df, use_container_width=True)
    
    # Seleccionar sesión para ver detalles
    st.subheader("🔍 Ver Detalles de Sesión")
    
    session_ids = [s['id'] for s in sesiones]
    session_names = [f"{s['session_name']} - {s['dataset_name']} ({s['created_at'].strftime('%Y-%m-%d %H:%M') if isinstance(s['created_at'], datetime) else str(s['created_at'])})" for s in sesiones]
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_index = st.selectbox(
            "Selecciona una sesión:",
            range(len(session_names)),
            format_func=lambda i: session_names[i]
        )
    
    with col2:
        if st.button("🗑️ Eliminar Sesión", type="secondary"):
            if eliminar_sesion(session_ids[selected_index], user_id):
                st.success("✅ Sesión eliminada exitosamente")
                st.rerun()
            else:
                st.error("❌ Error al eliminar la sesión")
    
    # Mostrar detalles de la sesión seleccionada
    sesion_seleccionada = sesiones[selected_index]
    
    with st.expander("📋 Detalles de la Sesión", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Nombre:** {sesion_seleccionada['session_name']}")
            st.write(f"**Dataset:** {sesion_seleccionada['dataset_name']}")
        
        with col2:
            st.write(f"**Fecha:** {sesion_seleccionada['created_at']}")
            st.write(f"**ID Sesión:** {sesion_seleccionada['id']}")
        
        # Información del dataset
        if sesion_seleccionada['dataset_info']:
            st.subheader("📊 Información del Dataset")
            dataset_info = sesion_seleccionada['dataset_info']
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Filas", dataset_info.get('rows', 'N/A'))
            with col2:
                st.metric("Columnas", dataset_info.get('columns', 'N/A'))
            with col3:
                st.metric("Valores Nulos", dataset_info.get('null_values', 'N/A'))
        
        # Resultados de modelos
        if sesion_seleccionada['model_results']:
            st.subheader("🤖 Resultados de Modelos")
            
            for tipo_modelo, resultados in sesion_seleccionada['model_results'].items():
                with st.expander(f"🔍 {tipo_modelo.replace('_', ' ').title()}"):
                    if tipo_modelo == 'regresion':
                        for modelo, metricas in resultados.items():
                            st.write(f"**{modelo}**")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("R²", f"{metricas.get('r2', 'N/A'):.4f}" if isinstance(metricas.get('r2'), (int, float)) else 'N/A')
                            with col2:
                                st.metric("RMSE", f"{metricas.get('rmse', 'N/A'):.4f}" if isinstance(metricas.get('rmse'), (int, float)) else 'N/A')
                            with col3:
                                st.metric("MAE", f"{metricas.get('mae', 'N/A'):.4f}" if isinstance(metricas.get('mae'), (int, float)) else 'N/A')
                    
                    elif tipo_modelo == 'logistica':
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Exactitud", f"{resultados.get('accuracy', 'N/A'):.4f}" if isinstance(resultados.get('accuracy'), (int, float)) else 'N/A')
                        with col2:
                            st.metric("Precisión", f"{resultados.get('precision', 'N/A'):.4f}" if isinstance(resultados.get('precision'), (int, float)) else 'N/A')
                        with col3:
                            st.metric("Recall", f"{resultados.get('recall', 'N/A'):.4f}" if isinstance(resultados.get('recall'), (int, float)) else 'N/A')
                        with col4:
                            st.metric("F1-Score", f"{resultados.get('f1_score', 'N/A'):.4f}" if isinstance(resultados.get('f1_score'), (int, float)) else 'N/A')
                    
                    elif 'arbol' in tipo_modelo:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Profundidad", resultados.get('depth', 'N/A'))
                        with col2:
                            st.metric("Número de Hojas", resultados.get('n_leaves', 'N/A'))
                        
                        if 'accuracy' in resultados:
                            st.metric("Exactitud", f"{resultados['accuracy']:.4f}")
                        elif 'r2' in resultados:
                            st.metric("R²", f"{resultados['r2']:.4f}")
    
    # Botón para cargar la sesión
    if st.button("📥 Cargar Esta Sesión", type="primary"):
        # Cargar los resultados en el estado de la sesión
        st.session_state.model_results = sesion_seleccionada['model_results']
        st.success("✅ Sesión cargada exitosamente")
        st.info("💡 Ahora puedes generar un reporte PDF con estos resultados")

def guardar_sesion_automatica():
    """Guardar automáticamente la sesión actual"""
    user_id = st.session_state.get('user_id')
    
    if not user_id:
        return
    
    # Verificar si hay resultados para guardar
    if not st.session_state.get('model_results') or not st.session_state.model_results:
        return
    
    # Generar nombre de sesión automático
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_name = f"Análisis_{timestamp}"
    
    # Información del dataset
    dataset_name = "Dataset sin nombre"
    dataset_info = {}
    
    if st.session_state.get('dataset') is not None:
        df = st.session_state.dataset
        dataset_name = f"Dataset_{timestamp}"
        dataset_info = {
            'rows': int(df.shape[0]),
            'columns': int(df.shape[1]),
            'null_values': int(df.isnull().sum().sum())
        }
    
    # Guardar sesión
    exito, resultado = guardar_sesion_analisis(
        user_id,
        session_name,
        dataset_name,
        dataset_info,
        st.session_state.model_results
    )
    
    if exito:
        st.success(f"✅ Sesión guardada automáticamente (ID: {resultado})")
    else:
        st.warning(f"⚠️ No se pudo guardar la sesión: {resultado}")
