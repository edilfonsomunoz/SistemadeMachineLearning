import streamlit as st
import pandas as pd
import numpy as np
from auth import mostrar_login, mostrar_registro, verificar_sesion
from preprocesamiento import mostrar_carga_datos, procesar_datos
from modelos import mostrar_modelos_regresion, mostrar_modelos_logisticos, mostrar_arboles_decision
from reportes import generar_reporte_pdf
from session_history import mostrar_historial_sesiones, guardar_sesion_automatica
from comparacion_modelos import mostrar_comparacion_modelos
from hyperparameter_tuning import mostrar_hyperparameter_tuning
from exportar_datos import mostrar_exportar_datos
import os

# Configuración de la página
st.set_page_config(
    page_title="ML Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Función principal de la aplicación"""
    
    # Inicializar variables de sesión
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'user_email' not in st.session_state:
        st.session_state.user_email = None
    if 'user_name' not in st.session_state:
        st.session_state.user_name = None
    if 'dataset' not in st.session_state:
        st.session_state.dataset = None
    if 'processed_dataset' not in st.session_state:
        st.session_state.processed_dataset = None
    if 'model_results' not in st.session_state:
        st.session_state.model_results = {}

    # Verificar autenticación
    if not st.session_state.authenticated:
        st.title("🔐 Sistema de Autenticación")
        
        # Tabs para login y registro
        tab1, tab2 = st.tabs(["Iniciar Sesión", "Registrarse"])
        
        with tab1:
            mostrar_login()
        
        with tab2:
            mostrar_registro()
    
    else:
        # Dashboard principal
        st.title("📊 Dashboard de Machine Learning")
        st.success(f"¡Bienvenido/a, {st.session_state.user_name}!")
        
        # Sidebar para navegación
        with st.sidebar:
            st.header("🧭 Navegación")
            
            # Botón de cerrar sesión
            if st.button("🚪 Cerrar Sesión", type="secondary"):
                st.session_state.authenticated = False
                st.session_state.user_id = None
                st.session_state.user_email = None
                st.session_state.user_name = None
                st.session_state.dataset = None
                st.session_state.processed_dataset = None
                st.session_state.model_results = {}
                st.rerun()
            
            st.divider()
            
            # Botón para guardar sesión (si hay resultados)
            if st.session_state.model_results:
                if st.button("💾 Guardar Sesión Actual", type="primary"):
                    guardar_sesion_automatica()
            
            st.divider()
            
            # Menú de opciones
            opcion = st.selectbox(
                "Selecciona una opción:",
                [
                    "🔹 Carga y Procesamiento de Datos",
                    "🔹 Modelos de Regresión", 
                    "🔹 Modelos Logísticos",
                    "🔹 Árboles de Decisión",
                    "🔹 Ajuste de Hiperparámetros",
                    "🔹 Comparación de Modelos",
                    "🔹 Exportar Datos y Resultados",
                    "🔹 Historial de Sesiones",
                    "🔹 Generar Reporte PDF"
                ]
            )
        
        # Contenido principal según la opción seleccionada
        if opcion == "🔹 Carga y Procesamiento de Datos":
            mostrar_carga_datos()
        
        elif opcion == "🔹 Modelos de Regresión":
            if st.session_state.processed_dataset is not None:
                mostrar_modelos_regresion(st.session_state.processed_dataset)
            else:
                st.warning("⚠️ Primero debes cargar y procesar un dataset.")
                st.info("💡 Ve a la opción 'Carga y Procesamiento de Datos' para comenzar.")
        
        elif opcion == "🔹 Modelos Logísticos":
            if st.session_state.processed_dataset is not None:
                mostrar_modelos_logisticos(st.session_state.processed_dataset)
            else:
                st.warning("⚠️ Primero debes cargar y procesar un dataset.")
                st.info("💡 Ve a la opción 'Carga y Procesamiento de Datos' para comenzar.")
        
        elif opcion == "🔹 Árboles de Decisión":
            if st.session_state.processed_dataset is not None:
                mostrar_arboles_decision(st.session_state.processed_dataset)
            else:
                st.warning("⚠️ Primero debes cargar y procesar un dataset.")
                st.info("💡 Ve a la opción 'Carga y Procesamiento de Datos' para comenzar.")
        
        elif opcion == "🔹 Ajuste de Hiperparámetros":
            mostrar_hyperparameter_tuning()
        
        elif opcion == "🔹 Comparación de Modelos":
            mostrar_comparacion_modelos()
        
        elif opcion == "🔹 Exportar Datos y Resultados":
            mostrar_exportar_datos()
        
        elif opcion == "🔹 Historial de Sesiones":
            mostrar_historial_sesiones()
        
        elif opcion == "🔹 Generar Reporte PDF":
            if st.session_state.model_results:
                generar_reporte_pdf(st.session_state.model_results)
            else:
                st.warning("⚠️ No hay resultados de modelos para generar el reporte.")
                st.info("💡 Ejecuta algunos modelos primero para generar el reporte.")

if __name__ == "__main__":
    main()
