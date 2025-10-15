# ...existing code...
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
    
    # Inicializar variables de sesión (forma segura)
    st.session_state.setdefault('authenticated', False)
    st.session_state.setdefault('user_id', None)
    st.session_state.setdefault('user_email', None)
    st.session_state.setdefault('user_name', None)
    st.session_state.setdefault('dataset', None)
    st.session_state.setdefault('processed_dataset', None)
    st.session_state.setdefault('model_results', {})

    # Verificar autenticación
    if not st.session_state['authenticated']:
        st.title("🔐 Sistema de Autenticación")
        
        # Tabs para login y registro
        tab1, tab2 = st.tabs(["Iniciar Sesión", "Registrarse"])
        
        with tab1:
            try:
                mostrar_login()
            except Exception as e:
                st.error("Error en el módulo de login. Revisa los logs.")
                st.exception(e)
        
        with tab2:
            try:
                mostrar_registro()
            except Exception as e:
                st.error("Error en el módulo de registro. Revisa los logs.")
                st.exception(e)
    
    else:
        # Dashboard principal
        st.title("📊 Dashboard de Machine Learning")
        st.success(f"¡Bienvenido/a, {st.session_state.get('user_name','Usuario')}!")
        
        # Sidebar para navegación
        with st.sidebar:
            st.header("🧭 Navegación")
            
            # Botón de cerrar sesión (sin parámetro `type` inválido)
            if st.button("🚪 Cerrar Sesión"):
                st.session_state['authenticated'] = False
                st.session_state['user_id'] = None
                st.session_state['user_email'] = None
                st.session_state['user_name'] = None
                st.session_state['dataset'] = None
                st.session_state['processed_dataset'] = None
                st.session_state['model_results'] = {}
                st.experimental_rerun()
            
            st.divider()
            
            # Botón para guardar sesión (si hay resultados)
            if st.session_state.get('model_results'):
                if st.button("💾 Guardar Sesión Actual"):
                    try:
                        guardar_sesion_automatica()
                        st.success("Sesión guardada.")
                    except Exception as e:
                        st.error("Error al guardar la sesión. Revisa los logs.")
                        st.exception(e)
            
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
            if st.session_state.get('processed_dataset') is not None:
                try:
                    mostrar_modelos_regresion(st.session_state['processed_dataset'])
                except Exception as e:
                    st.error("Error en modelos de regresión. Revisa los logs.")
                    st.exception(e)
            else:
                st.warning("⚠️ Primero debes cargar y procesar un dataset.")
                st.info("💡 Ve a la opción 'Carga y Procesamiento de Datos' para comenzar.")
        
        elif opcion == "🔹 Modelos Logísticos":
            if st.session_state.get('processed_dataset') is not None:
                try:
                    mostrar_modelos_logisticos(st.session_state['processed_dataset'])
                except Exception as e:
                    st.error("Error en modelos logísticos. Revisa los logs.")
                    st.exception(e)
            else:
                st.warning("⚠️ Primero debes cargar y procesar un dataset.")
                st.info("💡 Ve a la opción 'Carga y Procesamiento de Datos' para comenzar.")
        
        elif opcion == "🔹 Árboles de Decisión":
            if st.session_state.get('processed_dataset') is not None:
                try:
                    mostrar_arboles_decision(st.session_state['processed_dataset'])
                except Exception as e:
                    st.error("Error en árboles de decisión. Revisa los logs.")
                    st.exception(e)
            else:
                st.warning("⚠️ Primero debes cargar y procesar un dataset.")
                st.info("💡 Ve a la opción 'Carga y Procesamiento de Datos' para comenzar.")
        
        elif opcion == "🔹 Ajuste de Hiperparámetros":
            try:
                mostrar_hyperparameter_tuning()
            except Exception as e:
                st.error("Error en ajuste de hiperparámetros.")
                st.exception(e)
        
        elif opcion == "🔹 Comparación de Modelos":
            try:
                mostrar_comparacion_modelos()
            except Exception as e:
                st.error("Error en comparación de modelos.")
                st.exception(e)
        
        elif opcion == "🔹 Exportar Datos y Resultados":
            try:
                mostrar_exportar_datos()
            except Exception as e:
                st.error("Error en exportar datos.")
                st.exception(e)
        
        elif opcion == "🔹 Historial de Sesiones":
            try:
                mostrar_historial_sesiones()
            except Exception as e:
                st.error("Error en historial de sesiones.")
                st.exception(e)
        
        elif opcion == "🔹 Generar Reporte PDF":
            if st.session_state.get('model_results'):
                try:
                    generar_reporte_pdf(st.session_state['model_results'])
                except Exception as e:
                    st.error("Error al generar reporte PDF.")
                    st.exception(e)
            else:
                st.warning("⚠️ No hay resultados de modelos para generar el reporte.")
                st.info("💡 Ejecuta algunos modelos primero para generar el reporte.")

if __name__ == "__main__":
    main()
# ...existing code...