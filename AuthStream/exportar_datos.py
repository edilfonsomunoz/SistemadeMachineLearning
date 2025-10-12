import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io

def mostrar_exportar_datos():
    """Mostrar interfaz de exportación de datos"""
    st.header("📥 Exportar Datos y Resultados")
    
    st.info("""
    💡 **Exportación de Datos**
    
    Descarga tus datos procesados y resultados de modelos en formato CSV o Excel:
    - Dataset procesado con transformaciones aplicadas
    - Predicciones de modelos entrenados
    - Métricas y resultados de análisis
    """)
    
    # Tabs para diferentes tipos de exportación
    tab1, tab2, tab3 = st.tabs(["📊 Dataset Procesado", "🤖 Predicciones de Modelos", "📈 Métricas de Modelos"])
    
    with tab1:
        exportar_dataset_procesado()
    
    with tab2:
        exportar_predicciones()
    
    with tab3:
        exportar_metricas()

def exportar_dataset_procesado():
    """Exportar dataset procesado"""
    st.subheader("📊 Exportar Dataset Procesado")
    
    if st.session_state.get('processed_dataset') is None:
        st.warning("⚠️ No hay dataset procesado disponible")
        st.info("💡 Procesa un dataset primero en la opción 'Carga y Procesamiento de Datos'")
        return
    
    df = st.session_state.processed_dataset
    
    # Información del dataset
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Filas", df.shape[0])
    with col2:
        st.metric("Columnas", df.shape[1])
    with col3:
        st.metric("Tamaño (KB)", f"{df.memory_usage(deep=True).sum() / 1024:.1f}")
    
    # Vista previa
    with st.expander("👁️ Vista Previa del Dataset"):
        st.dataframe(df.head(20), use_container_width=True)
    
    # Opciones de exportación
    st.subheader("⚙️ Opciones de Exportación")
    
    col1, col2 = st.columns(2)
    
    with col1:
        formato = st.selectbox(
            "Formato de archivo:",
            ["CSV", "Excel (XLSX)"]
        )
        
        incluir_indice = st.checkbox("Incluir índice", value=False)
    
    with col2:
        if formato == "CSV":
            separador = st.selectbox(
                "Separador:",
                [",", ";", "\t", "|"],
                format_func=lambda x: {"," : "Coma (,)", ";" : "Punto y coma (;)", "\t" : "Tabulación", "|" : "Pipe (|)"}[x]
            )
        
        encoding = st.selectbox(
            "Codificación:",
            ["utf-8", "latin-1", "iso-8859-1"]
        )
    
    # Botón de descarga
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if formato == "CSV":
        nombre_archivo = f"dataset_procesado_{timestamp}.csv"
        
        # Generar CSV
        csv_buffer = io.StringIO()
        df.to_csv(
            csv_buffer,
            index=incluir_indice,
            sep=separador if formato == "CSV" else ",",
            encoding=encoding
        )
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="📥 Descargar CSV",
            data=csv_data,
            file_name=nombre_archivo,
            mime="text/csv",
            type="primary"
        )
    
    else:  # Excel
        nombre_archivo = f"dataset_procesado_{timestamp}.xlsx"
        
        # Generar Excel
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=incluir_indice, sheet_name='Dataset Procesado')
        
        excel_data = excel_buffer.getvalue()
        
        st.download_button(
            label="📥 Descargar Excel",
            data=excel_data,
            file_name=nombre_archivo,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary"
        )

def exportar_predicciones():
    """Exportar predicciones de modelos"""
    st.subheader("🤖 Exportar Predicciones de Modelos")
    
    if not st.session_state.get('model_results') or not st.session_state.model_results:
        st.warning("⚠️ No hay predicciones disponibles")
        st.info("💡 Entrena algunos modelos primero")
        return
    
    resultados = st.session_state.model_results
    
    # Seleccionar tipo de modelo
    tipos_disponibles = []
    
    if 'regresion' in resultados:
        tipos_disponibles.append("Modelos de Regresión")
    if 'logistica' in resultados:
        tipos_disponibles.append("Modelo Logístico")
    if 'arbol_clasificacion' in resultados:
        tipos_disponibles.append("Árbol de Clasificación")
    if 'arbol_regresion' in resultados:
        tipos_disponibles.append("Árbol de Regresión")
    
    if not tipos_disponibles:
        st.warning("⚠️ No hay predicciones con datos exportables")
        return
    
    tipo_seleccionado = st.selectbox("Selecciona el tipo de modelo:", tipos_disponibles)
    
    # Obtener predicciones según el tipo
    predicciones_df = None
    
    if tipo_seleccionado == "Modelos de Regresión":
        modelos_reg = list(resultados['regresion'].keys())
        modelo_elegido = st.selectbox("Selecciona el modelo:", modelos_reg)
        
        if 'y_test' in resultados['regresion'][modelo_elegido] and 'y_pred' in resultados['regresion'][modelo_elegido]:
            y_test = resultados['regresion'][modelo_elegido]['y_test']
            y_pred = resultados['regresion'][modelo_elegido]['y_pred']
            
            predicciones_df = pd.DataFrame({
                'Valor_Real': y_test if isinstance(y_test, (list, np.ndarray)) else y_test.values,
                'Prediccion': y_pred if isinstance(y_pred, (list, np.ndarray)) else y_pred,
                'Error': (y_test if isinstance(y_test, (list, np.ndarray)) else y_test.values) - y_pred,
                'Error_Absoluto': np.abs((y_test if isinstance(y_test, (list, np.ndarray)) else y_test.values) - y_pred)
            })
    
    elif tipo_seleccionado == "Modelo Logístico":
        if 'y_test' in resultados['logistica'] and 'y_pred' in resultados['logistica']:
            y_test = resultados['logistica']['y_test']
            y_pred = resultados['logistica']['y_pred']
            
            predicciones_df = pd.DataFrame({
                'Clase_Real': y_test if isinstance(y_test, (list, np.ndarray)) else y_test.values,
                'Clase_Predicha': y_pred if isinstance(y_pred, (list, np.ndarray)) else y_pred,
                'Correcto': (y_test if isinstance(y_test, (list, np.ndarray)) else y_test.values) == y_pred
            })
            
            # Agregar probabilidades si están disponibles
            if 'y_pred_proba' in resultados['logistica']:
                y_pred_proba = resultados['logistica']['y_pred_proba']
                if isinstance(y_pred_proba, np.ndarray) and y_pred_proba.ndim == 2:
                    for i in range(y_pred_proba.shape[1]):
                        predicciones_df[f'Probabilidad_Clase_{i}'] = y_pred_proba[:, i]
    
    elif tipo_seleccionado == "Árbol de Clasificación":
        if 'y_test' in resultados['arbol_clasificacion'] and 'y_pred' in resultados['arbol_clasificacion']:
            y_test = resultados['arbol_clasificacion']['y_test']
            y_pred = resultados['arbol_clasificacion']['y_pred']
            
            predicciones_df = pd.DataFrame({
                'Clase_Real': y_test if isinstance(y_test, (list, np.ndarray)) else y_test.values,
                'Clase_Predicha': y_pred if isinstance(y_pred, (list, np.ndarray)) else y_pred,
                'Correcto': (y_test if isinstance(y_test, (list, np.ndarray)) else y_test.values) == y_pred
            })
    
    elif tipo_seleccionado == "Árbol de Regresión":
        if 'y_test' in resultados['arbol_regresion'] and 'y_pred' in resultados['arbol_regresion']:
            y_test = resultados['arbol_regresion']['y_test']
            y_pred = resultados['arbol_regresion']['y_pred']
            
            predicciones_df = pd.DataFrame({
                'Valor_Real': y_test if isinstance(y_test, (list, np.ndarray)) else y_test.values,
                'Prediccion': y_pred if isinstance(y_pred, (list, np.ndarray)) else y_pred,
                'Error': (y_test if isinstance(y_test, (list, np.ndarray)) else y_test.values) - y_pred,
                'Error_Absoluto': np.abs((y_test if isinstance(y_test, (list, np.ndarray)) else y_test.values) - y_pred)
            })
    
    if predicciones_df is not None:
        # Vista previa
        with st.expander("👁️ Vista Previa de Predicciones"):
            st.dataframe(predicciones_df.head(20), use_container_width=True)
        
        # Estadísticas
        st.subheader("📊 Estadísticas")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total de Predicciones", len(predicciones_df))
        
        with col2:
            if 'Correcto' in predicciones_df.columns:
                exactitud = predicciones_df['Correcto'].mean() * 100
                st.metric("Exactitud", f"{exactitud:.2f}%")
            elif 'Error' in predicciones_df.columns:
                mae = predicciones_df['Error_Absoluto'].mean()
                st.metric("MAE", f"{mae:.4f}")
        
        # Opciones de exportación
        formato = st.selectbox(
            "Formato de archivo:",
            ["CSV", "Excel (XLSX)"],
            key="pred_formato"
        )
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if formato == "CSV":
            nombre_archivo = f"predicciones_{timestamp}.csv"
            
            csv_buffer = io.StringIO()
            predicciones_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="📥 Descargar Predicciones CSV",
                data=csv_data,
                file_name=nombre_archivo,
                mime="text/csv",
                type="primary"
            )
        
        else:
            nombre_archivo = f"predicciones_{timestamp}.xlsx"
            
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                predicciones_df.to_excel(writer, index=False, sheet_name='Predicciones')
            
            excel_data = excel_buffer.getvalue()
            
            st.download_button(
                label="📥 Descargar Predicciones Excel",
                data=excel_data,
                file_name=nombre_archivo,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )
    else:
        st.warning("⚠️ No hay datos de predicciones disponibles para este modelo")

def exportar_metricas():
    """Exportar métricas de modelos"""
    st.subheader("📈 Exportar Métricas de Modelos")
    
    if not st.session_state.get('model_results') or not st.session_state.model_results:
        st.warning("⚠️ No hay métricas disponibles")
        st.info("💡 Entrena algunos modelos primero")
        return
    
    resultados = st.session_state.model_results
    
    # Crear DataFrame con todas las métricas
    metricas_lista = []
    
    # Métricas de regresión
    if 'regresion' in resultados:
        for modelo, metricas in resultados['regresion'].items():
            metricas_lista.append({
                'Tipo': 'Regresión',
                'Modelo': modelo,
                'R²': metricas.get('r2', None),
                'MSE': metricas.get('mse', None),
                'RMSE': metricas.get('rmse', None),
                'MAE': metricas.get('mae', None),
                'Exactitud': None,
                'Precisión': None,
                'Recall': None,
                'F1-Score': None
            })
    
    # Métricas de logística
    if 'logistica' in resultados:
        metricas = resultados['logistica']
        metricas_lista.append({
            'Tipo': 'Clasificación',
            'Modelo': 'Regresión Logística',
            'R²': None,
            'MSE': None,
            'RMSE': None,
            'MAE': None,
            'Exactitud': metricas.get('accuracy', None),
            'Precisión': metricas.get('precision', None),
            'Recall': metricas.get('recall', None),
            'F1-Score': metricas.get('f1_score', None)
        })
    
    # Métricas de árboles
    if 'arbol_clasificacion' in resultados:
        metricas = resultados['arbol_clasificacion']
        metricas_lista.append({
            'Tipo': 'Clasificación',
            'Modelo': 'Árbol de Clasificación',
            'R²': None,
            'MSE': None,
            'RMSE': None,
            'MAE': None,
            'Exactitud': metricas.get('accuracy', None),
            'Precisión': metricas.get('precision', None),
            'Recall': metricas.get('recall', None),
            'F1-Score': metricas.get('f1_score', None)
        })
    
    if 'arbol_regresion' in resultados:
        metricas = resultados['arbol_regresion']
        metricas_lista.append({
            'Tipo': 'Regresión',
            'Modelo': 'Árbol de Regresión',
            'R²': metricas.get('r2', None),
            'MSE': metricas.get('mse', None),
            'RMSE': metricas.get('rmse', None),
            'MAE': metricas.get('mae', None),
            'Exactitud': None,
            'Precisión': None,
            'Recall': None,
            'F1-Score': None
        })
    
    if metricas_lista:
        metricas_df = pd.DataFrame(metricas_lista)
        
        # Vista previa
        with st.expander("👁️ Vista Previa de Métricas"):
            st.dataframe(metricas_df, use_container_width=True)
        
        # Estadísticas
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total de Modelos", len(metricas_df))
        
        with col2:
            tipos_unicos = metricas_df['Tipo'].nunique()
            st.metric("Tipos de Modelos", tipos_unicos)
        
        # Opciones de exportación
        formato = st.selectbox(
            "Formato de archivo:",
            ["CSV", "Excel (XLSX)"],
            key="metricas_formato"
        )
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if formato == "CSV":
            nombre_archivo = f"metricas_modelos_{timestamp}.csv"
            
            csv_buffer = io.StringIO()
            metricas_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="📥 Descargar Métricas CSV",
                data=csv_data,
                file_name=nombre_archivo,
                mime="text/csv",
                type="primary"
            )
        
        else:
            nombre_archivo = f"metricas_modelos_{timestamp}.xlsx"
            
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                metricas_df.to_excel(writer, index=False, sheet_name='Métricas')
                
                # Agregar hoja adicional con resumen
                resumen_df = pd.DataFrame({
                    'Tipo de Modelo': metricas_df.groupby('Tipo').size().index,
                    'Cantidad': metricas_df.groupby('Tipo').size().values
                })
                resumen_df.to_excel(writer, index=False, sheet_name='Resumen')
            
            excel_data = excel_buffer.getvalue()
            
            st.download_button(
                label="📥 Descargar Métricas Excel",
                data=excel_data,
                file_name=nombre_archivo,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )
    else:
        st.warning("⚠️ No hay métricas disponibles para exportar")
