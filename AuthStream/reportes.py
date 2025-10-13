import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import io
import base64
from datetime import datetime
import tempfile
import os

class ReportePDF(FPDF):
    """Clase personalizada para generar reportes PDF"""
    
    def header(self):
        """Encabezado del PDF"""
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Reporte de Análisis de Machine Learning', 0, 1, 'C')
        self.ln(5)
    
    def footer(self):
        """Pie de página del PDF"""
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}', 0, 0, 'C')
    
    def titulo_seccion(self, titulo):
        """Agregar título de sección"""
        self.ln(5)
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, titulo, 0, 1, 'L')
        self.ln(2)
    
    def subtitulo(self, subtitulo):
        """Agregar subtítulo"""
        self.ln(2)
        self.set_font('Arial', 'B', 12)
        self.cell(0, 8, subtitulo, 0, 1, 'L')
        self.ln(1)
    
    def texto_normal(self, texto):
        """Agregar texto normal"""
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 6, texto)
        self.ln(1)
    
    def tabla_metricas(self, titulo, datos):
        """Agregar tabla de métricas"""
        self.subtitulo(titulo)
        
        # Ancho de columnas
        col_width = 90
        
        # Encabezados
        self.set_font('Arial', 'B', 10)
        self.cell(col_width, 7, 'Métrica', 1, 0, 'C')
        self.cell(col_width, 7, 'Valor', 1, 1, 'C')
        
        # Datos
        self.set_font('Arial', '', 10)
        for metrica, valor in datos.items():
            if isinstance(valor, (int, float)):
                valor_str = f"{valor:.4f}" if isinstance(valor, float) else str(valor)
            else:
                valor_str = str(valor)
            
            self.cell(col_width, 6, metrica.replace('_', ' ').title(), 1, 0, 'L')
            self.cell(col_width, 6, valor_str, 1, 1, 'C')
        
        self.ln(5)

def generar_reporte_pdf(resultados_modelos):
    """Generar reporte PDF con los resultados de todos los modelos"""
    st.header("📄 Generación de Reporte PDF")
    
    if not resultados_modelos:
        st.error("❌ No hay resultados de modelos disponibles")
        return
    
    # Información del reporte
    st.subheader("📊 Resumen de Resultados Disponibles")
    
    for tipo_modelo, resultados in resultados_modelos.items():
        with st.expander(f"🔍 {tipo_modelo.replace('_', ' ').title()}"):
            if tipo_modelo == 'regresion':
                st.write("**Modelos de Regresión Entrenados:**")
                for modelo, metricas in resultados.items():
                    st.write(f"- {modelo}: R² = {metricas['r2']:.4f}")
            
            elif tipo_modelo == 'logistica':
                st.write("**Modelo Logístico:**")
                st.write(f"- Exactitud: {resultados['accuracy']:.4f}")
                if 'auc' in resultados:
                    st.write(f"- AUC: {resultados['auc']:.4f}")
            
            elif 'arbol' in tipo_modelo:
                st.write(f"**Árbol de Decisión - {tipo_modelo.split('_')[1].title()}:**")
                if 'accuracy' in resultados:
                    st.write(f"- Exactitud: {resultados['accuracy']:.4f}")
                elif 'r2' in resultados:
                    st.write(f"- R²: {resultados['r2']:.4f}")
                st.write(f"- Profundidad: {resultados['depth']}")
                st.write(f"- Número de hojas: {resultados['n_leaves']}")
    
    # Configuración del reporte
    st.subheader("⚙️ Configuración del Reporte")
    
    col1, col2 = st.columns(2)
    
    with col1:
        incluir_graficos = st.checkbox("📈 Incluir gráficos", value=True)
        incluir_metricas = st.checkbox("📊 Incluir tablas de métricas", value=True)
    
    with col2:
        incluir_interpretacion = st.checkbox("💡 Incluir interpretaciones", value=True)
        incluir_recomendaciones = st.checkbox("🎯 Incluir recomendaciones", value=True)
    
    # Información del usuario
    nombre_usuario = st.session_state.get('user_name', 'Usuario')
    email_usuario = st.session_state.get('user_email', 'No especificado')
    
    if st.button("📄 Generar Reporte PDF", type="primary"):
        try:
            # Crear PDF
            pdf = ReportePDF()
            pdf.add_page()
            
            # Información general
            pdf.titulo_seccion("INFORMACIÓN GENERAL")
            pdf.texto_normal(f"Usuario: {nombre_usuario}")
            pdf.texto_normal(f"Email: {email_usuario}")
            pdf.texto_normal(f"Fecha de generación: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
            pdf.texto_normal(f"Total de tipos de modelos analizados: {len(resultados_modelos)}")
            
            # Procesar cada tipo de modelo
            for tipo_modelo, resultados in resultados_modelos.items():
                
                if tipo_modelo == 'regresion':
                    pdf.add_page()
                    pdf.titulo_seccion("MODELOS DE REGRESIÓN")
                    
                    for modelo, metricas in resultados.items():
                        pdf.subtitulo(f"Modelo: {modelo}")
                        
                        if incluir_metricas:
                            datos_tabla = {
                                'R²': metricas['r2'],
                                'MSE': metricas['mse'],
                                'RMSE': metricas['rmse'],
                                'MAE': metricas['mae']
                            }
                            pdf.tabla_metricas("Métricas de Rendimiento", datos_tabla)
                        
                        if incluir_interpretacion:
                            interpretacion = generar_interpretacion_regresion(metricas['r2'])
                            pdf.texto_normal(f"Interpretación: {interpretacion}")
                
                elif tipo_modelo == 'logistica':
                    pdf.add_page()
                    pdf.titulo_seccion("MODELO LOGÍSTICO")
                    
                    if incluir_metricas:
                        datos_tabla = {
                            'Exactitud': resultados['accuracy'],
                            'Precisión': resultados['precision'],
                            'Recall': resultados['recall'],
                            'F1-Score': resultados['f1_score']
                        }
                        if 'auc' in resultados:
                            datos_tabla['AUC'] = resultados['auc']
                        
                        pdf.tabla_metricas("Métricas de Clasificación", datos_tabla)
                    
                    if incluir_interpretacion:
                        interpretacion = generar_interpretacion_clasificacion(resultados['accuracy'])
                        pdf.texto_normal(f"Interpretación: {interpretacion}")
                
                elif 'arbol' in tipo_modelo:
                    pdf.add_page()
                    tipo_arbol = tipo_modelo.split('_')[1].title()
                    pdf.titulo_seccion(f"ÁRBOL DE DECISIÓN - {tipo_arbol.upper()}")
                    
                    if incluir_metricas:
                        datos_tabla = {
                            'Profundidad': resultados['depth'],
                            'Número de hojas': resultados['n_leaves']
                        }
                        
                        if 'accuracy' in resultados:
                            datos_tabla.update({
                                'Exactitud': resultados['accuracy'],
                                'Precisión': resultados['precision'],
                                'Recall': resultados['recall'],
                                'F1-Score': resultados['f1_score']
                            })
                        elif 'r2' in resultados:
                            datos_tabla.update({
                                'R²': resultados['r2'],
                                'MSE': resultados['mse'],
                                'RMSE': resultados['rmse'],
                                'MAE': resultados['mae']
                            })
                        
                        pdf.tabla_metricas("Métricas del Árbol", datos_tabla)
                    
                    # Importancia de variables
                    if 'feature_importances' in resultados:
                        pdf.subtitulo("Importancia de Variables")
                        importancias = resultados['feature_importances']
                        importancias_ordenadas = sorted(importancias.items(), key=lambda x: x[1], reverse=True)
                        
                        for i, (variable, importancia) in enumerate(importancias_ordenadas[:5]):  # Top 5
                            pdf.texto_normal(f"{i+1}. {variable}: {importancia:.4f}")
            
            # Recomendaciones generales
            if incluir_recomendaciones:
                pdf.add_page()
                pdf.titulo_seccion("RECOMENDACIONES Y CONCLUSIONES")
                
                recomendaciones = generar_recomendaciones(resultados_modelos)
                for recomendacion in recomendaciones:
                    pdf.texto_normal(f"• {recomendacion}")
            
            # Guardar PDF en memoria
            pdf_output = io.BytesIO()
            pdf_content = pdf.output(dest='S').encode('latin-1')
            pdf_output.write(pdf_content)
            pdf_output.seek(0)
            
            # Botón de descarga
            st.success("✅ Reporte PDF generado exitosamente")
            
            # Crear nombre de archivo único
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            nombre_archivo = f"reporte_ml_{nombre_usuario.replace(' ', '_')}_{timestamp}.pdf"
            
            st.download_button(
                label="📥 Descargar Reporte PDF",
                data=pdf_output.getvalue(),
                file_name=nombre_archivo,
                mime="application/pdf",
                type="primary"
            )
            
            # Mostrar vista previa
            with st.expander("👁️ Vista Previa del Reporte"):
                st.info(f"📄 Archivo: {nombre_archivo}")
                st.info(f"📊 Páginas aproximadas: {pdf.page_no()}")
                st.info(f"🔍 Modelos incluidos: {', '.join(resultados_modelos.keys())}")
                
        except Exception as e:
            st.error(f"❌ Error al generar el reporte PDF: {str(e)}")
            st.info("💡 Asegúrate de haber ejecutado algunos modelos antes de generar el reporte")

def generar_interpretacion_regresion(r2):
    """Generar interpretación para modelos de regresión"""
    if r2 >= 0.9:
        return "El modelo muestra un excelente ajuste a los datos, explicando más del 90% de la variabilidad."
    elif r2 >= 0.7:
        return "El modelo presenta un buen ajuste, explicando una proporción significativa de la variabilidad."
    elif r2 >= 0.5:
        return "El modelo muestra un ajuste moderado. Se recomienda explorar mejoras en las características."
    else:
        return "El modelo presenta un ajuste pobre. Se recomienda revisar las características utilizadas o el tipo de modelo."

def generar_interpretacion_clasificacion(accuracy):
    """Generar interpretación para modelos de clasificación"""
    if accuracy >= 0.9:
        return "El modelo presenta un rendimiento excelente con alta exactitud en las predicciones."
    elif accuracy >= 0.8:
        return "El modelo muestra un buen rendimiento general para las tareas de clasificación."
    elif accuracy >= 0.7:
        return "El modelo presenta un rendimiento moderado. Considere ajustar hiperparámetros o características."
    else:
        return "El modelo necesita mejoras significativas. Revise los datos y la selección de características."

def generar_recomendaciones(resultados_modelos):
    """Generar recomendaciones basadas en todos los resultados"""
    recomendaciones = []
    
    # Recomendaciones por tipo de modelo
    if 'regresion' in resultados_modelos:
        mejor_r2 = max([m['r2'] for m in resultados_modelos['regresion'].values()])
        if mejor_r2 < 0.7:
            recomendaciones.append("Considere recopilar más características relevantes para mejorar el rendimiento de los modelos de regresión.")
        else:
            recomendaciones.append("Los modelos de regresión muestran buen rendimiento. Considere validación cruzada para confirmar la estabilidad.")
    
    if 'logistica' in resultados_modelos:
        accuracy = resultados_modelos['logistica']['accuracy']
        if accuracy < 0.8:
            recomendaciones.append("El modelo logístico podría beneficiarse de ingeniería de características adicional o ajuste de hiperparámetros.")
        else:
            recomendaciones.append("El modelo logístico muestra buen rendimiento. Considere evaluarlo en datos de validación independientes.")
    
    if any('arbol' in k for k in resultados_modelos.keys()):
        recomendaciones.append("Los árboles de decisión proporcionan interpretabilidad. Use la importancia de características para entender qué variables son más relevantes.")
    
    # Recomendaciones generales
    recomendaciones.extend([
        "Realice validación cruzada para evaluar la estabilidad de los modelos.",
        "Considere técnicas de ensemble para mejorar el rendimiento predictivo.",
        "Monitoree el rendimiento del modelo en producción y reentrenelo periódicamente.",
        "Documente las decisiones tomadas durante el desarrollo del modelo para reproducibilidad."
    ])
    
    return recomendaciones
