import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def mostrar_comparacion_modelos():
    """Mostrar dashboard de comparación de modelos"""
    st.header("📊 Comparación de Modelos")
    
    if not st.session_state.get('model_results') or not st.session_state.model_results:
        st.warning("⚠️ No hay resultados de modelos para comparar")
        st.info("💡 Ejecuta algunos modelos primero en las secciones de Regresión, Logística o Árboles de Decisión")
        return
    
    resultados = st.session_state.model_results
    
    # Mostrar resumen general
    st.subheader("📋 Resumen General")
    
    col1, col2, col3 = st.columns(3)
    
    tipos_modelos = list(resultados.keys())
    total_modelos = sum([len(resultados[tipo]) if isinstance(resultados[tipo], dict) and tipo == 'regresion' else 1 for tipo in tipos_modelos])
    
    with col1:
        st.metric("Tipos de Modelos", len(tipos_modelos))
    with col2:
        st.metric("Modelos Entrenados", total_modelos)
    with col3:
        mejores_modelos = obtener_mejores_modelos(resultados)
        mejor_nombre = 'N/A'
        if mejores_modelos and mejores_modelos['mejor_general']:
            mejor_nombre = mejores_modelos['mejor_general']['nombre']
        st.metric("Mejor Modelo", mejor_nombre)
    
    # Tabs para diferentes tipos de comparación
    tab1, tab2, tab3 = st.tabs(["📈 Modelos de Regresión", "📊 Modelos de Clasificación", "🎯 Comparación General"])
    
    with tab1:
        if 'regresion' in resultados:
            mostrar_comparacion_regresion(resultados['regresion'])
        else:
            st.info("No hay modelos de regresión para comparar")
    
    with tab2:
        modelos_clasificacion = {}
        if 'logistica' in resultados:
            modelos_clasificacion['Logística'] = resultados['logistica']
        if 'arbol_clasificacion' in resultados:
            modelos_clasificacion['Árbol de Clasificación'] = resultados['arbol_clasificacion']
        
        if modelos_clasificacion:
            mostrar_comparacion_clasificacion(modelos_clasificacion)
        else:
            st.info("No hay modelos de clasificación para comparar")
    
    with tab3:
        mostrar_comparacion_general(resultados)

def obtener_mejores_modelos(resultados):
    """Obtener los mejores modelos de cada categoría"""
    mejores = {
        'mejor_regresion': None,
        'mejor_clasificacion': None,
        'mejor_general': None
    }
    
    # Mejor modelo de regresión
    if 'regresion' in resultados:
        mejor_r2 = -float('inf')
        mejor_modelo_reg = None
        
        for modelo, metricas in resultados['regresion'].items():
            if metricas.get('r2', -float('inf')) > mejor_r2:
                mejor_r2 = metricas['r2']
                mejor_modelo_reg = {'nombre': modelo, 'r2': mejor_r2}
        
        mejores['mejor_regresion'] = mejor_modelo_reg
        mejores['mejor_general'] = mejor_modelo_reg
    
    # Mejor modelo de clasificación
    mejor_accuracy = -float('inf')
    mejor_modelo_clas = None
    
    if 'logistica' in resultados:
        accuracy = resultados['logistica'].get('accuracy', 0)
        if accuracy > mejor_accuracy:
            mejor_accuracy = accuracy
            mejor_modelo_clas = {'nombre': 'Regresión Logística', 'accuracy': accuracy}
    
    if 'arbol_clasificacion' in resultados:
        accuracy = resultados['arbol_clasificacion'].get('accuracy', 0)
        if accuracy > mejor_accuracy:
            mejor_accuracy = accuracy
            mejor_modelo_clas = {'nombre': 'Árbol de Clasificación', 'accuracy': accuracy}
    
    mejores['mejor_clasificacion'] = mejor_modelo_clas
    
    # Si no hay mejor modelo de regresión, usar el de clasificación como mejor general
    if mejores['mejor_general'] is None and mejor_modelo_clas is not None:
        mejores['mejor_general'] = mejor_modelo_clas
    
    return mejores

def mostrar_comparacion_regresion(modelos_regresion):
    """Mostrar comparación de modelos de regresión"""
    st.subheader("📈 Comparación de Modelos de Regresión")
    
    if not modelos_regresion:
        st.info("No hay modelos de regresión para comparar")
        return
    
    # Crear tabla de comparación
    datos_comparacion = []
    for modelo, metricas in modelos_regresion.items():
        datos_comparacion.append({
            'Modelo': modelo,
            'R²': metricas.get('r2', 0),
            'MSE': metricas.get('mse', 0),
            'RMSE': metricas.get('rmse', 0),
            'MAE': metricas.get('mae', 0)
        })
    
    df_comparacion = pd.DataFrame(datos_comparacion)
    df_comparacion = df_comparacion.sort_values('R²', ascending=False)
    
    # Mostrar tabla
    st.dataframe(
        df_comparacion.style.highlight_max(subset=['R²'], color='lightgreen')
                            .highlight_min(subset=['MSE', 'RMSE', 'MAE'], color='lightgreen'),
        use_container_width=True
    )
    
    # Gráfico de barras comparativo
    st.subheader("📊 Visualización Comparativa")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de R²
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(df_comparacion['Modelo'], df_comparacion['R²'], color='skyblue')
        
        # Colorear la mejor barra
        max_idx = df_comparacion['R²'].idxmax()
        bars[max_idx].set_color('green')
        
        ax.set_xlabel('R² Score')
        ax.set_title('Comparación de R² por Modelo')
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        # Gráfico de RMSE
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(df_comparacion['Modelo'], df_comparacion['RMSE'], color='salmon')
        
        # Colorear la mejor barra (menor RMSE)
        min_idx = df_comparacion['RMSE'].idxmin()
        bars[min_idx].set_color('green')
        
        ax.set_xlabel('RMSE')
        ax.set_title('Comparación de RMSE por Modelo')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    # Radar chart para comparación multidimensional
    st.subheader("🎯 Comparación Multidimensional")
    
    if len(modelos_regresion) <= 5:  # Solo para no sobrecargar el gráfico
        crear_radar_chart_regresion(df_comparacion)
    else:
        st.info("Demasiados modelos para mostrar en gráfico radar. Selecciona los mejores:")
        modelos_seleccionados = st.multiselect(
            "Selecciona modelos para comparar (máximo 5):",
            df_comparacion['Modelo'].tolist(),
            default=df_comparacion['Modelo'].head(3).tolist()
        )
        
        if modelos_seleccionados:
            df_filtrado = df_comparacion[df_comparacion['Modelo'].isin(modelos_seleccionados)]
            crear_radar_chart_regresion(df_filtrado)
    
    # Recomendación
    mejor_modelo = df_comparacion.iloc[0]
    st.success(f"🏆 **Mejor Modelo:** {mejor_modelo['Modelo']} (R² = {mejor_modelo['R²']:.4f})")

def mostrar_comparacion_clasificacion(modelos_clasificacion):
    """Mostrar comparación de modelos de clasificación"""
    st.subheader("📊 Comparación de Modelos de Clasificación")
    
    if not modelos_clasificacion:
        st.info("No hay modelos de clasificación para comparar")
        return
    
    # Crear tabla de comparación
    datos_comparacion = []
    for modelo, metricas in modelos_clasificacion.items():
        datos_comparacion.append({
            'Modelo': modelo,
            'Exactitud': metricas.get('accuracy', 0),
            'Precisión': metricas.get('precision', 0),
            'Recall': metricas.get('recall', 0),
            'F1-Score': metricas.get('f1_score', 0),
            'AUC': metricas.get('auc', 0) if 'auc' in metricas else None
        })
    
    df_comparacion = pd.DataFrame(datos_comparacion)
    df_comparacion = df_comparacion.sort_values('Exactitud', ascending=False)
    
    # Mostrar tabla
    st.dataframe(
        df_comparacion.style.highlight_max(subset=['Exactitud', 'Precisión', 'Recall', 'F1-Score'], color='lightgreen'),
        use_container_width=True
    )
    
    # Gráficos comparativos
    st.subheader("📊 Visualización Comparativa")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de métricas
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(df_comparacion))
        width = 0.2
        
        ax.bar(x - 1.5*width, df_comparacion['Exactitud'], width, label='Exactitud', color='skyblue')
        ax.bar(x - 0.5*width, df_comparacion['Precisión'], width, label='Precisión', color='lightgreen')
        ax.bar(x + 0.5*width, df_comparacion['Recall'], width, label='Recall', color='salmon')
        ax.bar(x + 1.5*width, df_comparacion['F1-Score'], width, label='F1-Score', color='gold')
        
        ax.set_xlabel('Modelos')
        ax.set_ylabel('Score')
        ax.set_title('Comparación de Métricas de Clasificación')
        ax.set_xticks(x)
        ax.set_xticklabels(df_comparacion['Modelo'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        # Gráfico de AUC si está disponible
        if df_comparacion['AUC'].notna().any():
            fig, ax = plt.subplots(figsize=(10, 6))
            
            modelos_con_auc = df_comparacion[df_comparacion['AUC'].notna()]
            bars = ax.barh(modelos_con_auc['Modelo'], modelos_con_auc['AUC'], color='mediumpurple')
            
            # Colorear la mejor barra
            if len(modelos_con_auc) > 0:
                max_idx = modelos_con_auc['AUC'].idxmax()
                bars[modelos_con_auc.index.get_loc(max_idx)].set_color('green')
            
            ax.set_xlabel('AUC Score')
            ax.set_title('Comparación de AUC (Area Under ROC Curve)')
            ax.set_xlim(0, 1)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        else:
            st.info("AUC no disponible para estos modelos")
    
    # Recomendación
    mejor_modelo = df_comparacion.iloc[0]
    st.success(f"🏆 **Mejor Modelo:** {mejor_modelo['Modelo']} (Exactitud = {mejor_modelo['Exactitud']:.4f})")

def mostrar_comparacion_general(resultados):
    """Mostrar comparación general de todos los modelos"""
    st.subheader("🎯 Comparación General")
    
    # Resumen por tipo de modelo
    st.write("### Resumen por Tipo de Modelo")
    
    resumen_datos = []
    
    if 'regresion' in resultados:
        num_modelos = len(resultados['regresion'])
        mejor_r2 = max([m.get('r2', 0) for m in resultados['regresion'].values()])
        resumen_datos.append({
            'Tipo': 'Regresión',
            'Cantidad': num_modelos,
            'Mejor Métrica': f"R² = {mejor_r2:.4f}"
        })
    
    if 'logistica' in resultados:
        accuracy = resultados['logistica'].get('accuracy', 0)
        resumen_datos.append({
            'Tipo': 'Logística',
            'Cantidad': 1,
            'Mejor Métrica': f"Exactitud = {accuracy:.4f}"
        })
    
    if 'arbol_clasificacion' in resultados:
        accuracy = resultados['arbol_clasificacion'].get('accuracy', 0)
        resumen_datos.append({
            'Tipo': 'Árbol de Clasificación',
            'Cantidad': 1,
            'Mejor Métrica': f"Exactitud = {accuracy:.4f}"
        })
    
    if 'arbol_regresion' in resultados:
        r2 = resultados['arbol_regresion'].get('r2', 0)
        resumen_datos.append({
            'Tipo': 'Árbol de Regresión',
            'Cantidad': 1,
            'Mejor Métrica': f"R² = {r2:.4f}"
        })
    
    df_resumen = pd.DataFrame(resumen_datos)
    st.dataframe(df_resumen, use_container_width=True)
    
    # Gráfico de distribución de modelos
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(df_resumen['Cantidad'], labels=df_resumen['Tipo'], autopct='%1.1f%%', startangle=90)
        ax.set_title('Distribución de Tipos de Modelos')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.write("### 📈 Estadísticas Generales")
        total_modelos = df_resumen['Cantidad'].sum()
        st.metric("Total de Modelos Entrenados", total_modelos)
        
        tipos_diferentes = len(df_resumen)
        st.metric("Tipos de Modelos Diferentes", tipos_diferentes)
        
        # Mostrar mejores modelos
        mejores = obtener_mejores_modelos(resultados)
        
        if mejores['mejor_regresion']:
            st.write(f"**Mejor Regresión:** {mejores['mejor_regresion']['nombre']}")
        
        if mejores['mejor_clasificacion']:
            st.write(f"**Mejor Clasificación:** {mejores['mejor_clasificacion']['nombre']}")

def crear_radar_chart_regresion(df_comparacion):
    """Crear gráfico de radar para comparación de modelos de regresión"""
    if df_comparacion.empty:
        return
    
    # Normalizar métricas para el radar chart
    # R² ya está entre 0 y 1
    # Normalizar MSE, RMSE, MAE inversamente (menor es mejor)
    df_norm = df_comparacion.copy()
    
    for col in ['MSE', 'RMSE', 'MAE']:
        if df_norm[col].max() > 0:
            df_norm[col + '_norm'] = 1 - (df_norm[col] / df_norm[col].max())
        else:
            df_norm[col + '_norm'] = 1
    
    # Categorías para el radar
    categorias = ['R²', 'MSE (inv)', 'RMSE (inv)', 'MAE (inv)']
    num_vars = len(categorias)
    
    # Crear gráfico
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Ángulos para cada eje
    angulos = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angulos += angulos[:1]
    
    # Colores para cada modelo
    colores = plt.cm.Set3(np.linspace(0, 1, len(df_norm)))
    
    # Graficar cada modelo
    for idx, row in df_norm.iterrows():
        valores = [row['R²'], row['MSE_norm'], row['RMSE_norm'], row['MAE_norm']]
        valores += valores[:1]
        
        ax.plot(angulos, valores, 'o-', linewidth=2, label=row['Modelo'], color=colores[idx])
        ax.fill(angulos, valores, alpha=0.15, color=colores[idx])
    
    # Configurar ejes
    ax.set_xticks(angulos[:-1])
    ax.set_xticklabels(categorias)
    ax.set_ylim(0, 1)
    ax.set_title('Comparación Multidimensional de Modelos de Regresión', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    st.pyplot(fig)
    plt.close()
