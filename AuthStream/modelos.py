import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error, 
                           accuracy_score, precision_score, recall_score, f1_score, 
                           confusion_matrix, roc_curve, auc, classification_report)
import warnings
warnings.filterwarnings('ignore')

def mostrar_modelos_regresion(df):
    """Mostrar interfaz para modelos de regresión"""
    st.header("📈 Modelos de Regresión")
    
    if df is None or df.empty:
        st.error("❌ No hay datos disponibles para entrenar modelos")
        return
    
    # Seleccionar variables
    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(columnas_numericas) < 2:
        st.error("❌ Se necesitan al menos 2 columnas numéricas para regresión")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        variable_objetivo = st.selectbox("🎯 Variable Objetivo (Y)", columnas_numericas)
    
    with col2:
        variables_predictoras = st.multiselect(
            "📊 Variables Predictoras (X)", 
            [col for col in columnas_numericas if col != variable_objetivo]
        )
    
    if not variables_predictoras:
        st.warning("⚠️ Selecciona al menos una variable predictora")
        return
    
    # Preparar datos
    X = df[variables_predictoras]
    y = df[variable_objetivo]
    
    # División de datos
    test_size = st.slider("📊 Tamaño del conjunto de prueba (%)", 10, 50, 20) / 100
    random_state = st.number_input("🎲 Random State", min_value=0, value=42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=int(random_state)
    )
    
    st.subheader("🔧 Configuración de Modelos")
    
    # Seleccionar modelos a entrenar
    modelos_seleccionados = st.multiselect(
        "Selecciona los modelos a entrenar:",
        [
            "Regresión Lineal Múltiple",
            "Regresión Polinómica",
            "Regresión con Kernel RBF", 
            "Regresión con Kernel Polinómico",
            "Regresión Ridge",
            "Regresión Lasso"
        ],
        default=["Regresión Lineal Múltiple"]
    )
    
    if st.button("🚀 Entrenar Modelos", type="primary"):
        entrenar_modelos_regresion(X_train, X_test, y_train, y_test, modelos_seleccionados, variables_predictoras, variable_objetivo)

def entrenar_modelos_regresion(X_train, X_test, y_train, y_test, modelos_seleccionados, var_predictoras, var_objetivo):
    """Entrenar y evaluar modelos de regresión"""
    resultados = {}
    
    # Escalar datos para algunos modelos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    progress_bar = st.progress(0)
    total_modelos = len(modelos_seleccionados)
    
    for i, modelo_nombre in enumerate(modelos_seleccionados):
        st.subheader(f"🔍 {modelo_nombre}")
        
        try:
            if modelo_nombre == "Regresión Lineal Múltiple":
                modelo = LinearRegression()
                modelo.fit(X_train, y_train)
                y_pred = modelo.predict(X_test)
                
                # Mostrar coeficientes
                coef_df = pd.DataFrame({
                    'Variable': var_predictoras,
                    'Coeficiente': modelo.coef_
                })
                st.write("📊 **Coeficientes:**")
                st.dataframe(coef_df, use_container_width=True)
                st.write(f"🎯 **Intercepto:** {modelo.intercept_:.4f}")
            
            elif modelo_nombre == "Regresión Polinómica":
                grado = st.slider(f"Grado polinómico para {modelo_nombre}", 2, 5, 2, key=f"poly_{i}")
                poly_features = PolynomialFeatures(degree=grado, include_bias=False)
                X_train_poly = poly_features.fit_transform(X_train)
                X_test_poly = poly_features.transform(X_test)
                
                modelo = LinearRegression()
                modelo.fit(X_train_poly, y_train)
                y_pred = modelo.predict(X_test_poly)
                
                st.write(f"📊 **Grado del polinomio:** {grado}")
                st.write(f"📊 **Características generadas:** {X_train_poly.shape[1]}")
            
            elif modelo_nombre == "Regresión con Kernel RBF":
                gamma = st.slider(f"Gamma para RBF", 0.001, 1.0, 0.1, key=f"rbf_{i}")
                modelo = KernelRidge(kernel='rbf', gamma=gamma)
                modelo.fit(X_train_scaled, y_train)
                y_pred = modelo.predict(X_test_scaled)
                
                st.write(f"📊 **Gamma:** {gamma}")
            
            elif modelo_nombre == "Regresión con Kernel Polinómico":
                grado = st.slider(f"Grado para Kernel Polinómico", 2, 5, 3, key=f"kernel_poly_{i}")
                modelo = KernelRidge(kernel='poly', degree=grado)
                modelo.fit(X_train_scaled, y_train)
                y_pred = modelo.predict(X_test_scaled)
                
                st.write(f"📊 **Grado del kernel:** {grado}")
            
            elif modelo_nombre == "Regresión Ridge":
                alpha = st.slider(f"Alpha para Ridge", 0.001, 10.0, 1.0, key=f"ridge_{i}")
                modelo = Ridge(alpha=alpha)
                modelo.fit(X_train, y_train)
                y_pred = modelo.predict(X_test)
                
                st.write(f"📊 **Alpha (regularización):** {alpha}")
                
                # Mostrar coeficientes
                coef_df = pd.DataFrame({
                    'Variable': var_predictoras,
                    'Coeficiente': modelo.coef_
                })
                st.write("📊 **Coeficientes regularizados:**")
                st.dataframe(coef_df, use_container_width=True)
            
            elif modelo_nombre == "Regresión Lasso":
                alpha = st.slider(f"Alpha para Lasso", 0.001, 10.0, 1.0, key=f"lasso_{i}")
                modelo = Lasso(alpha=alpha)
                modelo.fit(X_train, y_train)
                y_pred = modelo.predict(X_test)
                
                st.write(f"📊 **Alpha (regularización):** {alpha}")
                
                # Mostrar coeficientes (algunos pueden ser 0)
                coef_df = pd.DataFrame({
                    'Variable': var_predictoras,
                    'Coeficiente': modelo.coef_
                })
                st.write("📊 **Coeficientes regularizados (Lasso puede hacer algunos = 0):**")
                st.dataframe(coef_df, use_container_width=True)
                
                # Variables seleccionadas por Lasso
                variables_seleccionadas = np.sum(modelo.coef_ != 0)
                st.write(f"🎯 **Variables seleccionadas por Lasso:** {variables_seleccionadas} de {len(var_predictoras)}")
            
            # Calcular métricas
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            
            # Mostrar métricas
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("R²", f"{r2:.4f}")
            with col2:
                st.metric("MSE", f"{mse:.4f}")
            with col3:
                st.metric("RMSE", f"{rmse:.4f}")
            with col4:
                st.metric("MAE", f"{mae:.4f}")
            
            # Interpretación de R²
            if r2 >= 0.9:
                st.success("✅ Excelente ajuste (R² ≥ 0.9)")
            elif r2 >= 0.7:
                st.info("✅ Buen ajuste (R² ≥ 0.7)")
            elif r2 >= 0.5:
                st.warning("⚠️ Ajuste moderado (R² ≥ 0.5)")
            else:
                st.error("❌ Ajuste pobre (R² < 0.5)")
            
            # Gráfico de dispersión con línea de regresión
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(y_test, y_pred, alpha=0.7)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax.set_xlabel(f'Valores Reales ({var_objetivo})')
            ax.set_ylabel(f'Valores Predichos ({var_objetivo})')
            ax.set_title(f'{modelo_nombre} - Predicciones vs Valores Reales')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
            
            # Guardar resultados
            resultados[modelo_nombre] = {
                'r2': r2,
                'mse': mse, 
                'rmse': rmse,
                'mae': mae,
                'y_test': y_test,
                'y_pred': y_pred
            }
            
        except Exception as e:
            st.error(f"❌ Error al entrenar {modelo_nombre}: {str(e)}")
        
        progress_bar.progress((i + 1) / total_modelos)
    
    # Guardar resultados en sesión
    if 'model_results' not in st.session_state:
        st.session_state.model_results = {}
    st.session_state.model_results['regresion'] = resultados
    
    # Comparación de modelos
    if len(resultados) > 1:
        st.subheader("📊 Comparación de Modelos")
        comparacion_df = pd.DataFrame({
            'Modelo': list(resultados.keys()),
            'R²': [resultados[m]['r2'] for m in resultados.keys()],
            'RMSE': [resultados[m]['rmse'] for m in resultados.keys()],
            'MAE': [resultados[m]['mae'] for m in resultados.keys()]
        }).sort_values('R²', ascending=False)
        
        st.dataframe(comparacion_df, use_container_width=True)
        
        # Mejor modelo
        mejor_modelo = comparacion_df.iloc[0]['Modelo']
        st.success(f"🏆 **Mejor modelo:** {mejor_modelo} (R² = {comparacion_df.iloc[0]['R²']:.4f})")

def mostrar_modelos_logisticos(df):
    """Mostrar interfaz para modelos logísticos"""
    st.header("📊 Modelos Logísticos")
    
    if df is None or df.empty:
        st.error("❌ No hay datos disponibles para entrenar modelos")
        return
    
    # Identificar posibles variables categóricas para clasificación
    todas_columnas = df.columns.tolist()
    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Sugerir variables con pocos valores únicos como candidatas para clasificación
    candidatas_clasificacion = []
    for col in columnas_numericas:
        valores_unicos = df[col].nunique()
        if valores_unicos <= 10:  # Pocas clases únicas
            candidatas_clasificacion.append(col)
    
    if not candidatas_clasificacion:
        st.warning("⚠️ No se encontraron variables adecuadas para clasificación binaria/multiclase.")
        st.info("💡 Creando variable categórica basada en la mediana para demostración...")
        
        # Crear variable categórica basada en la mediana de una variable numérica
        if columnas_numericas:
            var_base = st.selectbox("Selecciona variable para categorizar:", columnas_numericas)
            mediana = df[var_base].median()
            df_modificado = df.copy()
            df_modificado['categoria_target'] = (df_modificado[var_base] > mediana).astype(int)
            candidatas_clasificacion = ['categoria_target']
            df = df_modificado
            st.info(f"✅ Variable 'categoria_target' creada: 1 si {var_base} > {mediana:.2f}, 0 en caso contrario")
    
    # Seleccionar variables
    col1, col2 = st.columns(2)
    
    with col1:
        variable_objetivo = st.selectbox("🎯 Variable Objetivo (Y)", candidatas_clasificacion)
    
    with col2:
        variables_predictoras = st.multiselect(
            "📊 Variables Predictoras (X)", 
            [col for col in columnas_numericas if col != variable_objetivo]
        )
    
    if not variables_predictoras:
        st.warning("⚠️ Selecciona al menos una variable predictora")
        return
    
    # Verificar que la variable objetivo sea binaria o tenga pocas clases
    clases_unicas = df[variable_objetivo].nunique()
    st.info(f"📊 Variable objetivo tiene {clases_unicas} clases únicas")
    
    if clases_unicas > 10:
        st.error("❌ La variable objetivo tiene demasiadas clases para regresión logística")
        return
    
    # Preparar datos
    X = df[variables_predictoras]
    y = df[variable_objetivo]
    
    # División de datos
    test_size = st.slider("📊 Tamaño del conjunto de prueba (%)", 10, 50, 20) / 100
    random_state = st.number_input("🎲 Random State", min_value=0, value=42, key="logistic_random_state")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=int(random_state), stratify=y
    )
    
    if st.button("🚀 Entrenar Modelo Logístico", type="primary"):
        entrenar_modelo_logistico(X_train, X_test, y_train, y_test, variables_predictoras, variable_objetivo, df)

def entrenar_modelo_logistico(X_train, X_test, y_train, y_test, var_predictoras, var_objetivo, df_original):
    """Entrenar y evaluar modelo de regresión logística"""
    try:
        st.subheader("🔍 Regresión Logística")
        
        # Escalar datos
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Entrenar modelo
        modelo = LogisticRegression(random_state=42, max_iter=1000)
        modelo.fit(X_train_scaled, y_train)
        
        # Predicciones
        y_pred = modelo.predict(X_test_scaled)
        y_pred_proba = modelo.predict_proba(X_test_scaled)
        
        # Mostrar coeficientes y odds ratios
        coef_df = pd.DataFrame({
            'Variable': var_predictoras,
            'Coeficiente': modelo.coef_[0] if len(modelo.coef_.shape) == 2 else modelo.coef_,
            'Odds Ratio': np.exp(modelo.coef_[0] if len(modelo.coef_.shape) == 2 else modelo.coef_)
        })
        st.write("📊 **Coeficientes y Odds Ratios:**")
        st.dataframe(coef_df, use_container_width=True)
        st.write(f"🎯 **Intercepto:** {modelo.intercept_[0]:.4f}")
        
        # Log-odds
        st.subheader("📈 Log-odds y Función Sigmoide")
        
        # Graficar función sigmoide
        z = np.linspace(-10, 10, 100)
        sigmoid = 1 / (1 + np.exp(-z))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Función sigmoide
        ax1.plot(z, sigmoid, 'b-', linewidth=2)
        ax1.set_xlabel('Z (log-odds)')
        ax1.set_ylabel('P(Y=1)')
        ax1.set_title('Función Sigmoide')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Umbral = 0.5')
        ax1.axvline(x=0, color='r', linestyle='--', alpha=0.7)
        ax1.legend()
        
        # Distribución de probabilidades predichas
        ax2.hist(y_pred_proba[:, 1], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Probabilidad Predicha')
        ax2.set_ylabel('Frecuencia')
        ax2.set_title('Distribución de Probabilidades Predichas')
        ax2.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        plt.close()
        
        # Métricas de clasificación
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Mostrar métricas
        st.subheader("📊 Métricas de Clasificación")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Exactitud", f"{accuracy:.4f}")
        with col2:
            st.metric("Precisión", f"{precision:.4f}")
        with col3:
            st.metric("Recall", f"{recall:.4f}")
        with col4:
            st.metric("F1-Score", f"{f1:.4f}")
        
        # Interpretación de métricas
        if accuracy >= 0.9:
            st.success("✅ Excelente rendimiento (Exactitud ≥ 90%)")
        elif accuracy >= 0.8:
            st.info("✅ Buen rendimiento (Exactitud ≥ 80%)")
        elif accuracy >= 0.7:
            st.warning("⚠️ Rendimiento moderado (Exactitud ≥ 70%)")
        else:
            st.error("❌ Rendimiento pobre (Exactitud < 70%)")
        
        # Matriz de confusión
        st.subheader("🎭 Matriz de Confusión")
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicción')
        ax.set_ylabel('Real')
        ax.set_title('Matriz de Confusión')
        st.pyplot(fig)
        plt.close()
        
        # Curva ROC (solo para clasificación binaria)
        clases_unicas = len(np.unique(y_test))
        if clases_unicas == 2:
            st.subheader("📈 Curva ROC")
            
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Línea base (AUC = 0.5)')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('Tasa de Falsos Positivos')
            ax.set_ylabel('Tasa de Verdaderos Positivos')
            ax.set_title('Curva ROC')
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
            
            # Interpretación del AUC
            if roc_auc >= 0.9:
                st.success(f"✅ Excelente capacidad discriminatoria (AUC = {roc_auc:.4f})")
            elif roc_auc >= 0.8:
                st.info(f"✅ Buena capacidad discriminatoria (AUC = {roc_auc:.4f})")
            elif roc_auc >= 0.7:
                st.warning(f"⚠️ Capacidad discriminatoria moderada (AUC = {roc_auc:.4f})")
            else:
                st.error(f"❌ Pobre capacidad discriminatoria (AUC = {roc_auc:.4f})")
            
            # Guardar resultados
            if 'model_results' not in st.session_state:
                st.session_state.model_results = {}
            st.session_state.model_results['logistica'] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': roc_auc,
                'confusion_matrix': cm,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
        else:
            st.info("📊 Curva ROC disponible solo para clasificación binaria")
            
            # Guardar resultados para clasificación multiclase
            if 'model_results' not in st.session_state:
                st.session_state.model_results = {}
            st.session_state.model_results['logistica'] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': cm,
                'y_test': y_test,
                'y_pred': y_pred
            }
        
        # Reporte de clasificación detallado
        st.subheader("📋 Reporte de Clasificación Detallado")
        reporte = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        reporte_df = pd.DataFrame(reporte).transpose()
        st.dataframe(reporte_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error al entrenar el modelo logístico: {str(e)}")

def mostrar_arboles_decision(df):
    """Mostrar interfaz para árboles de decisión"""
    st.header("🌳 Árboles de Decisión")
    
    if df is None or df.empty:
        st.error("❌ No hay datos disponibles para entrenar modelos")
        return
    
    # Seleccionar tipo de árbol
    tipo_arbol = st.selectbox(
        "Selecciona el tipo de árbol:",
        ["Clasificación", "Regresión"]
    )
    
    # Identificar variables apropiadas según el tipo
    todas_columnas = df.columns.tolist()
    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if tipo_arbol == "Clasificación":
        # Buscar variables categóricas o con pocos valores únicos
        candidatas_objetivo = []
        for col in columnas_numericas:
            valores_unicos = df[col].nunique()
            if valores_unicos <= 10:
                candidatas_objetivo.append(col)
        
        if not candidatas_objetivo:
            st.warning("⚠️ No se encontraron variables adecuadas para clasificación.")
            st.info("💡 Creando variable categórica basada en la mediana...")
            
            if columnas_numericas:
                var_base = st.selectbox("Selecciona variable para categorizar:", columnas_numericas)
                mediana = df[var_base].median()
                df_modificado = df.copy()
                df_modificado['categoria_target'] = (df_modificado[var_base] > mediana).astype(int)
                candidatas_objetivo = ['categoria_target']
                df = df_modificado
                st.info(f"✅ Variable 'categoria_target' creada: 1 si {var_base} > {mediana:.2f}, 0 en caso contrario")
        
        variable_objetivo = st.selectbox("🎯 Variable Objetivo (Y)", candidatas_objetivo)
    else:
        variable_objetivo = st.selectbox("🎯 Variable Objetivo (Y)", columnas_numericas)
    
    variables_predictoras = st.multiselect(
        "📊 Variables Predictoras (X)", 
        [col for col in columnas_numericas if col != variable_objetivo]
    )
    
    if not variables_predictoras:
        st.warning("⚠️ Selecciona al menos una variable predictora")
        return
    
    # Configuración del árbol
    st.subheader("⚙️ Configuración del Árbol")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_depth = st.slider("📏 Profundidad máxima", 1, 20, 5)
        min_samples_split = st.slider("🔄 Min. muestras para dividir", 2, 20, 2)
    
    with col2:
        min_samples_leaf = st.slider("🍃 Min. muestras en hoja", 1, 20, 1)
        test_size = st.slider("📊 Tamaño del conjunto de prueba (%)", 10, 50, 20) / 100
    
    random_state = st.number_input("🎲 Random State", min_value=0, value=42, key="tree_random_state")
    
    # Preparar datos
    X = df[variables_predictoras]
    y = df[variable_objetivo]
    
    # División de datos (80% entrenamiento, 20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=int(random_state)
    )
    
    st.info(f"📊 División de datos: {len(X_train)} entrenamiento, {len(X_test)} prueba")
    
    if st.button("🚀 Entrenar Árbol de Decisión", type="primary"):
        entrenar_arbol_decision(X_train, X_test, y_train, y_test, variables_predictoras, 
                              variable_objetivo, tipo_arbol, max_depth, min_samples_split, 
                              min_samples_leaf, int(random_state))

def entrenar_arbol_decision(X_train, X_test, y_train, y_test, var_predictoras, var_objetivo, 
                          tipo_arbol, max_depth, min_samples_split, min_samples_leaf, random_state):
    """Entrenar y evaluar árbol de decisión"""
    try:
        st.subheader(f"🌳 Árbol de Decisión - {tipo_arbol}")
        
        # Crear el modelo según el tipo
        if tipo_arbol == "Clasificación":
            modelo = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state
            )
        else:
            modelo = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state
            )
        
        # Entrenar el modelo
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        
        # Información del árbol
        st.subheader("📊 Información del Árbol")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Profundidad Real", modelo.get_depth())
        with col2:
            st.metric("Número de Hojas", modelo.get_n_leaves())
        with col3:
            st.metric("Nodos Totales", modelo.tree_.node_count)
        with col4:
            st.metric("Características Usadas", np.sum(modelo.feature_importances_ > 0))
        
        # Importancia de variables
        st.subheader("📈 Importancia de Variables")
        importancia_df = pd.DataFrame({
            'Variable': var_predictoras,
            'Importancia': modelo.feature_importances_
        }).sort_values('Importancia', ascending=False)
        
        st.dataframe(importancia_df, use_container_width=True)
        
        # Gráfico de importancia
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=importancia_df, x='Importancia', y='Variable', ax=ax)
        ax.set_title('Importancia de Variables')
        ax.set_xlabel('Importancia')
        st.pyplot(fig)
        plt.close()
        
        # Métricas de evaluación
        st.subheader("📊 Métricas de Evaluación")
        
        if tipo_arbol == "Clasificación":
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Exactitud", f"{accuracy:.4f}")
            with col2:
                st.metric("Precisión", f"{precision:.4f}")
            with col3:
                st.metric("Recall", f"{recall:.4f}")
            with col4:
                st.metric("F1-Score", f"{f1:.4f}")
            
            # Matriz de confusión
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicción')
            ax.set_ylabel('Real')
            ax.set_title('Matriz de Confusión')
            st.pyplot(fig)
            plt.close()
            
        else:
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("R²", f"{r2:.4f}")
            with col2:
                st.metric("MSE", f"{mse:.4f}")
            with col3:
                st.metric("RMSE", f"{rmse:.4f}")
            with col4:
                st.metric("MAE", f"{mae:.4f}")
            
            # Interpretación de R²
            if r2 >= 0.9:
                st.success("✅ Excelente ajuste (R² ≥ 0.9)")
            elif r2 >= 0.7:
                st.info("✅ Buen ajuste (R² ≥ 0.7)")
            elif r2 >= 0.5:
                st.warning("⚠️ Ajuste moderado (R² ≥ 0.5)")
            else:
                st.error("❌ Ajuste pobre (R² < 0.5)")
            
            # Gráfico de predicciones vs reales
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(y_test, y_pred, alpha=0.7)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax.set_xlabel(f'Valores Reales ({var_objetivo})')
            ax.set_ylabel(f'Valores Predichos ({var_objetivo})')
            ax.set_title('Predicciones vs Valores Reales')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        # Visualización del árbol
        st.subheader("🌳 Visualización del Árbol")
        
        # Ajustar tamaño de la figura según profundidad
        profundidad = modelo.get_depth()
        
        if profundidad <= 3:
            figsize = (15, 10)
            fontsize = 12
        elif profundidad <= 5:
            figsize = (20, 12)
            fontsize = 10
        elif profundidad <= 8:
            figsize = (25, 15)
            fontsize = 8
            st.info("🌳 El árbol es profundo. La visualización puede ser difícil de leer. Usa zoom para ver detalles.")
        else:
            figsize = (30, 20)
            fontsize = 6
            st.warning("⚠️ El árbol es muy profundo. La visualización puede ser muy grande. Considera reducir la profundidad máxima.")
        
        fig, ax = plt.subplots(figsize=figsize)
        plot_tree(modelo, 
                 feature_names=var_predictoras,
                 filled=True, 
                 rounded=True,
                 fontsize=fontsize,
                 ax=ax)
        ax.set_title(f'Árbol de Decisión - {tipo_arbol}', fontsize=16, fontweight='bold')
        st.pyplot(fig)
        plt.close()
        
        # Ganancia de información / Criterio de división
        if hasattr(modelo, 'tree_'):
            st.subheader("📊 Información del Criterio de División")
            if tipo_arbol == "Clasificación":
                criterio = "Gini" if modelo.criterion == 'gini' else "Entropía"
                st.write(f"**Criterio utilizado:** {criterio}")
                
                # Calcular ganancia de información en el nodo raíz
                if len(np.unique(y_train)) > 1:
                    impureza_inicial = modelo.tree_.impurity[0]
                    st.write(f"**Impureza inicial (nodo raíz):** {impureza_inicial:.4f}")
            else:
                st.write(f"**Criterio utilizado:** MSE (Error Cuadrático Medio)")
                impureza_inicial = modelo.tree_.impurity[0]
                st.write(f"**MSE inicial (nodo raíz):** {impureza_inicial:.4f}")
        
        # Guardar resultados
        if 'model_results' not in st.session_state:
            st.session_state.model_results = {}
        
        if tipo_arbol == "Clasificación":
            st.session_state.model_results['arbol_clasificacion'] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'depth': modelo.get_depth(),
                'n_leaves': modelo.get_n_leaves(),
                'feature_importances': dict(zip(var_predictoras, modelo.feature_importances_)),
                'y_test': y_test,
                'y_pred': y_pred
            }
        else:
            st.session_state.model_results['arbol_regresion'] = {
                'r2': r2,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'depth': modelo.get_depth(),
                'n_leaves': modelo.get_n_leaves(),
                'feature_importances': dict(zip(var_predictoras, modelo.feature_importances_)),
                'y_test': y_test,
                'y_pred': y_pred
            }
        
    except Exception as e:
        st.error(f"❌ Error al entrenar el árbol de decisión: {str(e)}")
