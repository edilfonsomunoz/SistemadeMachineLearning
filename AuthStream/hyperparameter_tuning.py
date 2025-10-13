import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import make_scorer, r2_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

def mostrar_hyperparameter_tuning():
    """Mostrar interfaz de ajuste de hiperparámetros"""
    st.header("⚙️ Ajuste de Hiperparámetros")
    
    if st.session_state.get('processed_dataset') is None:
        st.warning("⚠️ Primero debes cargar y procesar un dataset")
        st.info("💡 Ve a la opción 'Carga y Procesamiento de Datos' para comenzar")
        return
    
    df = st.session_state.processed_dataset
    
    st.info("""
    💡 **Grid Search con Validación Cruzada**
    
    Esta herramienta te permite optimizar automáticamente los hiperparámetros de tus modelos:
    - Prueba múltiples combinaciones de parámetros
    - Usa validación cruzada para evaluar cada combinación
    - Encuentra los mejores parámetros para tu modelo
    """)
    
    # Seleccionar tipo de tarea
    tipo_tarea = st.selectbox(
        "Selecciona el tipo de tarea:",
        ["Regresión", "Clasificación"]
    )
    
    # Configurar variables
    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(columnas_numericas) < 2:
        st.error("❌ Se necesitan al menos 2 columnas numéricas")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        if tipo_tarea == "Clasificación":
            # Buscar variables categóricas
            candidatas = [col for col in columnas_numericas if df[col].nunique() <= 10]
            if not candidatas:
                st.warning("⚠️ No hay variables adecuadas para clasificación")
                variable_objetivo = st.selectbox("🎯 Variable Objetivo (Y)", columnas_numericas)
            else:
                variable_objetivo = st.selectbox("🎯 Variable Objetivo (Y)", candidatas)
        else:
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
    
    # Configuración de validación cruzada
    st.subheader("🔧 Configuración de Validación Cruzada")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cv_folds = st.slider("Número de Folds (K)", 3, 10, 5)
        test_size = st.slider("Tamaño de prueba (%)", 10, 40, 20) / 100
    
    with col2:
        random_state = st.number_input("🎲 Random State", min_value=0, value=42, key="tuning_random_state")
    
    # Seleccionar modelo para optimizar
    if tipo_tarea == "Regresión":
        modelo_opciones = [
            "Ridge Regression",
            "Lasso Regression",
            "Decision Tree Regressor",
            "Random Forest Regressor"
        ]
    else:
        modelo_opciones = [
            "Logistic Regression",
            "Decision Tree Classifier",
            "Random Forest Classifier"
        ]
    
    modelo_seleccionado = st.selectbox("🤖 Selecciona el modelo a optimizar:", modelo_opciones)
    
    # Configurar grid de hiperparámetros
    st.subheader("📋 Configuración de Hiperparámetros")
    
    param_grid = configurar_param_grid(modelo_seleccionado, tipo_tarea)
    
    # Mostrar grid configurado
    with st.expander("👁️ Ver Grid de Hiperparámetros"):
        st.json(param_grid)
        total_combinaciones = np.prod([len(v) for v in param_grid.values()])
        st.info(f"📊 Total de combinaciones a probar: {total_combinaciones}")
        st.info(f"⏱️ Estimación de evaluaciones: {total_combinaciones * cv_folds}")
    
    # Botón para ejecutar grid search
    if st.button("🚀 Ejecutar Grid Search", type="primary"):
        ejecutar_grid_search(
            X, y, modelo_seleccionado, param_grid, cv_folds, 
            test_size, int(random_state), tipo_tarea
        )

def configurar_param_grid(modelo_nombre, tipo_tarea):
    """Configurar grid de hiperparámetros según el modelo"""
    param_grid = {}
    
    if modelo_nombre == "Ridge Regression":
        st.write("**Parámetros de Ridge Regression:**")
        col1, col2 = st.columns(2)
        
        with col1:
            alpha_min = st.number_input("Alpha mínimo", 0.001, 10.0, 0.1, key="ridge_alpha_min")
            alpha_max = st.number_input("Alpha máximo", 0.1, 100.0, 10.0, key="ridge_alpha_max")
            alpha_steps = st.number_input("Pasos de Alpha", 3, 10, 5, key="ridge_alpha_steps")
        
        with col2:
            solvers = st.multiselect(
                "Solvers",
                ['auto', 'svd', 'cholesky', 'lsqr', 'sag'],
                default=['auto', 'svd']
            )
        
        param_grid = {
            'alpha': np.logspace(np.log10(alpha_min), np.log10(alpha_max), alpha_steps).tolist(),
            'solver': solvers if solvers else ['auto']
        }
    
    elif modelo_nombre == "Lasso Regression":
        st.write("**Parámetros de Lasso Regression:**")
        col1, col2 = st.columns(2)
        
        with col1:
            alpha_min = st.number_input("Alpha mínimo", 0.001, 10.0, 0.1, key="lasso_alpha_min")
            alpha_max = st.number_input("Alpha máximo", 0.1, 100.0, 10.0, key="lasso_alpha_max")
            alpha_steps = st.number_input("Pasos de Alpha", 3, 10, 5, key="lasso_alpha_steps")
        
        with col2:
            max_iter = st.multiselect(
                "Max Iteraciones",
                [1000, 2000, 5000, 10000],
                default=[1000, 5000]
            )
        
        param_grid = {
            'alpha': np.logspace(np.log10(alpha_min), np.log10(alpha_max), alpha_steps).tolist(),
            'max_iter': max_iter if max_iter else [1000]
        }
    
    elif modelo_nombre == "Logistic Regression":
        st.write("**Parámetros de Logistic Regression:**")
        col1, col2 = st.columns(2)
        
        with col1:
            C_values = st.multiselect(
                "Valores de C (inverso de regularización)",
                [0.001, 0.01, 0.1, 1, 10, 100],
                default=[0.1, 1, 10]
            )
        
        with col2:
            solvers = st.multiselect(
                "Solvers",
                ['liblinear', 'lbfgs', 'sag', 'saga'],
                default=['liblinear', 'lbfgs']
            )
        
        param_grid = {
            'C': C_values if C_values else [1],
            'solver': solvers if solvers else ['liblinear'],
            'max_iter': [1000, 5000]
        }
    
    elif "Decision Tree" in modelo_nombre:
        st.write(f"**Parámetros de {modelo_nombre}:**")
        col1, col2 = st.columns(2)
        
        with col1:
            max_depth = st.multiselect(
                "Profundidad Máxima",
                [3, 5, 7, 10, 15, 20, None],
                default=[5, 10, None]
            )
            min_samples_split = st.multiselect(
                "Min Samples Split",
                [2, 5, 10, 20],
                default=[2, 10]
            )
        
        with col2:
            min_samples_leaf = st.multiselect(
                "Min Samples Leaf",
                [1, 2, 5, 10],
                default=[1, 5]
            )
            criterion = st.multiselect(
                "Criterio",
                ['gini', 'entropy'] if tipo_tarea == "Clasificación" else ['squared_error', 'friedman_mse'],
                default=['gini', 'entropy'] if tipo_tarea == "Clasificación" else ['squared_error']
            )
        
        param_grid = {
            'max_depth': max_depth if max_depth else [None],
            'min_samples_split': min_samples_split if min_samples_split else [2],
            'min_samples_leaf': min_samples_leaf if min_samples_leaf else [1],
            'criterion': criterion if criterion else (['gini'] if tipo_tarea == "Clasificación" else ['squared_error'])
        }
    
    elif "Random Forest" in modelo_nombre:
        st.write(f"**Parámetros de {modelo_nombre}:**")
        col1, col2 = st.columns(2)
        
        with col1:
            n_estimators = st.multiselect(
                "Número de Árboles",
                [10, 50, 100, 200],
                default=[50, 100]
            )
            max_depth = st.multiselect(
                "Profundidad Máxima",
                [5, 10, 15, None],
                default=[10, None]
            )
        
        with col2:
            min_samples_split = st.multiselect(
                "Min Samples Split",
                [2, 5, 10],
                default=[2, 5]
            )
            min_samples_leaf = st.multiselect(
                "Min Samples Leaf",
                [1, 2, 4],
                default=[1, 2]
            )
        
        param_grid = {
            'n_estimators': n_estimators if n_estimators else [100],
            'max_depth': max_depth if max_depth else [None],
            'min_samples_split': min_samples_split if min_samples_split else [2],
            'min_samples_leaf': min_samples_leaf if min_samples_leaf else [1]
        }
    
    return param_grid

def ejecutar_grid_search(X, y, modelo_nombre, param_grid, cv_folds, test_size, random_state, tipo_tarea):
    """Ejecutar Grid Search con validación cruzada"""
    
    try:
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Crear modelo base
        if modelo_nombre == "Ridge Regression":
            modelo_base = Ridge()
            scoring = 'r2'
        elif modelo_nombre == "Lasso Regression":
            modelo_base = Lasso()
            scoring = 'r2'
        elif modelo_nombre == "Logistic Regression":
            modelo_base = LogisticRegression()
            scoring = 'accuracy'
        elif modelo_nombre == "Decision Tree Regressor":
            modelo_base = DecisionTreeRegressor(random_state=random_state)
            scoring = 'r2'
        elif modelo_nombre == "Decision Tree Classifier":
            modelo_base = DecisionTreeClassifier(random_state=random_state)
            scoring = 'accuracy'
        elif modelo_nombre == "Random Forest Regressor":
            modelo_base = RandomForestRegressor(random_state=random_state, n_jobs=-1)
            scoring = 'r2'
        elif modelo_nombre == "Random Forest Classifier":
            modelo_base = RandomForestClassifier(random_state=random_state, n_jobs=-1)
            scoring = 'accuracy'
        
        # Ejecutar Grid Search
        st.info("🔄 Ejecutando Grid Search... Esto puede tomar unos momentos...")
        
        grid_search = GridSearchCV(
            estimator=modelo_base,
            param_grid=param_grid,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1,
            verbose=0
        )
        
        with st.spinner("⏳ Buscando los mejores hiperparámetros..."):
            grid_search.fit(X_train, y_train)
        
        # Resultados
        st.success("✅ Grid Search completado exitosamente")
        
        # Mostrar mejores parámetros
        st.subheader("🏆 Mejores Hiperparámetros")
        
        mejores_params_df = pd.DataFrame([grid_search.best_params_]).T
        mejores_params_df.columns = ['Valor']
        mejores_params_df.index.name = 'Parámetro'
        st.dataframe(mejores_params_df, use_container_width=True)
        
        # Métricas del mejor modelo
        st.subheader("📊 Rendimiento del Mejor Modelo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(f"Score CV (promedio {cv_folds}-fold)", f"{grid_search.best_score_:.4f}")
        
        with col2:
            # Evaluar en test set
            y_pred = grid_search.best_estimator_.predict(X_test)
            if tipo_tarea == "Regresión":
                test_score = r2_score(y_test, y_pred)
                st.metric("R² en Test Set", f"{test_score:.4f}")
            else:
                test_score = accuracy_score(y_test, y_pred)
                st.metric("Exactitud en Test Set", f"{test_score:.4f}")
        
        # Mostrar todos los resultados
        st.subheader("📈 Todos los Resultados del Grid Search")
        
        resultados_cv = pd.DataFrame(grid_search.cv_results_)
        
        # Seleccionar columnas relevantes
        cols_mostrar = ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
        resultados_display = resultados_cv[cols_mostrar].copy()
        resultados_display.columns = ['Parámetros', 'Score Promedio', 'Desviación Estándar', 'Ranking']
        resultados_display = resultados_display.sort_values('Ranking')
        
        st.dataframe(resultados_display, use_container_width=True)
        
        # Visualizaciones
        st.subheader("📊 Visualización de Resultados")
        
        # Gráfico de scores
        visualizar_resultados_grid_search(resultados_cv, param_grid)
        
        # Comparación de validación cruzada
        st.subheader("🔄 Validación Cruzada del Mejor Modelo")
        
        cv_scores = cross_val_score(
            grid_search.best_estimator_,
            X_train,
            y_train,
            cv=cv_folds,
            scoring=scoring
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Score Promedio CV", f"{cv_scores.mean():.4f}")
        with col2:
            st.metric("Desviación Estándar CV", f"{cv_scores.std():.4f}")
        with col3:
            st.metric("Score Mínimo CV", f"{cv_scores.min():.4f}")
        
        # Gráfico de scores por fold
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(1, cv_folds + 1), cv_scores, color='skyblue', edgecolor='black')
        ax.axhline(y=cv_scores.mean(), color='r', linestyle='--', label=f'Promedio: {cv_scores.mean():.4f}')
        ax.set_xlabel('Fold')
        ax.set_ylabel('Score')
        ax.set_title(f'Scores de Validación Cruzada ({cv_folds}-fold)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
        
        # Guardar mejor modelo en sesión
        st.session_state['best_tuned_model'] = grid_search.best_estimator_
        st.session_state['best_params'] = grid_search.best_params_
        
        st.success("💾 Mejor modelo guardado en la sesión")
        
    except Exception as e:
        st.error(f"❌ Error durante Grid Search: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

def visualizar_resultados_grid_search(resultados_cv, param_grid):
    """Visualizar resultados del grid search"""
    
    # Encontrar el parámetro más variado
    param_principal = max(param_grid.keys(), key=lambda k: len(param_grid[k]))
    
    if len(param_grid) == 1:
        # Solo un parámetro
        fig, ax = plt.subplots(figsize=(12, 6))
        
        valores = [str(p[param_principal]) for p in resultados_cv['params']]
        scores = resultados_cv['mean_test_score']
        
        ax.plot(range(len(valores)), scores, 'o-', linewidth=2, markersize=8)
        ax.set_xticks(range(len(valores)))
        ax.set_xticklabels(valores, rotation=45, ha='right')
        ax.set_xlabel(param_principal)
        ax.set_ylabel('Score Promedio')
        ax.set_title(f'Rendimiento vs {param_principal}')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
    elif len(param_grid) == 2:
        # Dos parámetros - heatmap
        params = list(param_grid.keys())
        
        # Crear pivot table
        resultados_pivot = resultados_cv[['params', 'mean_test_score']].copy()
        resultados_pivot['param1'] = resultados_pivot['params'].apply(lambda x: str(x[params[0]]))
        resultados_pivot['param2'] = resultados_pivot['params'].apply(lambda x: str(x[params[1]]))
        
        pivot_table = resultados_pivot.pivot_table(
            values='mean_test_score',
            index='param1',
            columns='param2',
            aggfunc='mean'
        )
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax)
        ax.set_xlabel(params[1])
        ax.set_ylabel(params[0])
        ax.set_title('Heatmap de Rendimiento')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
    else:
        # Múltiples parámetros - gráfico de barras
        fig, ax = plt.subplots(figsize=(14, 6))
        
        top_n = min(20, len(resultados_cv))
        top_resultados = resultados_cv.nsmallest(top_n, 'rank_test_score')
        
        indices = range(top_n)
        scores = top_resultados['mean_test_score']
        
        bars = ax.bar(indices, scores, color='skyblue', edgecolor='black')
        bars[0].set_color('green')
        
        ax.set_xlabel('Configuración (ordenado por ranking)')
        ax.set_ylabel('Score Promedio')
        ax.set_title(f'Top {top_n} Configuraciones')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
