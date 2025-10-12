import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import io

def mostrar_carga_datos():
    """Mostrar interfaz para carga y procesamiento de datos"""
    st.header("📂 Carga y Procesamiento de Datos")
    
    tab_cargar, tab_generar = st.tabs(["📤 Subir Archivo", "🎲 Generar Dataset Aleatorio"])
    
    with tab_cargar:
        # Subir archivo
        uploaded_file = st.file_uploader(
            "Sube tu archivo de datos",
            type=['csv', 'xlsx', 'xls'],
            help="Formatos soportados: CSV, Excel (.xlsx, .xls)"
        )
        
        if uploaded_file is not None:
            try:
                df = leer_archivo(uploaded_file)
                st.session_state.dataset = df
                st.success(f"✅ Archivo cargado exitosamente: {uploaded_file.name}")
                mostrar_info_y_procesamiento(df)
            except Exception as e:
                st.error(f"❌ Error al procesar el archivo: {str(e)}")
        else:
            st.info("📁 Por favor, sube un archivo CSV o Excel para comenzar")
            mostrar_ejemplo()
    
    with tab_generar:
        st.info("Genera un dataset sintético para pruebas rápidas (por ejemplo, rendimiento de automóviles)")
        df_gen = generar_dataset_aleatorio_ui()
        if df_gen is not None:
            st.session_state.dataset = df_gen
            st.success("✅ Dataset aleatorio generado y cargado en sesión")
            mostrar_info_y_procesamiento(df_gen)

def leer_archivo(uploaded_file):
    """Detección automática y lectura de CSV/Excel con manejo de separadores comunes"""
    nombre = uploaded_file.name.lower()
    if nombre.endswith('.csv'):
        # Intentar con separadores típicos
        contenido = uploaded_file.read()
        # Necesario para reusar el buffer
        buffer = io.BytesIO(contenido)
        # Probar con coma, punto y coma, tab
        for sep in [',', ';', '\t', '|']:
            try:
                buffer.seek(0)
                df = pd.read_csv(buffer, sep=sep)
                if df.shape[1] > 1 or sep == ',':
                    return df
            except Exception:
                continue
        buffer.seek(0)
        return pd.read_csv(buffer)  # fallback
    elif nombre.endswith(('.xlsx', '.xls')):
        uploaded_file.seek(0)
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Formato de archivo no soportado")

def mostrar_info_y_procesamiento(df):
    """Sección reusable para mostrar info y botones de procesamiento"""
    # Mostrar información básica del dataset
    st.subheader("📊 Información General del Dataset")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Filas", df.shape[0])
    with col2:
        st.metric("Columnas", df.shape[1])
    with col3:
        st.metric("Memoria (KB)", f"{df.memory_usage(deep=True).sum() / 1024:.1f}")
    # Mostrar primeras filas
    st.subheader("🔍 Primeras Filas del Dataset")
    st.dataframe(df.head(10), use_container_width=True)
    # Información de columnas
    st.subheader("📋 Información de Columnas")
    info_df = pd.DataFrame({
        'Columna': df.columns,
        'Tipo de Dato': df.dtypes.astype(str),
        'Valores Nulos': df.isnull().sum(),
        'Porcentaje Nulos': (df.isnull().sum() / len(df) * 100).round(2)
    })
    st.dataframe(info_df, use_container_width=True)
    # Procesamiento de datos
    st.subheader("⚙️ Procesamiento de Datos")
    valores_nulos = df.isnull().sum().sum()
    if valores_nulos > 0:
        st.warning(f"⚠️ Se encontraron {valores_nulos} valores nulos en el dataset")
        if st.button("🔧 Rellenar Valores Nulos con la Media", type="primary"):
            df_procesado = procesar_valores_nulos(df)
            st.session_state.processed_dataset = df_procesado
            st.success("✅ Valores nulos rellenados con la media")
            nuevos_nulos = df_procesado.isnull().sum().sum()
            if nuevos_nulos == 0:
                st.success("✅ Verificado: No quedan valores nulos en el dataset")
            else:
                st.warning(f"⚠️ Aún quedan {nuevos_nulos} valores nulos")
            st.subheader("📊 Dataset Procesado")
            st.dataframe(df_procesado.head(10), use_container_width=True)
    else:
        st.success("✅ No se encontraron valores nulos en el dataset")
        st.session_state.processed_dataset = df
    # Procesamiento de variables categóricas
    if st.session_state.processed_dataset is not None:
        mostrar_procesamiento_categoricas(st.session_state.processed_dataset)

def mostrar_ejemplo():
    with st.expander("💡 Ejemplo de formato de datos esperado"):
        ejemplo_df = pd.DataFrame({
            'mpg': [18.0, 15.0, 18.0, 16.0, 17.0],
            'cylinders': [8, 8, 8, 8, 8],
            'displacement': [307.0, 350.0, 318.0, 304.0, 302.0],
            'horsepower': [130.0, 165.0, 150.0, 150.0, 140.0],
            'weight': [3504, 3693, 3436, 3433, 3449],
            'acceleration': [12.0, 11.5, 11.0, 12.0, 10.5],
            'model_year': [70, 70, 70, 70, 70],
            'origin': ['usa', 'usa', 'usa', 'usa', 'usa'],
            'car_name': ['chevrolet chevelle malibu', 'buick skylark 320', 'plymouth satellite', 'amc rebel sst', 'ford torino']
        })
        st.dataframe(ejemplo_df, use_container_width=True)

def generar_dataset_aleatorio_ui():
    """UI para generar un dataset sintético tipo automóviles"""
    st.subheader("🎲 Parámetros del Dataset Aleatorio")
    n = st.slider("Número de filas", min_value=50, max_value=5000, value=200, step=50)
    incluir_nulos = st.checkbox("Introducir valores nulos aleatorios", value=True)
    proporcion_nulos = st.slider("Proporción de nulos (%)", min_value=0, max_value=50, value=5, step=1)
    random_state = st.number_input("Semilla aleatoria", min_value=0, value=42, step=1)
    if st.button("🚀 Generar Dataset"):
        df_gen = generar_dataset_aleatorio(n, incluir_nulos, proporcion_nulos/100.0, int(random_state))
        st.subheader("🔍 Vista Previa del Dataset Generado")
        st.dataframe(df_gen.head(15), use_container_width=True)
        return df_gen
    return None

def generar_dataset_aleatorio(n=200, incluir_nulos=True, prop_nulos=0.05, random_state=42):
    """Genera un dataset sintético inspirado en automóviles"""
    rng = np.random.default_rng(random_state)
    cylinders = rng.choice([4, 6, 8], size=n, p=[0.5, 0.3, 0.2])
    displacement = rng.normal(200, 50, size=n).clip(70, 450)
    horsepower = rng.normal(120, 40, size=n).clip(50, 400)
    weight = rng.normal(3000, 500, size=n).clip(1500, 5500).astype(int)
    acceleration = rng.normal(12, 2, size=n).clip(7, 20)
    model_year = rng.integers(70, 83, size=n)
    origin = rng.choice(['usa', 'europe', 'japan'], size=n, p=[0.6, 0.25, 0.15])
    car_name = [f"car_{i}" for i in range(n)]
    # mpg dependiente de otras variables con ruido
    mpg = (
        50 - 0.01*weight - 0.02*displacement - 0.03*horsepower + 0.2*acceleration + rng.normal(0, 2, size=n)
    ).clip(5, 60)
    df = pd.DataFrame({
        'mpg': mpg,
        'cylinders': cylinders,
        'displacement': displacement,
        'horsepower': horsepower,
        'weight': weight,
        'acceleration': acceleration,
        'model_year': model_year,
        'origin': origin,
        'car_name': car_name
    })
    if incluir_nulos and prop_nulos > 0:
        df = introducir_nulos_aleatorios(df, prop_nulos, rng)
    return df

def introducir_nulos_aleatorios(df, prop, rng):
    """Introduce nulos aleatorios por columna según una proporción"""
    df_out = df.copy()
    total = df_out.size
    n_nulos = int(total * prop)
    for _ in range(n_nulos):
        i = rng.integers(0, df_out.shape[0])
        j = rng.integers(0, df_out.shape[1])
        df_out.iat[i, j] = np.nan
    return df_out

def procesar_valores_nulos(df):
    """Procesar valores nulos rellenando con la media para columnas numéricas"""
    df_procesado = df.copy()
    
    # Para columnas numéricas, rellenar con la media
    columnas_numericas = df_procesado.select_dtypes(include=[np.number]).columns
    for columna in columnas_numericas:
        if df_procesado[columna].isnull().any():
            media = df_procesado[columna].mean()
            df_procesado[columna].fillna(media, inplace=True)
    
    # Para columnas categóricas, rellenar con la moda
    columnas_categoricas = df_procesado.select_dtypes(include=['object']).columns
    for columna in columnas_categoricas:
        if df_procesado[columna].isnull().any():
            moda = df_procesado[columna].mode()
            if len(moda) > 0:
                df_procesado[columna].fillna(moda[0], inplace=True)
            else:
                df_procesado[columna].fillna('Unknown', inplace=True)
    
    return df_procesado

def mostrar_procesamiento_categoricas(df):
    """Mostrar opciones para procesar variables categóricas y gestión de variables"""
    st.subheader("🏷️ Procesamiento de Variables Categóricas")
    
    # Tabs para diferentes operaciones
    tab1, tab2, tab3 = st.tabs(["🔄 Convertir a Numérico", "✏️ Editar Variables", "🗑️ Eliminar Variables"])
    
    with tab1:
        # Identificar columnas categóricas
        columnas_categoricas = df.select_dtypes(include=['object']).columns.tolist()
        
        if columnas_categoricas:
            st.info(f"Se encontraron {len(columnas_categoricas)} columnas categóricas: {', '.join(columnas_categoricas)}")
            
            # Seleccionar columnas a convertir
            cols_a_convertir = st.multiselect(
                "Selecciona las columnas categóricas a convertir:",
                columnas_categoricas,
                default=columnas_categoricas
            )
            
            if cols_a_convertir:
                # Seleccionar método de codificación
                metodo = st.selectbox(
                    "Selecciona el método de codificación:",
                    ["Label Encoding", "One-Hot Encoding (get_dummies)"]
                )
                
                if st.button("🔄 Aplicar Codificación", type="primary", key="btn_codificar"):
                    df_codificado = aplicar_codificacion(df, cols_a_convertir, metodo)
                    st.session_state.processed_dataset = df_codificado
                    
                    st.success(f"✅ Codificación {metodo} aplicada exitosamente")
                    
                    # Mostrar resultado
                    st.subheader("📊 Dataset con Variables Codificadas")
                    st.dataframe(df_codificado.head(10), use_container_width=True)
                    
                    # Mostrar información actualizada
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Columnas Originales", df.shape[1])
                    with col2:
                        st.metric("Columnas Después de Codificar", df_codificado.shape[1])
                    
                    # Verificar multicolinealidad para One-Hot Encoding
                    if metodo == "One-Hot Encoding (get_dummies)":
                        st.info("💡 Se ha aplicado drop_first=True para evitar multicolinealidad")
        else:
            st.success("✅ No se encontraron variables categóricas para procesar")
    
    with tab2:
        mostrar_edicion_variables(df)
    
    with tab3:
        mostrar_eliminacion_variables(df)
    
    # Botón para guardar datos procesados finales
    st.divider()
    st.subheader("💾 Guardar Datos Procesados")
    
    if st.session_state.get('processed_dataset') is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Filas", st.session_state.processed_dataset.shape[0])
        with col2:
            st.metric("Columnas", st.session_state.processed_dataset.shape[1])
        with col3:
            st.metric("Datos Nulos", st.session_state.processed_dataset.isnull().sum().sum())
        
        if st.button("💾 Guardar y Usar en Todos los Modelos", type="primary", key="btn_guardar_final"):
            # Guardar dataset procesado como el dataset principal
            st.session_state.dataset = st.session_state.processed_dataset.copy()
            st.success("✅ Datos procesados guardados exitosamente")
            st.info("💡 Estos datos se usarán automáticamente en todos los modelos (Regresión, Logística, Árboles de Decisión)")
            
            # Mostrar preview del dataset guardado
            with st.expander("👀 Ver datos guardados"):
                st.dataframe(st.session_state.dataset.head(20), use_container_width=True)
    else:
        st.info("📊 No hay datos procesados para guardar. Aplica primero algún procesamiento.")

def mostrar_edicion_variables(df):
    """Mostrar interfaz para editar variables"""
    st.write("### ✏️ Editar Variables")
    
    # Trabajar con el dataset procesado si existe, si no con el original
    df_trabajo = st.session_state.get('processed_dataset', df).copy()
    
    # Seleccionar columna a editar
    columna_editar = st.selectbox(
        "Selecciona la columna a editar:",
        df_trabajo.columns.tolist(),
        key="select_col_editar"
    )
    
    if columna_editar:
        col1, col2 = st.columns(2)
        
        with col1:
            # Opción 1: Renombrar columna
            st.write("**Renombrar Columna**")
            nuevo_nombre = st.text_input(
                "Nuevo nombre para la columna:",
                value=columna_editar,
                key="txt_nuevo_nombre"
            )
            
            if st.button("✏️ Renombrar", key="btn_renombrar"):
                if nuevo_nombre and nuevo_nombre != columna_editar:
                    df_trabajo.rename(columns={columna_editar: nuevo_nombre}, inplace=True)
                    st.session_state.processed_dataset = df_trabajo
                    st.success(f"✅ Columna '{columna_editar}' renombrada a '{nuevo_nombre}'")
                    st.rerun()
                else:
                    st.warning("⚠️ Ingresa un nombre diferente")
        
        with col2:
            # Opción 2: Reemplazar valores
            st.write("**Reemplazar Valores**")
            
            # Mostrar valores únicos si son pocos
            valores_unicos = df_trabajo[columna_editar].unique()
            if len(valores_unicos) <= 20:
                st.write(f"Valores únicos: {', '.join(map(str, valores_unicos[:10]))}")
                if len(valores_unicos) > 10:
                    st.write(f"... y {len(valores_unicos) - 10} más")
            
            valor_buscar = st.text_input("Valor a reemplazar:", key="txt_valor_buscar")
            valor_nuevo = st.text_input("Nuevo valor:", key="txt_valor_nuevo")
            
            if st.button("🔄 Reemplazar", key="btn_reemplazar"):
                if valor_buscar and valor_nuevo:
                    # Intentar conversión de tipo
                    try:
                        if df_trabajo[columna_editar].dtype in ['int64', 'float64']:
                            valor_buscar = float(valor_buscar)
                            valor_nuevo = float(valor_nuevo)
                    except:
                        pass
                    
                    cantidad = (df_trabajo[columna_editar] == valor_buscar).sum()
                    df_trabajo[columna_editar] = df_trabajo[columna_editar].replace(valor_buscar, valor_nuevo)
                    st.session_state.processed_dataset = df_trabajo
                    st.success(f"✅ Reemplazados {cantidad} valores de '{valor_buscar}' por '{valor_nuevo}'")
                    st.rerun()
                else:
                    st.warning("⚠️ Completa ambos campos")
        
        # Mostrar preview de la columna editada
        st.write("**Vista Previa de la Columna:**")
        st.dataframe(df_trabajo[[columna_editar]].head(10), use_container_width=True)

        # Gestión de categorías para evitar colinealidad
        if df_trabajo[columna_editar].dtype == 'object' or str(df_trabajo[columna_editar].dtype) == 'category':
            st.write("### 🧩 Gestionar Categorías (evitar colinealidad)")
            valores_unicos_cat = df_trabajo[columna_editar].astype(str).unique().tolist()
            st.write(f"Categorías detectadas: {', '.join(map(str, valores_unicos_cat[:10]))}" if len(valores_unicos_cat) <= 10 else f"Categorías detectadas: {', '.join(map(str, valores_unicos_cat[:10]))} ... (+{len(valores_unicos_cat)-10})")

            col_a, col_b = st.columns(2)

            with col_a:
                categorias_combinar = st.multiselect(
                    "Selecciona categorías a combinar:",
                    options=valores_unicos_cat,
                    key="multiselect_combinar_cat"
                )
                nuevo_label_cat = st.text_input("Nuevo nombre para las categorías seleccionadas:", key="txt_nuevo_label_cat")
                if st.button("🔗 Combinar categorías", key="btn_combinar_cat"):
                    if categorias_combinar and nuevo_label_cat:
                        df_trabajo[columna_editar] = df_trabajo[columna_editar].astype(str).replace({c: nuevo_label_cat for c in categorias_combinar})
                        st.session_state.processed_dataset = df_trabajo
                        st.success(f"✅ Combinadas {len(categorias_combinar)} categorías en '{nuevo_label_cat}'")
                        st.rerun()
                    else:
                        st.warning("⚠️ Selecciona categorías y define el nuevo nombre")

            with col_b:
                categorias_eliminar = st.multiselect(
                    "Selecciona categorías a eliminar (filtrar filas):",
                    options=valores_unicos_cat,
                    key="multiselect_eliminar_cat"
                )
                if st.button("🗑️ Eliminar categorías (filtrar)", key="btn_eliminar_cat"):
                    if categorias_eliminar:
                        filas_antes = df_trabajo.shape[0]
                        df_trabajo = df_trabajo[~df_trabajo[columna_editar].astype(str).isin(categorias_eliminar)]
                        filas_despues = df_trabajo.shape[0]
                        st.session_state.processed_dataset = df_trabajo
                        st.success(f"✅ Eliminadas {filas_antes - filas_despues} filas con categorías seleccionadas")
                        st.rerun()
                    else:
                        st.warning("⚠️ Selecciona al menos una categoría a eliminar")

def mostrar_eliminacion_variables(df):
    """Mostrar interfaz para eliminar variables"""
    st.write("### 🗑️ Eliminar Variables")
    
    # Trabajar con el dataset procesado si existe, si no con el original
    df_trabajo = st.session_state.get('processed_dataset', df).copy()
    
    # Seleccionar columnas a eliminar
    columnas_eliminar = st.multiselect(
        "Selecciona las columnas a eliminar:",
        df_trabajo.columns.tolist(),
        key="multiselect_eliminar"
    )
    
    if columnas_eliminar:
        st.warning(f"⚠️ Se eliminarán {len(columnas_eliminar)} columna(s): {', '.join(columnas_eliminar)}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Columnas Actuales", df_trabajo.shape[1])
        with col2:
            st.metric("Columnas Después de Eliminar", df_trabajo.shape[1] - len(columnas_eliminar))
        
        if st.button("🗑️ Eliminar Columnas Seleccionadas", type="secondary", key="btn_eliminar_cols"):
            df_trabajo = df_trabajo.drop(columns=columnas_eliminar)
            st.session_state.processed_dataset = df_trabajo
            st.success(f"✅ {len(columnas_eliminar)} columna(s) eliminada(s) exitosamente")
            st.rerun()
        
        # Mostrar preview sin las columnas a eliminar
        st.write("**Vista Previa (sin columnas a eliminar):**")
        df_preview = df_trabajo.drop(columns=columnas_eliminar)
        st.dataframe(df_preview.head(10), use_container_width=True)
    else:
        st.info("📝 Selecciona las columnas que deseas eliminar")

def aplicar_codificacion(df, columnas_categoricas, metodo):
    """Aplicar codificación a variables categóricas"""
    df_codificado = df.copy()
    
    if metodo == "Label Encoding":
        # Aplicar Label Encoding
        le = LabelEncoder()
        for columna in columnas_categoricas:
            df_codificado[columna] = le.fit_transform(df_codificado[columna].astype(str))
    
    elif metodo == "One-Hot Encoding (get_dummies)":
        # Aplicar One-Hot Encoding
        df_codificado = pd.get_dummies(df_codificado, columns=columnas_categoricas, drop_first=True)
    
    return df_codificado

def procesar_datos(df):
    """Función general para procesar datos"""
    # Esta función puede ser expandida según necesidades específicas
    df_procesado = procesar_valores_nulos(df)
    return df_procesado
