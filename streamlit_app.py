import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import io

# --- Configuración de la Página ---
# Usamos layout="wide" para que el dashboard ocupe toda la pantalla
st.set_page_config(layout="wide", page_title="Analizador Estratégico de Datos")

# --- Funciones de Carga y Procesamiento ---

# Usamos @st.cache_data para que streamlit guarde en caché el archivo cargado.
# Si el usuario interactúa con un widget, el archivo no se vuelve a cargar,
# haciendo la app mucho más rápida.
@st.cache_data
def load_data(file, sample_rows=None):
    """Carga datos desde un archivo CSV o XLSX, con opción de muestreo."""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, nrows=sample_rows)
        elif file.name.endswith(('.xls', '.xlsx')):
            # 'openpyxl' es necesario para leer .xlsx
            df = pd.read_excel(file, engine='openpyxl', nrows=sample_rows)
        else:
            st.error("Formato de archivo no soportado. Use .csv o .xlsx")
            return None
        
        # Convertir nombres de columnas a string (para evitar errores)
        df.columns = df.columns.astype(str)
        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        return None

@st.cache_data
def calculate_feature_importance(df, target_variable):
    """Calcula la importancia de variables usando RandomForest."""
    
    # 1. Crear una copia para no modificar el original en caché
    df_processed = df.copy()

    # 2. Separar X (features) e y (target)
    if target_variable not in df_processed.columns:
        st.error("La variable objetivo seleccionada no se encuentra.")
        return None
        
    y = df_processed[target_variable]
    X = df_processed.drop(columns=[target_variable])

    # 3. Preprocesamiento simple
    imputer_num = SimpleImputer(strategy='median')
    imputer_cat = SimpleImputer(strategy='most_frequent')
    le = LabelEncoder()

    numeric_cols = X.select_dtypes(include=['number']).columns
    categorical_cols = X.select_dtypes(exclude=['number']).columns

    # Imputar y codificar
    if not numeric_cols.empty:
        X[numeric_cols] = imputer_num.fit_transform(X[numeric_cols])
        
    for col in categorical_cols:
        X[col] = imputer_cat.fit_transform(X[[col]])
        X[col] = le.fit_transform(X[col])

    # 4. Entrenar modelo (solo si 'y' es numérica)
    if pd.api.types.is_numeric_dtype(y):
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)
        
        # 5. Obtener importancia
        importance = pd.Series(model.feature_importances_, index=X.columns)
        return importance.sort_values(ascending=False)
    else:
        st.warning("La variable objetivo debe ser numérica para calcular la importancia (Regresión).")
        return None

# --- Layout de la Aplicación ---

st.title("📊 Tablero de Control Estratégico")
st.write("Cargue su archivo .csv o .xlsx para analizar los datos y la importancia de las variables.")

# --- Barra Lateral (Sidebar) ---
st.sidebar.header("1. Carga de Datos")
uploaded_file = st.sidebar.file_uploader("Seleccione un archivo", type=["csv", "xlsx"])

sample_check = st.sidebar.checkbox("Analizar solo las primeras 50,000 filas")
sample_size = 50000 if sample_check else None

st.sidebar.markdown("""
**Nota sobre Archivos Pesados:**
* Los archivos grandes pueden tardar en procesarse.
* Streamlit Community Cloud tiene RAM limitada. Si su app falla, use la opción de **muestreo**.
""")

# --- Cuerpo Principal del Dashboard ---
if uploaded_file is not None:
    df = load_data(uploaded_file, sample_size)

    if df is not None:
        st.success(f"¡Archivo cargado exitosamente! Se muestran {len(df)} filas.")
        
        # Crear pestañas para el dashboard
        tab1, tab2, tab3, tab4 = st.tabs([
            "Resumen General", 
            "Análisis de Variables", 
            "Matriz de Correlación", 
            "Importancia Estratégica"
        ])

        # --- Pestaña 1: Resumen General ---
        with tab1:
            st.header("Resumen General del Conjunto de Datos")
            
            # Mostrar un 'head' del dataframe
            st.subheader("Vista Previa de los Datos")
            st.dataframe(df.head())

            col1, col2 = st.columns(2)
            
            with col1:
                # Estadísticas Descriptivas
                st.subheader("Estadísticas Descriptivas")
                st.dataframe(df.describe(include='all').T)

            with col2:
                # Información de Tipos de Datos y Nulos
                st.subheader("Tipos de Datos y Valores Faltantes")
                # Capturamos la salida de df.info() en un buffer de texto
                buffer = io.StringIO()
                df.info(buf=buffer)
                s = buffer.getvalue()
                st.text(s)

        # --- Pestaña 2: Análisis de Variables ---
        with tab2:
            st.header("Análisis Univariable y Bivariable")
            st.write("Seleccione variables para explorar su distribución y relación.")

            col1, col2 = st.columns(2)
            
            with col1:
                # Selección de variable para Histograma
                st.subheader("Distribución (Histograma)")
                var_hist = st.selectbox("Seleccione una variable numérica", 
                                        options=df.select_dtypes(include=['number']).columns, 
                                        key='hist')
                if var_hist:
                    fig_hist = px.histogram(df, x=var_hist, nbins=50, title=f"Distribución de {var_hist}")
                    st.plotly_chart(fig_hist, use_container_width=True)

            with col2:
                # Selección de variables para Scatter plot
                st.subheader("Relación (Scatter Plot)")
                var_scatter_x = st.selectbox("Seleccione variable Eje X", 
                                             options=df.columns, 
                                             key='scatter_x')
                var_scatter_y = st.selectbox("Seleccione variable Eje Y", 
                                             options=df.columns, 
                                             key='scatter_y')
                if var_scatter_x and var_scatter_y:
                    fig_scatter = px.scatter(df.sample(min(1000, len(df))), # Usamos muestra para no sobrecargar
                                             x=var_scatter_x, 
                                             y=var_scatter_y, 
                                             title=f"Relación entre {var_scatter_x} y {var_scatter_y}")
                    st.plotly_chart(fig_scatter, use_container_width=True)

        # --- Pestaña 3: Matriz de Correlación ---
        with tab3:
            st.header("Matriz de Correlación")
            st.write("Muestra la correlación lineal (Pearson) entre las variables **numéricas**.")
            
            try:
                corr_matrix = df.corr(numeric_only=True)
                fig_corr = px.imshow(corr_matrix, 
                                     text_auto=True, 
                                     aspect="auto",
                                     title="Matriz de Correlación")
                st.plotly_chart(fig_corr, use_container_width=True)
            except Exception as e:
                st.warning(f"No se pudo calcular la matriz de correlación. Error: {e}")

        # --- Pestaña 4: Importancia Estratégica ---
        with tab4:
            st.header("Importancia de Variables (Feature Importance)")
            st.info("""
            Esta sección utiliza un modelo de Machine Learning (Random Forest) para determinar 
            qué variables influyen más en una **variable objetivo**.
            
            **Debe seleccionar una variable objetivo NUMÉRICA (ej. 'Ventas', 'Costos', 'Puntaje').**
            """)

            # Selección de la variable objetivo
            numeric_cols = df.select_dtypes(include=['number']).columns
            if not numeric_cols.empty:
                target_var = st.selectbox("Seleccione su variable objetivo (Y)", options=numeric_cols)

                if st.button(f"Calcular Importancia para '{target_var}'"):
                    with st.spinner("Entrenando modelo y calculando..."):
                        importance_series = calculate_feature_importance(df, target_var)
                        
                        if importance_series is not None:
                            # Crear un DataFrame para Plotly
                            importance_df = pd.DataFrame({
                                'Variable': importance_series.index,
                                'Importancia': importance_series.values
                            })
                            
                            # Graficar
                            fig_imp = px.bar(importance_df.sort_values(by='Importancia', ascending=True), 
                                             x='Importancia', 
                                             y='Variable', 
                                             orientation='h',
                                             title=f"Variables más importantes que influyen en '{target_var}'")
                            st.plotly_chart(fig_imp, use_container_width=True)
                        else:
                            st.error("No se pudo calcular la importancia. Asegúrese de que la variable objetivo sea numérica.")
            else:
                st.warning("No se encontraron variables numéricas para seleccionar como objetivo.")

else:
    st.info("Esperando a que se cargue un archivo .csv o .xlsx...")
