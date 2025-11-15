import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Constructor de Dashboards",
    page_icon="🚀",
    layout="wide"
)

# --- Estado de Sesión ---
# Usamos session_state para almacenar el DataFrame cargado
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_name' not in st.session_state:
    st.session_state.df_name = ""

# --- Función de Carga de Datos ---
@st.cache_data
def load_data(file):
    """Carga datos desde un archivo CSV o Excel."""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file, engine='openpyxl')
        else:
            st.error("Formato de archivo no soportado. Por favor, sube un CSV o XLSX.")
            return None
        return df
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")
        return None

# --- Barra Lateral (Sidebar) ---
st.sidebar.header("Configuración del Dashboard")

# 1. Carga de Archivo
uploaded_file = st.sidebar.file_uploader(
    "Sube tu archivo (CSV o XLSX)", 
    type=["csv", "xlsx"]
)

if uploaded_file:
    # Si se sube un nuevo archivo, actualiza el estado de sesión
    if uploaded_file.name != st.session_state.df_name:
        st.session_state.df = load_data(uploaded_file)
        st.session_state.df_name = uploaded_file.name
        st.sidebar.success(f"¡Archivo '{uploaded_file.name}' cargado!")

# --- Cuerpo Principal del Dashboard ---
st.title("🚀 Constructor de Dashboards Interactivo")
st.markdown("Sube tus datos y crea tus propios tableros de análisis al instante.")

if st.session_state.df is None:
    st.info("Por favor, sube un archivo CSV o XLSX para comenzar.")
else:
    # Si hay datos cargados, mostramos los tableros
    df = st.session_state.df

    # 2. Filtros Globales (en la barra lateral)
    st.sidebar.header("Filtros Globales")
    
    # Identificar columnas categóricas y numéricas
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Crear filtros para columnas categóricas
    global_filters = {}
    for col in categorical_cols:
        if df[col].nunique() < 50: # Solo crear filtros para columnas con < 50 valores únicos
            options = df[col].unique().tolist()
            selected = st.sidebar.multiselect(f"Filtrar por {col}", options, default=options)
            global_filters[col] = selected

    # Aplicar filtros
    df_filtered = df.copy()
    for col, selected_values in global_filters.items():
        df_filtered = df_filtered[df_filtered[col].isin(selected_values)]

    # --- Creación de Tabs para los Dashboards ---
    tab_informativo, tab_operativo, tab_analitico, tab_estrategico = st.tabs(
        ["📄 Informativo", "📈 Operativo (KPIs)", "📊 Analítico", "♟️ Estratégico"]
    )

    # --- Tab 1: Informativo (Datos Crudos y Resumen) ---
    with tab_informativo:
        st.header("Tablero Informativo: Vista de Datos")
        st.subheader(f"Datos de: {st.session_state.df_name} (Filtrados)")
        st.dataframe(df_filtered)
        
        st.subheader("Resumen Estadístico (Datos Numéricos)")
        if numerical_cols:
            st.dataframe(df_filtered[numerical_cols].describe())
        else:
            st.info("No se encontraron columnas numéricas para el resumen.")

    # --- Tab 2: Operativo (KPIs) ---
    with tab_operativo:
        st.header("Tablero Operativo: Métricas Clave (KPIs)")
        st.markdown("Selecciona una columna numérica para ver sus KPIs.")
        
        if not numerical_cols:
            st.warning("No hay columnas numéricas para calcular KPIs.")
        else:
            kpi_col = st.selectbox("Selecciona una métrica para KPIs", numerical_cols, index=0)
            
            # Calcular KPIs
            total_sum = df_filtered[kpi_col].sum()
            average = df_filtered[kpi_col].mean()
            max_val = df_filtered[kpi_col].max()
            min_val = df_filtered[kpi_col].min()
            count = df_filtered[kpi_col].count()
            
            # Mostrar KPIs en columnas
            col1, col2, col3 = st.columns(3)
            col1.metric("Suma Total", f"{total_sum:,.2f}")
            col2.metric("Promedio", f"{average:,.2f}")
            col3.metric("Conteo de Registros", f"{count:,}")
            
            col4, col5 = st.columns(2)
            col4.metric("Valor Máximo", f"{max_val:,.2f}")
            col5.metric("Valor Mínimo", f"{min_val:,.2f}")

    # --- Tab 3: Analítico (Distribuciones y Relaciones) ---
    with tab_analitico:
        st.header("Tablero Analítico: Exploración Visual")
        
        if not numerical_cols:
            st.warning("Se necesitan columnas numéricas para estos gráficos.")
        else:
            # Gráfico de Distribución (Histograma)
            st.subheader("Análisis de Distribución")
            hist_col = st.selectbox("Columna para Histograma", numerical_cols, index=0)
            fig_hist = px.histogram(df_filtered, x=hist_col, title=f"Distribución de {hist_col}", nbins=50)
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Gráfico de Dispersión (Relación)
            st.subheader("Análisis de Relación (Gráfico de Dispersión)")
            if len(numerical_cols) >= 2:
                col_x = st.selectbox("Selecciona Eje X", numerical_cols, index=0)
                col_y = st.selectbox("Selecciona Eje Y", numerical_cols, index=1)
                color_col = st.selectbox("Selecciona Columna para Color (Opcional)", [None] + categorical_cols)
                
                fig_scatter = px.scatter(df_filtered, x=col_x, y=col_y, color=color_col,
                                         title=f"Relación entre {col_x} y {col_y}")
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("Se necesitan al menos dos columnas numéricas para un gráfico de dispersión.")

    # --- Tab 4: Estratégico (Agregaciones) ---
    with tab_estrategico:
        st.header("Tablero Estratégico: Análisis Agregado")
        st.markdown("Agrupa tus datos para ver tendencias de alto nivel.")
        
        if not categorical_cols or not numerical_cols:
            st.warning("Se necesita al menos una columna categórica y una numérica para este análisis.")
        else:
            group_col = st.selectbox("Agrupar por (Dimensión)", categorical_cols, index=0)
            metric_col = st.selectbox("Calcular (Métrica)", numerical_cols, index=0)
            agg_func = st.selectbox("Función de Agregación", ["sum", "mean", "count"])
            
            # Realizar la agregación
            try:
                if agg_func == 'sum':
                    df_agg = df_filtered.groupby(group_col)[metric_col].sum().reset_index()
                elif agg_func == 'mean':
                    df_agg = df_filtered.groupby(group_col)[metric_col].mean().reset_index()
                elif agg_func == 'count':
                    df_agg = df_filtered.groupby(group_col)[metric_col].count().reset_index()
                
                # Graficar resultado
                fig_bar = px.bar(df_agg, x=group_col, y=metric_col, 
                                 title=f"{agg_func.capitalize()} de {metric_col} por {group_col}",
                                 color=group_col)
                st.plotly_chart(fig_bar, use_container_width=True)
                
                st.subheader("Datos Agregados")
                st.dataframe(df_agg)
                
            except Exception as e:
                st.error(f"Error al agregar datos: {e}")
