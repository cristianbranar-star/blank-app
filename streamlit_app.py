import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt  # <-- Importe agregado
import seaborn as sns            # <-- Importe agregado

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Predicciones Metrosalud",
    page_icon="🏥",
    layout="wide"
)

# --- Título y Descripción ---
st.title("🏥 Proyecto Metrosalud - Primera Infancia")
st.markdown("""
Esta aplicación utiliza un modelo de Red Neuronal (IA) entrenado para 
predecir... *(Aquí debes completar el objetivo de tu modelo, ej: 'el riesgo de desnutrición', 'el estado de vacunación', etc.)*
""")

# --- Carga de Modelos y Pre-procesadores ---
# Usamos @st.cache_resource para cargar los modelos solo una vez

@st.cache_resource
def cargar_modelo():
    """Carga el modelo de Keras y los pre-procesadores."""
    try:
        modelo = load_model('modelo_primera_infancia.h5')
        scaler = joblib.load('scaler_X.pkl')
        encoder = joblib.load('encoder_y.pkl')
        return modelo, scaler, encoder
    except FileNotFoundError:
        st.error("Error Crítico: Faltan los archivos del modelo ('modelo_primera_infancia.h5' o 'scaler_X.pkl').")
        st.error("Asegúrate de haber subido los archivos .h5 y .pkl a tu repositorio de GitHub.")
        return None, None, None
    except Exception as e:
        st.error(f"Error al cargar los modelos: {e}")
        return None, None, None

modelo, scaler, encoder = cargar_modelo()

# --- Barra Lateral de Interacción (Inputs del Usuario) ---
st.sidebar.header("Ingresar Datos del Paciente")

if modelo is None:
    st.sidebar.error("La app no puede funcionar sin los archivos del modelo.")
else:
    # --- Formulario de Inputs ---
    # !!! IMPORTANTE: REEMPLAZA ESTO CON TUS VERDADERAS CARACTERÍSTICAS (FEATURES) !!!
    # El orden y tipo de dato debe ser EXACTAMENTE el mismo que usaste para entrenar.
    
    st.sidebar.write("Debes reemplazar estos campos de ejemplo:")
    
    # Ejemplo Característica 1 (Numérica)
    edad_meses = st.sidebar.slider("Edad (meses)", min_value=0, max_value=60, value=24, help="Edad del infante en meses.")
    
    # Ejemplo Característica 2 (Numérica)
    peso_kg = st.sidebar.number_input("Peso (kg)", min_value=1.0, max_value=25.0, value=10.0, step=0.1)
    
    # Ejemplo Característica 3 (Categórica)
    vacunas = st.sidebar.selectbox("Esquema de Vacunación", ['Completo', 'Incompleto', 'No Aplica'])
    
    # Ejemplo Característica 4 (Numérica)
    talla_cm = st.sidebar.number_input("Talla (cm)", min_value=40.0, max_value=120.0, value=75.0)
    
    # Botón para predecir
    submit_button = st.sidebar.button("Realizar Predicción", type="primary")

# --- Lógica de Predicción ---
if submit_button and modelo is not None:
    st.header("Resultado de la Predicción", divider='rainbow')
    
    try:
        # 1. Crear el DataFrame de entrada para el pre-procesamiento
        # Debe tener los MISMOS NOMBRES DE COLUMNAS que tu X_train original
        
        # !!! REEMPLAZA ESTO !!!
        # Crea un diccionario con los nombres de columna correctos
        input_data_dict = {
            'col_edad': [edad_meses],
            'col_peso': [peso_kg],
            'col_vacunas': [vacunas],
            'col_talla': [talla_cm]
            # ... añade todas tus columnas ...
        }
        
        input_df = pd.DataFrame(input_data_dict)
        st.write("Datos de entrada (pre-procesamiento):")
        st.dataframe(input_df)

        # 2. Pre-procesar los datos
        # Esta es la razón por la que DEBES guardar tu scaler.
        # Asumiendo que usaste un ColumnTransformer o un pipeline.
        # Si escalaste todo, sería algo como:
        
        # --- (INICIO) Ejemplo de Pre-procesamiento ---
        # Esto es muy específico de tu notebook, debes adaptarlo.
        # Supongamos que 'col_edad', 'col_peso', 'col_talla' eran numéricas
        # y 'col_vacunas' era categórica.
        
        # Separar columnas numéricas y categóricas (ejemplo)
        # Esto es solo un EJEMPLO. Debes usar tu lógica de scaler/encoder
        
        # Aplicar el scaler a los datos numéricos de entrada
        # Asumiendo que el scaler se ajustó a ['col_edad', 'col_peso', 'col_talla']
        # datos_numericos = input_df[['col_edad', 'col_peso', 'col_talla']]
        # datos_numericos_scaled = scaler.transform(datos_numericos)
        
        # Aplicar el encoder a los datos categóricos de entrada
        # (Si usaste One-Hot, es más complejo y es mejor usar un Pipeline)
        # (Por simplicidad, asumiremos que tu scaler los procesa todos o que
        # tu modelo puede manejar diferentes tipos, lo cual es raro)
        
        # *** Simulación de escalado simple ***
        # Es más probable que tu scaler espere un array de todas las features
        # en un orden específico.
        
        # Ejemplo:
        # 1. Convertir 'vacunas' a número (ej. LabelEncoding manual)
        # input_df['col_vacunas'] = input_df['col_vacunas'].map({'Completo': 2, 'Incompleto': 1, 'No Aplica': 0})
        
        # 2. Crear el array de numpy en el orden correcto
        # features_para_scaler = input_df[['col_edad', 'col_peso', 'col_talla', 'col_vacunas']].values
        
        # 3. Aplicar scaler
        # features_scaled = scaler.transform(features_para_scaler)
        
        # --- (FIN) Ejemplo de Pre-procesamiento ---

        # DADO QUE NO PUEDO SABER TU PRE-PROCESAMIENTO, USARÉ UN SIMULADOR
        # ¡¡¡ DEBES REEMPLAZAR ESTA LÍNEA !!!
        features_scaled = np.random.rand(1, modelo.input_shape[1])
        st.warning("Advertencia: Usando datos de predicción simulados. Debes conectar tu lógica de pre-procesamiento (scaler/encoder) aquí.")
        

        # 3. Realizar la predicción
        prediccion_prob = modelo.predict(features_scaled)
        
        # 4. Interpretar el resultado
        # Si es clasificación multiclase (softmax), obtén la clase con mayor prob.
        clase_predicha_idx = np.argmax(prediccion_prob, axis=1)[0]
        
        # Usar el encoder de 'y' para obtener la etiqueta original
        # Asumiendo que 'encoder' es el encoder de 'y' (la variable objetivo)
        etiqueta_predicha = encoder.categories_[0][clase_predicha_idx]

        st.success(f"**Predicción del Modelo:** {etiqueta_predicha}")
        
        st.write("Probabilidades (debug):")
        st.dataframe(pd.DataFrame(prediccion_prob, columns=encoder.categories_[0]))

    except Exception as e:
        st.error(f"Error durante la predicción: {e}")
        st.error("Verifica que tu lógica de pre-procesamiento (scaler/encoder) en `streamlit_app.py` sea idéntica a la de tu notebook.")


# --- SECCIÓN DE ANÁLISIS EXPLORATORIO (GRÁFICOS) ---
# Aquí es donde integramos los gráficos de tu notebook.
st.header("Análisis Exploratorio del Proyecto", divider='rainbow')
st.markdown("""
Aquí puedes mostrar los gráficos de Matplotlib/Seaborn de tu notebook 
para dar contexto a los resultados de la predicción.
""")

# --- Cargador de datos para análisis ---
@st.cache_data
def cargar_datos_analisis(archivo_csv):
    """Carga el CSV para los gráficos de análisis."""
    df = pd.read_csv(archivo_csv)
    return df

# --- Gráfico de Ejemplo 1 ---
st.subheader("Gráfico de Ejemplo: Distribución de Edad")
st.markdown("Pega aquí el código de tus gráficos del notebook.")

# Debes subir tu archivo CSV de análisis al repositorio de GitHub
# y poner el nombre aquí.
nombre_archivo_csv = 'datos_analisis_metrosalud.csv' # <-- CAMBIA ESTO

try:
    df_analisis = cargar_datos_analisis(nombre_archivo_csv)
    
    # --- Pega tu código de gráfico aquí ---
    # Ejemplo (debes reemplazar 'col_edad' por tu columna real)
    fig, ax = plt.subplots()
    if 'col_edad' in df_analisis.columns:
        sns.histplot(df_analisis['col_edad'], kde=True, ax=ax, bins=20)
        ax.set_title('Distribución de Edad de Pacientes')
        ax.set_xlabel('Edad (meses)')
        ax.set_ylabel('Frecuencia')
        st.pyplot(fig) # <-- Este comando "integra" el gráfico en Streamlit
    else:
        st.warning(f"La columna 'col_edad' no se encontró en {nombre_archivo_csv}. Mostrando datos del CSV:")
        st.dataframe(df_analisis.head())

    # --- Puedes añadir más gráficos ---
    # st.subheader("Gráfico 2: ...")
    # fig2, ax2 = plt.subplots()
    # ... (tu código de seaborn/matplotlib) ...
    # st.pyplot(fig2)


except FileNotFoundError:
    st.error(f"Error: No se encontró el archivo de datos '{nombre_archivo_csv}'.")
    st.error(f"Por favor, sube tu archivo CSV de análisis a tu repositorio de GitHub para que los gráficos funcionen.")
except Exception as e:
    st.error(f"Error al cargar o graficar los datos: {e}")
