import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st

from joblib import load

# Importa configurações de caminhos de arquivos (dados limpos, dados geoespaciais e modelo treinado)
from notebooks.src.config import DADOS_GEO_MEDIAN, DADOS_LIMPOS, MODELO_FINAL


# Armazena em cache o carregamento dos dados limpos para evitar recarregamento desnecessário
@st.cache_data
def carregar_dados_limpos():
    return pd.read_parquet(DADOS_LIMPOS)


# Armazena em cache o carregamento dos dados geoespaciais (GeoParquet)
@st.cache_data
def carregar_dados_geo():
    return gpd.read_parquet(DADOS_GEO_MEDIAN)


# Armazena em cache o modelo carregado, pois é um recurso pesado e deve ser reutilizado
@st.cache_resource
def carregar_modelo():
    return load(MODELO_FINAL)


# Carrega os dados limpos, os dados geográficos e o modelo previamente treinado
df = carregar_dados_limpos()
gdf_geo = carregar_dados_geo()
modelo = carregar_modelo()


# Título da aplicação Streamlit
st.title("Previsão de preços de imóveis")

# Entrada do usuário: longitude do imóvel
longitude = st.number_input("Longitude", value=-122.33)

# Entrada do usuário: latitude do imóvel
latitude = st.number_input("Latitude", value=37.88)

# Entrada do usuário: idade mediana do imóvel
housing_median_age = st.number_input("Idade do imóvel", value=10)

# Entrada do usuário: número total de cômodos no distrito
total_rooms = st.number_input("Total de cômodos", value=800)

# Entrada do usuário: número total de quartos no distrito
total_bedrooms = st.number_input("Total de quartos", value=100)

# Entrada do usuário: população total no distrito
population = st.number_input("População", value=300)

# Entrada do usuário: número de domicílios no distrito
households = st.number_input("Domicílios", value=100)

# Entrada do usuário: renda mediana (em múltiplos de US$ 10k), usando um slider para melhor interação
median_income = st.slider("Renda média (múltiplos de US$ 10k)", 0.5, 15.0, 4.5, 0.5)

# Entrada do usuário: proximidade ao oceano (variável categórica), com opções baseadas nos dados existentes
ocean_proximity = st.selectbox("Proximidade do oceano", df["ocean_proximity"].unique())

# Entrada do usuário: categoria de renda mediana (possivelmente usada no modelo como feature categórica)
median_income_cat = st.number_input("Categoria de renda", value=4)

# Entrada do usuário: número médio de quartos por domicílio
rooms_per_household = st.number_input("Quartos por domicílio", value=7)

# Entrada do usuário: proporção de quartos em relação ao total de cômodos
bedrooms_per_room = st.number_input("Quartos por cômodo", value=0.2)

# Entrada do usuário: número médio de pessoas por domicílio
population_per_household = st.number_input("Pessoas por domicílio", value=2)

# Agrupa todas as entradas do usuário em um dicionário para formar uma única observação
entrada_modelo = {
    "longitude": longitude,
    "latitude": latitude,
    "housing_median_age": housing_median_age,
    "total_rooms": total_rooms,
    "total_bedrooms": total_bedrooms,
    "population": population,
    "households": households,
    "median_income": median_income,
    "ocean_proximity": ocean_proximity,
    "median_income_cat": median_income_cat,
    "rooms_per_household": rooms_per_household,
    "bedrooms_per_room": bedrooms_per_room,
    "population_per_household": population_per_household,
}

# Converte o dicionário de entrada em um DataFrame do pandas (necessário para o modelo)
df_entrada_modelo = pd.DataFrame(entrada_modelo, index=[0])

# Botão na interface para acionar a previsão
botao_previsao = st.button("Prever preço")

# Se o botão for clicado, realiza a previsão com o modelo carregado
if botao_previsao:
    preco = modelo.predict(df_entrada_modelo)  # Faz a previsão com base nos dados de entrada
    st.write(f"Preço previsto: US$ {preco[0][0]:.2f}")  # Exibe o preço previsto com formatação