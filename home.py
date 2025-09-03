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


condados = list(gdf_geo["name"].sort_values())

selecionar_condado = st.selectbox("Condado", condados)

# longitude/latitude do imóvel
longitude = gdf_geo.query("name == @selecionar_condado")["longitude"].values
latitude = gdf_geo.query("name == @selecionar_condado")["latitude"].values

# Entrada do usuário: idade mediana do imóvel
housing_median_age = st.number_input("Idade do imóvel", value=10, min_value=1, max_value=50)

# número total de cômodos no distrito
total_rooms = gdf_geo.query("name == @selecionar_condado")["total_rooms"].values
# número total de quartos no distrito
total_bedrooms = gdf_geo.query("name == @selecionar_condado")["total_bedrooms"].values
# população total no distrito
population = gdf_geo.query("name == @selecionar_condado")["population"].values
# Entrada do usuário: número de domicílios no distrito
households = gdf_geo.query("name == @selecionar_condado")["households"].values

# Entrada do usuário: renda mediana (em múltiplos de US$ 10k), usando um slider para melhor interação
median_income = st.slider("Renda média (milhares de US$)", 5.0, 100.0, 45.0, 5.0)

# Entrada do usuário: proximidade ao oceano (variável categórica), com opções baseadas nos dados existentes
ocean_proximity = gdf_geo.query("name == @selecionar_condado")["ocean_proximity"].values

# Entrada do usuário: categoria de renda mediana (possivelmente usada no modelo como feature categórica)
bins_income = [0, 1.5, 3, 4.5, 6, np.inf]
median_income_cat = np.digitize(median_income / 10, bins=bins_income)

# Entrada do usuário: número médio de quartos por domicílio
rooms_per_household = gdf_geo.query("name == @selecionar_condado")["rooms_per_household"].values
# Entrada do usuário: proporção de quartos em relação ao total de cômodos
bedrooms_per_room = gdf_geo.query("name == @selecionar_condado")["bedrooms_per_room"].values
# Entrada do usuário: número médio de pessoas por domicílio
population_per_household = gdf_geo.query("name == @selecionar_condado")["population_per_household"].values

# Agrupa todas as entradas do usuário em um dicionário para formar uma única observação
entrada_modelo = {
    "longitude": longitude,
    "latitude": latitude,
    "housing_median_age": housing_median_age,
    "total_rooms": total_rooms,
    "total_bedrooms": total_bedrooms,
    "population": population,
    "households": households,
    "median_income": median_income / 10,
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