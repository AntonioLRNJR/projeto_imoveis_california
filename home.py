import geopandas as gpd   # Biblioteca para trabalhar com dados geoespaciais
import numpy as np        # Biblioteca para operações numéricas e vetoriais
import pandas as pd       # Biblioteca para manipulação de dados em DataFrames
import pydeck as pdk      # Biblioteca para visualização de mapas interativos
import shapely            # Biblioteca para manipulação de geometrias espaciais
import streamlit as st    # Framework para criação de aplicações web interativas

from joblib import load   # Utilitário para carregar modelos treinados salvos

# Importa variáveis de configuração (caminhos dos dados e modelo)
from notebooks.src.config import DADOS_GEO_MEDIAN, DADOS_LIMPOS, MODELO_FINAL


@st.cache_data
def carregar_dados_limpos():
    """
    Carrega os dados limpos a partir de um arquivo Parquet.
    Retorna:
        DataFrame do pandas contendo os dados processados.
    """
    return pd.read_parquet(DADOS_LIMPOS)


@st.cache_data
def carregar_dados_geo():
    """
    Carrega dados geoespaciais, corrige geometrias inválidas e extrai coordenadas dos polígonos.

    Passos:
    - Lê arquivo parquet com informações geográficas.
    - Explode MultiPolygons em polígonos individuais.
    - Corrige geometrias inválidas (buffer(0)).
    - Orienta polígonos no sentido anti-horário.
    - Extrai coordenadas dos polígonos para visualização no pydeck.

    Retorna:
        GeoDataFrame processado com coluna 'geometry' contendo coordenadas.
    """
    gdf_geo = gpd.read_parquet(DADOS_GEO_MEDIAN)

    # Divide multipolígonos em polígonos individuais
    gdf_geo = gdf_geo.explode(ignore_index=True)

    # Função para corrigir geometrias inválidas e orientar polígonos
    def fix_and_orient_geometry(geometry):
        if not geometry.is_valid:
            geometry = geometry.buffer(0)  # Corrige geometria inválida
        if isinstance(
            geometry, (shapely.geometry.Polygon, shapely.geometry.MultiPolygon)
        ):
            geometry = shapely.geometry.polygon.orient(geometry, sign=1.0)
        return geometry

    # Aplica correção nas geometrias
    gdf_geo["geometry"] = gdf_geo["geometry"].apply(fix_and_orient_geometry)

    # Função para extrair coordenadas dos polígonos
    def get_polygon_coordinates(geometry):
        return (
            [[[x, y] for x, y in geometry.exterior.coords]]
            if isinstance(geometry, shapely.geometry.Polygon)
            else [
                [[x, y] for x, y in polygon.exterior.coords]
                for polygon in geometry.geoms
            ]
        )

    # Aplica extração das coordenadas
    gdf_geo["geometry"] = gdf_geo["geometry"].apply(get_polygon_coordinates)

    return gdf_geo


@st.cache_resource
def carregar_modelo():
    """
    Carrega o modelo de Machine Learning salvo em disco.
    Retorna:
        Modelo treinado carregado via joblib.
    """
    return load(MODELO_FINAL)


# Carregamento dos dados e modelo
df = carregar_dados_limpos()
gdf_geo = carregar_dados_geo()
modelo = carregar_modelo()

# Título da aplicação Streamlit
st.title("Previsão de preços de imóveis")

# Lista de condados disponíveis (ordenada)
condados = sorted(gdf_geo["name"].unique())

# Divide a tela em duas colunas
coluna1, coluna2 = st.columns(2)

with coluna1:

    # Cria formulário de entrada de dados
    with st.form(key="formulario"):

        # Caixa de seleção para escolher o condado
        selecionar_condado = st.selectbox("Condado", condados)

        # Busca coordenadas do condado selecionado
        longitude = gdf_geo.query("name == @selecionar_condado")["longitude"].values
        latitude = gdf_geo.query("name == @selecionar_condado")["latitude"].values

        # Input para idade do imóvel
        housing_median_age = st.number_input(
            "Idade do imóvel", value=10, min_value=1, max_value=50
        )

        # Busca valores referentes ao condado selecionado
        total_rooms = gdf_geo.query("name == @selecionar_condado")["total_rooms"].values
        total_bedrooms = gdf_geo.query("name == @selecionar_condado")[
            "total_bedrooms"
        ].values
        population = gdf_geo.query("name == @selecionar_condado")["population"].values
        households = gdf_geo.query("name == @selecionar_condado")["households"].values

        # Slider para renda média
        median_income = st.slider(
            "Renda média (milhares de US$)", 5.0, 100.0, 45.0, 5.0
        )

        # Escala da renda
        median_income_scale = median_income / 10

        # Proximidade ao oceano
        ocean_proximity = gdf_geo.query("name == @selecionar_condado")[
            "ocean_proximity"
        ].values

        # Categoriza renda em faixas
        bins_income = [0, 1.5, 3, 4.5, 6, np.inf]
        median_income_cat = np.digitize(median_income_scale, bins=bins_income)

        # Outras variáveis derivadas
        rooms_per_household = gdf_geo.query("name == @selecionar_condado")[
            "rooms_per_household"
        ].values
        bedrooms_per_room = gdf_geo.query("name == @selecionar_condado")[
            "bedrooms_per_room"
        ].values
        population_per_household = gdf_geo.query("name == @selecionar_condado")[
            "population_per_household"
        ].values

        # Monta dicionário com dados de entrada para o modelo
        entrada_modelo = {
            "longitude": longitude,
            "latitude": latitude,
            "housing_median_age": housing_median_age,
            "total_rooms": total_rooms,
            "total_bedrooms": total_bedrooms,
            "population": population,
            "households": households,
            "median_income": median_income_scale,
            "ocean_proximity": ocean_proximity,
            "median_income_cat": median_income_cat,
            "rooms_per_household": rooms_per_household,
            "bedrooms_per_room": bedrooms_per_room,
            "population_per_household": population_per_household,
        }

        # Converte em DataFrame
        df_entrada_modelo = pd.DataFrame(entrada_modelo)

        # Botão para submeter o formulário
        botao_previsao = st.form_submit_button("Prever preço")

    # Se usuário clicar no botão, faz previsão com o modelo
    if botao_previsao:
        preco = modelo.predict(df_entrada_modelo)
        st.metric(label="Preço previsto: (US$)", value=f"{preco[0][0]:.2f}")

with coluna2:

    # Define a visão inicial do mapa (latitude e longitude do condado escolhido)
    view_state = pdk.ViewState(
        latitude=float(latitude[0]),  # Conversão para float padrão
        longitude=float(longitude[0]),  # Conversão para float padrão
        zoom=5,
        min_zoom=5,
        max_zoom=15,
    )

    # Camada para exibir todos os condados
    polygon_layer = pdk.Layer(
        "PolygonLayer",
        data=gdf_geo[["name", "geometry"]],
        get_polygon="geometry",
        get_fill_color=[0, 0, 255, 100],  # Azul translúcido
        get_line_color=[255, 255, 255],   # Contorno branco
        get_line_width=50,
        pickable=True,
        auto_highlight=True,
    )

    # Seleciona o condado escolhido pelo usuário
    condado_selecionado = gdf_geo.query("name == @selecionar_condado")

    # Camada de destaque para o condado selecionado
    highlight_layer = pdk.Layer(
        "PolygonLayer",
        data=condado_selecionado[["name", "geometry"]],
        get_polygon="geometry",
        get_fill_color=[255, 0, 0, 100],  # Vermelho translúcido
        get_line_color=[0, 0, 0],         # Contorno preto
        get_line_width=500,
        pickable=True,
        auto_highlight=True,
    )

    # Tooltip (informação ao passar o mouse)
    tooltip = {
        "html": "<b>Condado:</b> {name}",
        "style": {"backgroundColor": "steelblue", "color": "white", "fontsize": "10px"},
    }

    # Criação do mapa interativo
    mapa = pdk.Deck(
        initial_view_state=view_state,
        map_style="light",
        layers=[polygon_layer, highlight_layer],
        tooltip=tooltip,
    )

    # Renderiza o mapa no Streamlit
    st.pydeck_chart(mapa)
