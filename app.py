import streamlit as st
import pandas as pd
import geopandas as gpd
from PIL import Image

st.set_page_config(page_title="Nursery - Exploratory Demo", layout="wide")

ORTHO_SCALE = 4.228  # scale factor: full-resolution pixels / preview pixels

@st.cache_data
def load_data():
    inventory = pd.read_csv("inventario_app.csv")
    block_mapping = pd.read_csv("mappatura_blocchi.csv")
    contours = gpd.read_file("contorni_tutte_piante.geojson")
    preview = Image.open("preview_rgb.png")
    return inventory, block_mapping, contours, preview

st.title("🌿 Nursery — Exploratory Demo")

inventory, block_mapping, contours, preview = load_data()

st.write(f"Loaded {len(inventory)} plants, {len(block_mapping)} plants across "
         f"{block_mapping['blocco'].nunique()} blocks, {len(contours)} plant contours.")
st.image(preview, caption="Nursery orthomosaic (preview)", use_container_width=True)
