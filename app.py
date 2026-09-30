import numpy as np

# --- Calibrated parameters (see notebook analysis) ---
STD_MINIMA_AFFIDABILE = 0.03
NDVI_HIGH_THRESHOLD = 0.929   # 75th percentile, whole nursery
NDVI_LOW_THRESHOLD = 0.752    # 25th percentile, whole nursery
SOGLIA_STRESS = -1.5
SOGLIA_OTTIMALE = 1.0


def compute_block_health(block_plant_ids, inventory):
    """Compute NDVI-based health status for each plant in a block, using
    z-score relative to the block's own mean/std."""
    df = inventory[inventory['plant_id'].isin(block_plant_ids)].copy()
    mean_ndvi = df['ndvi'].mean()
    std_ndvi = df['ndvi'].std()

    df['ndvi_zscore'] = (df['ndvi'] - mean_ndvi) / (std_ndvi + 1e-6)

    def classify(z):
        if z < SOGLIA_STRESS:
            return 'stressed'
        elif z > SOGLIA_OTTIMALE:
            return 'vigorous'
        return 'normal'

    df['health_status'] = df['ndvi_zscore'].apply(classify)
    return df, mean_ndvi, std_ndvi


def explain_ndvi_range(mean_ndvi, min_ndvi, max_ndvi):
    """Plain-language explanation of where this block's NDVI sits."""
    if mean_ndvi > NDVI_HIGH_THRESHOLD:
        level = "high"
        meaning = "most plants in this block look vigorous and densely leaved"
    elif mean_ndvi < NDVI_LOW_THRESHOLD:
        level = "low"
        meaning = "many plants in this block show signs of reduced vigor or possible stress"
    else:
        level = "moderate"
        meaning = "this block shows a typical, mixed range — some healthier plants, some less so"

    return (f"NDVI values in this block range from **{min_ndvi:.2f}** to **{max_ndvi:.2f}** "
            f"(higher values mean denser, healthier-looking foliage). "
            f"Compared to the rest of the nursery, this is a **{level}** range — {meaning}.")


# --- UI ---
st.header("Block Health Overview")

available_blocks = sorted(block_mapping['blocco'].unique())
selected_block = st.selectbox("Select a block", available_blocks)

block_ids = block_mapping[block_mapping['blocco'] == selected_block]['plant_id']
block_health, mean_ndvi, std_ndvi = compute_block_health(block_ids, inventory)

col1, col2 = st.columns([2, 1])

with col1:
    st.write(explain_ndvi_range(mean_ndvi, block_health['ndvi'].min(), block_health['ndvi'].max()))

    if std_ndvi < STD_MINIMA_AFFIDABILE:
        st.warning(
            "⚠️ **Low-confidence warning**: the plants in this block have very similar "
            "NDVI values to each other (little natural variation). This makes it harder "
            "to reliably tell which plants truly stand out — small, perfectly normal "
            "differences between plants might be flagged as a warning by mistake. "
            "We recommend using the additional temperature check below before drawing "
            "conclusions about any specific plant."
        )

with col2:
    counts = block_health['health_status'].value_counts()
    st.metric("Plants in block", len(block_health))
    st.write(f"🟢 Vigorous: {counts.get('vigorous', 0)}")
    st.write(f"🟡 Normal: {counts.get('normal', 0)}")
    st.write(f"🔴 Possibly stressed: {counts.get('stressed', 0)}")

st.dataframe(block_health[['plant_id', 'ndvi', 'ndvi_zscore', 'health_status']].sort_values('ndvi_zscore'))
