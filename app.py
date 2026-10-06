import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import json
from shapely.geometry import shape

st.set_page_config(page_title="Nursery - Exploratory Demo", layout="wide")

ORTHO_SCALE = 4.228

STD_MINIMA_AFFIDABILE = 0.03
NDVI_HIGH_THRESHOLD = 0.929
NDVI_LOW_THRESHOLD = 0.752
CWSI_STRESS_THRESHOLD = 0.5

STATUS_COLORS = {'vigorous': '#2ecc71', 'normal': '#f1c40f', 'stressed': '#e74c3c',
                 'not assessed': '#95a5a6'}
CLASS_EN = {'stressata': 'stressed', 'normale': 'normal', 'vigorosa': 'vigorous'}
RELIABILITY_EN = {'alta': 'High', 'media': 'Medium', 'bassa': 'Low'}

# Plants with confidence <= UNCERTAIN_MAX are "uncertain" (their verdict was recomputed on the
# sunlit half of the canopy); plants at 75% are "moderately uncertain" (verdict kept as is).
UNCERTAIN_MAX = 50
MODERATE_MAX = 75


@st.cache_data
def load_data():
    inventory = pd.read_csv("inventario_app.csv")
    block_mapping = pd.read_csv("mappatura_blocchi.csv")

    with open("contorni_tutte_piante.geojson", encoding="utf-8-sig") as f:
        geojson_data = json.load(f)
    contours = pd.DataFrame([
        {'plant_id': feat['properties']['plant_id'], 'geometry': shape(feat['geometry'])}
        for feat in geojson_data['features']
    ])
    contours['plant_id'] = contours['plant_id'].astype(int)

    preview = np.array(Image.open("preview_rgb.png"))
    with open("block_boundaries.json", encoding="utf-8-sig") as f:
        block_boundaries = {k: np.array(v) for k, v in json.load(f).items()}

    # Final verdict per plant (output of the shading analysis, blocks 1-4)
    verdict = pd.read_csv("verdetto_finale_4_blocchi.csv")
    verdict['plant_id'] = verdict['plant_id'].astype(int)
    verdict['status_original'] = verdict['classe_originale'].map(CLASS_EN)
    verdict['status_bright'] = verdict['classe_pixel_luminosi'].map(CLASS_EN)
    verdict['status_final'] = verdict['classe_finale'].map(CLASS_EN)
    verdict['uncertainty'] = np.select(
        [verdict['confidenza'] <= UNCERTAIN_MAX, verdict['confidenza'] <= MODERATE_MAX],
        ['high', 'moderate'], default='none')
    verdict = verdict[['plant_id', 'status_original', 'status_bright', 'status_final', 'uncertainty',
                       'confidenza', 'affidabilita', 'distanza_soglia', 'borderline_solo_a_75']]
    return inventory, block_mapping, contours, preview, block_boundaries, verdict


@st.cache_data
def compute_physical_thresholds(inventory):
    """25th/75th percentiles across the whole nursery, used to describe a
    plant's canopy area/diameter as small/medium/large."""
    return {
        'area_m2': (inventory['area_m2'].quantile(0.25), inventory['area_m2'].quantile(0.75)),
        'diametro_chioma_m': (inventory['diametro_chioma_m'].quantile(0.25),
                                inventory['diametro_chioma_m'].quantile(0.75)),
    }


def size_label(value, low, high):
    if pd.isna(value):
        return None
    if value < low:
        return "small"
    elif value > high:
        return "large"
    return "medium-sized"


def build_plant_summary(prow, thresholds):
    """Plain-language sentence describing this plant's canopy size."""
    area_label = size_label(prow['area_m2'], *thresholds['area_m2'])
    if area_label:
        return (f"This plant has a **{area_label} canopy** "
                f"compared to other plants in the nursery.")
    return "Not enough data to describe this plant's overall size."


def build_block_health(block_plant_ids, inventory, verdict):
    """NDVI statistics of the block + final health verdict of each plant (from the shading analysis)."""
    df = inventory[inventory['plant_id'].isin(block_plant_ids)][['plant_id', 'ndvi', 'tir_mean']].merge(
        verdict, on='plant_id', how='left')
    n_not_assessed = int(df['status_final'].isna().sum())
    df['status_final'] = df['status_final'].fillna('not assessed')
    df['uncertainty'] = df['uncertainty'].fillna('none')
    return df, df['ndvi'].mean(), df['ndvi'].std(), n_not_assessed


def explain_block_variability(mean_ndvi, std_ndvi, min_ndvi, max_ndvi, low_confidence):
    if mean_ndvi > NDVI_HIGH_THRESHOLD:
        level, meaning = "high", "most plants in this block look vigorous and densely leaved"
    elif mean_ndvi < NDVI_LOW_THRESHOLD:
        level, meaning = "low", "many plants in this block show signs of reduced vigor or possible stress"
    else:
        level, meaning = "moderate", "this block shows a typical, mixed range — some healthier plants, some less so"

    text = (f"Overall, this block sits in a **{level}** vigor range — {meaning}. "
            f"(Values observed span from {min_ndvi:.2f} to {max_ndvi:.2f} on our internal scale.)\n\n")

    if low_confidence:
        text += (
            "⚠️ **However, take individual differences within this block with caution.** "
            "The plants here are unusually similar to one another — there isn't much natural "
            "variation to compare against. This means that even small, perfectly normal "
            "differences between two plants might look like a warning sign when they aren't. "
            "We recommend using the temperature check below before concluding that any single "
            "plant is under stress."
        )
    else:
        text += (
            "✅ There's enough natural variation among plants in this block to make "
            "reliable comparisons — if a plant is flagged as stressed, it genuinely stands "
            "out from the others here."
        )
    return text


def compute_cwsi(t_plant, t_ref_healthy, t_dry):
    if pd.isna(t_plant):
        return None
    cwsi = (t_plant - t_ref_healthy) / (t_dry - t_ref_healthy)
    return float(np.clip(cwsi, 0, 1))


def _centroids(subset, c_min_p, r_min_p):
    cx, cy, ids, hover = [], [], [], []
    for _, row in subset.iterrows():
        c = row['geometry'].centroid
        cx.append(c.x / ORTHO_SCALE - c_min_p)
        cy.append(c.y / ORTHO_SCALE - r_min_p)
        ids.append(int(row['plant_id']))
        hover.append(f"Plant {int(row['plant_id'])}")
    return cx, cy, ids, hover


def build_zoom_figure(selected_block, block_boundaries, preview_img, color_by_status=False,
                       block_contours=None, show_moderate=False):
    pad = 0.15
    rows_full = block_boundaries[selected_block][:, 0]
    cols_full = block_boundaries[selected_block][:, 1]
    r_min = rows_full.min() - (rows_full.max() - rows_full.min()) * pad
    r_max = rows_full.max() + (rows_full.max() - rows_full.min()) * pad
    c_min = cols_full.min() - (cols_full.max() - cols_full.min()) * pad
    c_max = cols_full.max() + (cols_full.max() - cols_full.min()) * pad
    r_min_p, r_max_p = r_min / ORTHO_SCALE, r_max / ORTHO_SCALE
    c_min_p, c_max_p = c_min / ORTHO_SCALE, c_max / ORTHO_SCALE

    crop = preview_img[max(int(r_min_p), 0):int(r_max_p), max(int(c_min_p), 0):int(c_max_p)]

    fig = go.Figure()
    fig.add_trace(go.Image(z=crop))

    if color_by_status:
        groups = {status: block_contours[block_contours['status_final'] == status]
                  for status in STATUS_COLORS}
        color_map = STATUS_COLORS
    else:
        groups = {'plant': block_contours}
        color_map = {'plant': '#3498db'}

    for status, subset in groups.items():
        if subset.empty:
            continue
        color = color_map[status]
        xs, ys = [], []
        for _, row in subset.iterrows():
            coords = np.array(row['geometry'].exterior.coords) / ORTHO_SCALE
            xs.extend((coords[:, 0] - c_min_p).tolist() + [None])
            ys.extend((coords[:, 1] - r_min_p).tolist() + [None])
        cx, cy, ids, hover = _centroids(subset, c_min_p, r_min_p)

        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='lines', fill='toself',
            line=dict(color=color, width=1), fillcolor=color, opacity=0.5,
            hoverinfo='skip', showlegend=color_by_status, name=status.capitalize()
        ))
        fig.add_trace(go.Scatter(
            x=cx, y=cy, mode='markers',
            marker=dict(size=10, color=color, line=dict(color='black', width=1)),
            customdata=ids, hovertext=hover, hoverinfo='text', showlegend=False
        ))

    # Uncertainty markers: a ring around the plant's marker
    if color_by_status:
        rings = [('high', 'Uncertain verdict', 22, 3)]
        if show_moderate:
            rings.append(('moderate', 'Somewhat uncertain', 18, 1.5))
        for level, label, size, width in rings:
            subset = block_contours[block_contours['uncertainty'] == level]
            if subset.empty:
                continue
            cx, cy, ids, hover = _centroids(subset, c_min_p, r_min_p)
            fig.add_trace(go.Scatter(
                x=cx, y=cy, mode='markers',
                marker=dict(symbol='circle', size=size, color='rgba(0,0,0,0)',
                            line=dict(color='black' if level == 'high' else '#555555', width=width)),
                customdata=ids, hoverinfo='skip', showlegend=True, name=label
            ))

    fig.update_layout(
        height=550, margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor='x', autorange='reversed'),
        legend=dict(orientation='h', y=-0.05),
        clickmode='event+select'
    )
    return fig


def extract_selected_id(event):
    if event and event.get("selection", {}).get("points"):
        point = event["selection"]["points"][0]
        if "customdata" in point:
            cd = point["customdata"]
            return cd[0] if isinstance(cd, (list, tuple)) else cd
    return None


def describe_uncertainty(prow):
    """Plain-language box about how stable this plant's verdict is with respect to canopy shading."""
    orig, bright, final = prow['status_original'], prow['status_bright'], prow['status_final']
    level = prow['uncertainty']
    if level == 'high':
        msg = (f"⚠️ **Uncertain verdict.** Inside the canopy of this plant, the sunlit and the shaded "
               f"leaves give different results. Using all the canopy the plant is **{orig}**; "
               f"using only its sunlit half it is **{bright}**.")
        if final != orig:
            msg += f" We therefore report it as **{final}**, but the verdict should be taken with caution."
        else:
            msg += f" Both versions agree on **{final}** for the final verdict, but it is close to a class boundary."
        st.warning(msg)
    elif level == 'moderate':
        msg = ("🔸 **Fairly stable verdict.** In 3 of the 4 light levels inside the canopy the plant gets the "
               "same status; in the remaining one it is classified differently.")
        if bool(prow['borderline_solo_a_75']):
            msg += " This plant is close to a class boundary, so it may switch with a small change in the data."
        st.info(msg)
    else:
        st.success("✅ **Reliable verdict.** The plant gets the same status in all four light levels inside its canopy "
                   "(from the darkest to the brightest part).")


# --- Load data ---
inventory, block_mapping, contours, preview_img, block_boundaries, verdict = load_data()
T_DRY_GLOBAL = inventory['tir_mean'].quantile(0.99)
PHYSICAL_THRESHOLDS = compute_physical_thresholds(inventory)

st.title("🌿 Nursery — Exploratory Demo")

available_blocks = [b for b in sorted(block_mapping['blocco'].unique()) if b in block_boundaries]
skipped_blocks = sorted(set(block_mapping['blocco'].unique()) - set(available_blocks))
if skipped_blocks:
    st.warning(f"Blocks without a boundary in block_boundaries.json are hidden: {', '.join(map(str, skipped_blocks))}")
selected_block = st.selectbox("Select a block", available_blocks)

block_ids = block_mapping[block_mapping['blocco'] == selected_block]['plant_id']
block_health, mean_ndvi, std_ndvi, n_not_assessed = build_block_health(block_ids, inventory, verdict)
low_confidence = std_ndvi < STD_MINIMA_AFFIDABILE
counts = block_health['status_final'].value_counts()
n_uncertain = int((block_health['uncertainty'] == 'high').sum())

block_contours = contours[contours['plant_id'].isin(block_ids)].merge(
    block_health[['plant_id', 'ndvi', 'status_final', 'uncertainty', 'tir_mean']], on='plant_id'
)

if "plant_inventory" not in st.session_state:
    st.session_state.plant_inventory = None
if "plant_health" not in st.session_state:
    st.session_state.plant_health = None

tab1, tab2 = st.tabs(["🌳 Physical Inventory", "🩺 Health Status"])

# ============================================================
# TAB 1 — physical characteristics
# ============================================================
with tab1:
    st.metric("Total plants in this block", len(block_health))

    col_map, col_zoom = st.columns([1, 1])
    with col_map:
        st.subheader("Nursery overview")
        boundary = block_boundaries[selected_block] / ORTHO_SCALE
        bx = boundary[:, 1].tolist() + [boundary[0, 1]]
        by = boundary[:, 0].tolist() + [boundary[0, 0]]
        fig_overview = go.Figure()
        fig_overview.add_trace(go.Image(z=preview_img))
        fig_overview.add_trace(go.Scatter(
            x=bx, y=by, mode='lines', fill='toself',
            line=dict(color='white', width=2), fillcolor='rgba(255,255,255,0.15)',
            hoverinfo='skip', showlegend=False
        ))
        fig_overview.update_layout(
            height=450, margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor='x', autorange='reversed')
        )
        st.plotly_chart(fig_overview, use_container_width=True, key="overview_tab1")

    with col_zoom:
        st.subheader(f"Zoom on {selected_block} — click a plant")
        fig_zoom_inv = build_zoom_figure(selected_block, block_boundaries, preview_img,
                                           color_by_status=False, block_contours=block_contours)
        event_inv = st.plotly_chart(fig_zoom_inv, use_container_width=True,
                                      on_select="rerun", key="zoom_inventory")
        pid = extract_selected_id(event_inv)
        if pid is not None:
            st.session_state.plant_inventory = pid

    if st.session_state.plant_inventory is not None:
        pid = st.session_state.plant_inventory
        prow = inventory[inventory['plant_id'] == pid]
        if not prow.empty:
            prow = prow.iloc[0]
            st.markdown("---")
            st.subheader(f"Plant {pid} — physical characteristics")
            st.write(build_plant_summary(prow, PHYSICAL_THRESHOLDS))

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Canopy area", f"{prow['area_m2']:.2f} m²")
                st.caption("The ground area covered by this plant's foliage, seen from above.")
            with c2:
                st.metric("Crown diameter", f"{prow['diametro_chioma_m']:.2f} m")
                st.caption("The width of the canopy — how wide the plant spreads.")

# ============================================================
# TAB 2 — health status
# ============================================================
with tab2:
    st.subheader(f"Health overview — {selected_block}")
    st.markdown(explain_block_variability(mean_ndvi, std_ndvi,
                                            block_health['ndvi'].min(), block_health['ndvi'].max(),
                                            low_confidence))

    st.markdown("---")
    col_map2, col_zoom2 = st.columns([1, 1])
    with col_map2:
        st.subheader("Block status at a glance")
        st.write(f"🟢 Vigorous: {counts.get('vigorous', 0)}  ·  "
                 f"🟡 Normal: {counts.get('normal', 0)}  ·  "
                 f"🔴 Possibly stressed: {counts.get('stressed', 0)}")
        st.write(f"⭕ Uncertain verdict (depends on light inside the canopy): **{n_uncertain}**")
        if n_not_assessed:
            st.caption(f"{n_not_assessed} plants in this block could not be assessed (too few canopy pixels).")
        show_moderate = st.checkbox("Also mark somewhat uncertain plants (3 of 4 light levels agree)", value=False)
        with st.expander("How is the uncertainty computed?"):
            st.markdown(
                "Parts of a canopy are lit and parts are in shade, and shade changes the colour signal. "
                "For every plant we split its canopy into **four light levels** (darkest to brightest quarter) and "
                "recompute the health status in each. If the status is the same in all four, the verdict is reliable. "
                "If it changes in at least half of them, the plant is marked as **uncertain** (black ring) and its "
                "status is recomputed using only the **sunlit half** of the canopy. "
                "The other plants keep the verdict computed on the whole canopy."
            )
        boundary = block_boundaries[selected_block] / ORTHO_SCALE
        bx = boundary[:, 1].tolist() + [boundary[0, 1]]
        by = boundary[:, 0].tolist() + [boundary[0, 0]]
        fig_overview2 = go.Figure()
        fig_overview2.add_trace(go.Image(z=preview_img))
        fig_overview2.add_trace(go.Scatter(
            x=bx, y=by, mode='lines', fill='toself',
            line=dict(color='white', width=2), fillcolor='rgba(255,255,255,0.15)',
            hoverinfo='skip', showlegend=False
        ))
        fig_overview2.update_layout(
            height=450, margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor='x', autorange='reversed')
        )
        st.plotly_chart(fig_overview2, use_container_width=True, key="overview_tab2")

    with col_zoom2:
        st.subheader(f"Zoom on {selected_block} — click a plant")
        fig_zoom_health = build_zoom_figure(selected_block, block_boundaries, preview_img,
                                              color_by_status=True, block_contours=block_contours,
                                              show_moderate=show_moderate)
        event_health = st.plotly_chart(fig_zoom_health, use_container_width=True,
                                         on_select="rerun", key="zoom_health")
        pid = extract_selected_id(event_health)
        if pid is not None:
            st.session_state.plant_health = pid

    if st.session_state.plant_health is not None:
        pid = st.session_state.plant_health
        prow = block_health[block_health['plant_id'] == pid]
        if not prow.empty:
            prow = prow.iloc[0]
            st.markdown("---")
            st.subheader(f"Plant {pid} — health status")
            c1, c2, c3 = st.columns(3)
            c1.metric("Status", prow['status_final'].capitalize())
            c2.metric("Leaf temperature", f"{prow['tir_mean']:.1f}°C" if pd.notna(prow['tir_mean']) else "n/a")
            c3.metric("Verdict reliability", RELIABILITY_EN.get(prow['affidabilita'], "n/a"))

            if prow['status_final'] == 'not assessed':
                st.info("This plant could not be assessed (too few canopy pixels).")
            else:
                describe_uncertainty(prow)

            st.markdown("---")
            st.subheader("🌡️ Additional check: leaf temperature")
            st.write(
                "If you know the typical leaf temperature of a **healthy** plant of this "
                "species, enter it below. We'll use it to double-check the result with an "
                "independent method based on temperature, instead of leaf color."
            )
            ref_temp = st.number_input(
                "Average leaf temperature of a healthy plant of this species (°C)",
                min_value=0.0, max_value=50.0, value=None, step=0.1,
                placeholder="e.g. 22.5", key="ref_temp_input"
            )

            if ref_temp is not None:
                cwsi = compute_cwsi(prow['tir_mean'], ref_temp, T_DRY_GLOBAL)
                if cwsi is None:
                    st.info(f"No temperature data available for plant {pid} — cannot run this check.")
                else:
                    cwsi_says_stressed = cwsi > CWSI_STRESS_THRESHOLD
                    ndvi_says_stressed = prow['status_final'] == 'stressed'
                    st.metric("Water stress index (CWSI)", f"{cwsi:.2f}",
                              help="0 = no water stress, 1 = maximum water stress")

                    if cwsi_says_stressed == ndvi_says_stressed:
                        if ndvi_says_stressed:
                            st.error("✅ Both checks agree: this plant shows signs of stress "
                                     "in both leaf color and temperature. Worth a closer look.")
                        else:
                            st.success("✅ Both checks agree: this plant looks healthy, both "
                                       "in leaf color and temperature.")
                    else:
                        st.warning(
                            f"⚠️ The two checks disagree: leaf color suggests "
                            f"**{'stress' if ndvi_says_stressed else 'no stress'}**, while "
                            f"temperature suggests **{'stress' if cwsi_says_stressed else 'no stress'}**. "
                            f"This might mean the stress is only just beginning to show in one "
                            f"signal but not the other yet, or that the reference temperature "
                            f"you entered isn't fully representative for this block."
                        )
    else:
        st.info("Click a plant on the map above to see its health status and run the temperature check.")
