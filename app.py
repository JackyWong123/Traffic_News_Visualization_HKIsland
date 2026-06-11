# app.py
import os
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
from traffic_processor import TrafficIncidentEngine

st.set_page_config(layout="wide", page_title="HKI Traffic Dashboard")
st.title("Traffic News Visualization for Hong Kong Island")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# RESOURCE INTERFACES & PIPELINE EXTRACTION
# ==========================================
@st.cache_data
def run_spatial_processing_pipeline():
    engine = TrafficIncidentEngine(
        road_path=os.path.join(BASE_DIR, "iG1000_HKI_FGDB", "hki_centerline.zip"),
        boundary_path=os.path.join(BASE_DIR, "iG1000_HKI_FGDB", "hki_boundary.zip"),
        building_path=os.path.join(BASE_DIR, "iG1000_HKI_FGDB", "hki_building.zip"),
        xml_path=os.path.join(BASE_DIR, "combined.xml"),
        distance_threshold=500
    )
    return engine.process_active_incidents()

df_incidents, gdf_spatial = run_spatial_processing_pipeline()

# ==========================================
# UI FEATURE 1: CLASSIFICATION DESIGN MAP
# ==========================================
CATEGORY_STYLES = {
    "Accident": {"color": "red", "icon": "car", "prefix": "fa"},
    "Road Works": {"color": "orange", "icon": "wrench", "prefix": "fa"},
    "Congestion": {"color": "amber", "icon": "clock-o", "prefix": "fa"},
    "Road Closure": {"color": "darkred", "icon": "ban", "prefix": "fa"},
    "General Alert": {"color": "blue", "icon": "exclamation-circle", "prefix": "fa"}
}

# Initialize session state memory to track clicked incidents across both widgets
if "selected_inc_id" not in st.session_state:
    st.session_state.selected_inc_id = None

# Calculate representation coordinates for mapping
if not df_incidents.empty and 'lat' not in df_incidents.columns:
    gdf_4326 = gdf_spatial.to_crs(epsg=4326)
    rep_points = []
    
    for inc_id in df_incidents['IncidentID']:
        inc_geom = gdf_4326[gdf_4326['IncidentID'] == inc_id]
        if not inc_geom.empty:
            point_feats = inc_geom[inc_geom.geometry.geom_type == 'Point']
            if not point_feats.empty:
                rep_pt = point_feats.iloc[0]['geometry']
            else:
                rep_pt = inc_geom.unary_union.centroid
            rep_points.append({'IncidentID': inc_id, 'lat': rep_pt.y, 'lng': rep_pt.x})
            
    if rep_points:
        df_rep = pd.DataFrame(rep_points)
        df_incidents = df_incidents.merge(df_rep, on='IncidentID', how='inner')

# Identify row index mapping to support programmatic table auto-selection
default_selection_idx = []
if st.session_state.selected_inc_id in df_incidents['IncidentID'].values:
    matched_row_idx = df_incidents[df_incidents['IncidentID'] == st.session_state.selected_inc_id].index[0]
    default_selection_idx = [int(matched_row_idx)]

# ==========================================
# INTERACTIVE SCREEN LAYOUT GENERATION
# ==========================================
col_table, col_map = st.columns([4, 5])

with col_table:
    st.subheader("Traffic News Log")
    st.caption("Select a row below or click an icon on the map canvas to project detailed street links.")
    
    selection = st.dataframe(
        df_incidents,
        use_container_width=True,
        hide_index=True,
        column_order=["IncidentID", "Category", "Location", "Details"],
        on_select="rerun",
        selection_mode="single-row",
        height=550, 
        # 🎯 THE FIX: Swapped out selection_rows for the correct dictionary format
        selection_default={"selection": {"rows": default_selection_idx}},
        column_config={
            "IncidentID": st.column_config.TextColumn("Case ID", width="small"),
            "Category": st.column_config.TextColumn("Type", width="medium"),
            "Location": st.column_config.TextColumn("Corridor Location Context", width="large"), 
            "Details": st.column_config.TextColumn("Full Descriptive Summary Log View", width="max")
        }
    )
    
    # Sync table selection events to the master controller state
    if selection and selection['selection']['rows']:
        row_idx = selection['selection']['rows'][0]
        table_selected_id = df_incidents.iloc[row_idx]['IncidentID']
        if st.session_state.selected_inc_id != table_selected_id:
            st.session_state.selected_inc_id = table_selected_id
            st.rerun()

with col_map:
    st.subheader("Map Visualization")
    
    # Base map setup
    m = folium.Map(location=[22.28552, 114.15769], zoom_start=12, tiles="cartodbpositron")
    
    # 🎯 STEP 1: Render categorical icon representations across HKI map layout
    for _, row in df_incidents.iterrows():
        is_current = (row['IncidentID'] == st.session_state.selected_inc_id)
        style = CATEGORY_STYLES.get(row['Category'], CATEGORY_STYLES["General Alert"])
        
        # Give the active selection a distinct visual outline glow
        icon_color = "white" if is_current else style["color"]
        bg_color = "darkpurple" if is_current else style["color"]
        
        folium.Marker(
            location=[row['lat'], row['lng']],
            icon=folium.Icon(color=bg_color, icon_color=icon_color, icon=style["icon"], prefix=style["prefix"]),
            tooltip=f"[{row['Category']}] ID: {row['IncidentID']} - {row['Location']}"
        ).add_to(m)
        
    # 🎯 STEP 2: Project detailed linear routes for selected cases
    if st.session_state.selected_inc_id:
        matched_shapes = gdf_spatial[gdf_spatial['IncidentID'] == st.session_state.selected_inc_id]
        
        if not matched_shapes.empty:
            matched_shapes_4326 = matched_shapes.to_crs(epsg=4326)
            
            for _, row in matched_shapes_4326.iterrows():
                geom = row['geometry']
                if geom is None or geom.geom_type == 'Point': 
                    continue
                    
                folium.GeoJson(
                    geom,
                    style_function=lambda x: {'color': '#E63946', 'weight': 7, 'opacity': 0.9}
                ).add_to(m)
            
            total_bounds = matched_shapes_4326.total_bounds
            m.fit_bounds([[total_bounds[1], total_bounds[0]], [total_bounds[3], total_bounds[2]]])
    else:
        if 'lat' in df_incidents.columns and not df_incidents.empty:
            m.fit_bounds([[df_incidents['lat'].min(), df_incidents['lng'].min()], 
                          [df_incidents['lat'].max(), df_incidents['lng'].max()]])

    # Render out map widget and listen for map-click coordinates
    map_data = st_folium(m, width="100%", height=550, returned_objects=["last_object_clicked"])
    
    # 🎯 STEP 3: Listen for map-marker click event assignments
    if map_data and map_data.get("last_object_clicked"):
        click_lat = map_data["last_object_clicked"]["lat"]
        click_lng = map_data["last_object_clicked"]["lng"]
        
        coordinate_match = df_incidents[
            (abs(df_incidents['lat'] - click_lat) < 0.0001) & 
            (abs(df_incidents['lng'] - click_lng) < 0.0001)
        ]
        
        if not coordinate_match.empty:
            new_map_selection = coordinate_match.iloc[0]['IncidentID']
            if st.session_state.selected_inc_id != new_map_selection:
                st.session_state.selected_inc_id = new_map_selection
                st.rerun()

# ==========================================
# LOWER PANEL FULL TEXT DISCOVERY GRID
# ==========================================
if st.session_state.selected_inc_id:
    active_record = df_incidents[df_incidents['IncidentID'] == st.session_state.selected_inc_id].iloc[0]
    st.write("---")
    st.markdown(f"### Full Traffic News (Case ID: {active_record['IncidentID']})")
    st.info(f"**Incident Category Classification:** {active_record['Category']}\n\n**Location Corridor:** {active_record['Location']}\n\n**Incident Description:** {active_record['Details']}")
