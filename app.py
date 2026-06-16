# app.py
import os
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
from traffic_processor import TrafficIncidentEngine

st.set_page_config(layout="wide", page_title="HKI Traffic Dashboard")

# Custom CSS styling to align headers parallel and fix layout padding
st.markdown("""
    <style>
        div[data-testid="stColumn"] {
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }
        .unified-header {
            margin-bottom: 12px;
            height: 60px;
        }
        .table-header {
            font-weight: bold;
            background-color: #F0F2F6;
            padding: 10px;
            border-radius: 6px;
            margin-bottom: 8px;
        }
        .selected-row-box {
            border: 2px solid #E63946;
            background-color: #FFF5F5;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 10px;
        }
        .standard-row-box {
            border-bottom: 1px solid #EDEDED;
            padding: 10px 4px;
            margin-bottom: 2px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Traffic News Visualization for Hong Kong Island")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Inside app.py
@st.cache_data
def run_spatial_processing_pipeline():
    # 🎯 Safeguard: Explicitly calculate the absolute base directory
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Instantiate the updated intersection-graph engine using strict keywords
    engine = TrafficIncidentEngine(
        road_path=os.path.join(BASE_DIR, "iG1000_HKI_FGDB", "hki_centerline.zip"),
        boundary_path=os.path.join(BASE_DIR, "iG1000_HKI_FGDB", "hki_boundary.zip"),
        building_path=os.path.join(BASE_DIR, "iG1000_HKI_FGDB", "hki_building.zip"),
        xml_path=os.path.join(BASE_DIR, "combined.xml"),
        intersection_path=os.path.join(BASE_DIR, "hki_intersection.geojson"), # Added to align with the new 2-part engine
        distance_threshold=500
    )
    return engine.process_active_incidents()

raw_incidents, gdf_spatial = run_spatial_processing_pipeline()

# ==========================================
# BI-DIRECTIONAL SELECTION & STATE LOCKS
# ==========================================
if "selected_inc_id" not in st.session_state:
    st.session_state.selected_inc_id = None

df_incidents = raw_incidents.copy()

# Lock pinning coordinates directly onto row attributes
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
                combined_line = inc_geom.unary_union
                if combined_line.geom_type in ['LineString', 'MultiLineString']:
                    rep_pt = combined_line.interpolate(0.5, normalized=True)
                else:
                    rep_pt = combined_line.representative_point()
            rep_points.append({'IncidentID': inc_id, 'lat': rep_pt.y, 'lng': rep_pt.x})
            
    if rep_points:
        df_rep = pd.DataFrame(rep_points)
        df_incidents = df_incidents.merge(df_rep, on='IncidentID', how='inner')

CATEGORY_STYLES = {
    "Accident": {"color": "red", "icon": "car", "prefix": "fa"},
    "Road Works": {"color": "orange", "icon": "wrench", "prefix": "fa"},
    "Congestion": {"color": "amber", "icon": "clock-o", "prefix": "fa"},
    "Road Closure": {"color": "darkred", "icon": "ban", "prefix": "fa"},
    "General Alert": {"color": "blue", "icon": "exclamation-circle", "prefix": "fa"}
}

# ==========================================
# INTERACTIVE SCREEN LAYOUT GENERATION
# ==========================================
col_table, col_map = st.columns([10, 9])

with col_table:
    st.markdown("<div class='unified-header'><h3>Traffic News Log</h3></div>", unsafe_allow_html=True)
    
    # SEARCH & SORT CONTROLS
    col_search, col_sort = st.columns([2, 1])
    with col_search:
        search_query = st.text_input("Search logs (ID, Location, or Keyword)", "", placeholder="Type here to filter...")
    with col_sort:
        sort_option = st.selectbox("Sort By", ["Default", "Case ID", "Incident Type", "Location"])

    # Apply Search Rules
    df_filtered = df_incidents.copy()
    if search_query:
        df_filtered = df_filtered[
            df_filtered['IncidentID'].astype(str).str.contains(search_query, case=False, na=False) |
            df_filtered['Category'].str.contains(search_query, case=False, na=False) |
            df_filtered['Location'].str.contains(search_query, case=False, na=False) |
            df_filtered['Details'].str.contains(search_query, case=False, na=False)
        ]

    # Apply Sorting Rules
    if sort_option == "Case ID":
        df_filtered = df_filtered.sort_values(by="IncidentID", ascending=True)
    elif sort_option == "Incident Type":
        df_filtered = df_filtered.sort_values(by="Category", ascending=True)
    elif sort_option == "Location":
        df_filtered = df_filtered.sort_values(by="Location", ascending=True)

    # 🎯 DYNAMIC TOP-PINNING LOGIC: Force the selected incident to the absolute top of the table view
    if st.session_state.selected_inc_id and st.session_state.selected_inc_id in df_filtered['IncidentID'].values:
        selected_row = df_filtered[df_filtered['IncidentID'] == st.session_state.selected_inc_id]
        remaining_rows = df_filtered[df_filtered['IncidentID'] != st.session_state.selected_inc_id]
        df_filtered = pd.concat([selected_row, remaining_rows]).reset_index(drop=True)

    # TABLE HEADERS
    st.markdown("""
        <div class='table-header'>
            <table style='width:100%; border-collapse:collapse; table-layout:fixed; font-size:14px;'>
                <tr>
                    <td style='width:15%; font-weight:bold; color:#333;'>Case ID</td>
                    <td style='width:15%; font-weight:bold; color:#333;'>Type</td>
                    <td style='width:25%; font-weight:bold; color:#333;'>Corridor Context</td>
                    <td style='width:45%; font-weight:bold; color:#333;'>Full Description Log</td>
                </tr>
            </table>
        </div>
    """, unsafe_allow_html=True)

    # TRADITIONAL WRAPPED DATA ROWS
    with st.container(height=450):
        if df_filtered.empty:
            st.info("No matching traffic incidents found.")
        else:
            for _, row in df_filtered.iterrows():
                is_selected = (row['IncidentID'] == st.session_state.selected_inc_id)
                
                # Apply high-visibility red bounding container if selected
                row_class = "selected-row-box" if is_selected else "standard-row-box"
                
                st.markdown(f"<div class='{row_class}'>", unsafe_allow_html=True)
                r_cols = st.columns([15, 15, 25, 45])
                
                # Column 1: Row Selection Button
                btn_label = f"🎯 {row['IncidentID']}" if is_selected else str(row['IncidentID'])
                if r_cols[0].button(btn_label, key=f"row_id_{row['IncidentID']}", use_container_width=True, type="primary" if is_selected else "secondary"):
                    st.session_state.selected_inc_id = row['IncidentID']
                    st.rerun()
                
                # Columns 2, 3, & 4: Markdown strings with auto-wrapping lines
                r_cols[1].markdown(f"<div style='padding-top:6px; font-size:13px;'>{row['Category']}</div>", unsafe_allow_html=True)
                r_cols[2].markdown(f"<div style='padding-top:6px; font-size:13px; font-weight:bold;'>{row['Location']}</div>", unsafe_allow_html=True)
                r_cols[3].markdown(f"<div style='padding-top:6px; font-size:13px; line-height:1.4; color:#333;'>{row['Details']}</div>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)

with col_map:
    st.markdown("<div class='unified-header'><h3>Map Visualization</h3></div>", unsafe_allow_html=True)
    
    # Establish stationary view coordinates before drawing map to avoid layout tearing
    map_center = [22.28552, 114.15769]
    zoom_level = 11
    
    if st.session_state.selected_inc_id:
        active_rows = df_incidents[df_incidents['IncidentID'] == st.session_state.selected_inc_id]
        if not active_rows.empty:
            map_center = [active_rows.iloc[0]['lat'], active_rows.iloc[0]['lng']]
            zoom_level = 15
            
    m = folium.Map(location=map_center, zoom_start=zoom_level, tiles="cartodbpositron")
    
    for _, row in df_incidents.iterrows():
        is_current = (row['IncidentID'] == st.session_state.selected_inc_id)
        style = CATEGORY_STYLES.get(row['Category'], CATEGORY_STYLES["General Alert"])
        
        # Selected pin changes to high-visibility darkpurple
        bg_color = "darkpurple" if is_current else style["color"]
        
        # 🎯 POPUP FIX: No popup parameter attached to the Marker container.
        # This allows the mouse click to pass cleanly straight to st_folium's callback loop.
        folium.Marker(
            location=[row['lat'], row['lng']],
            icon=folium.Icon(color=bg_color, icon_color="white", icon=style["icon"], prefix=style["prefix"]),
            tooltip=f"ID: {row['IncidentID']} - {row['Location']}"
        ).add_to(m)
        
    if st.session_state.selected_inc_id:
        matched_shapes = gdf_spatial[gdf_spatial['IncidentID'] == st.session_state.selected_inc_id]
        if not matched_shapes.empty:
            matched_shapes_4326 = matched_shapes.to_crs(epsg=4326)
            for _, row in matched_shapes_4326.iterrows():
                geom = row['geometry']
                if geom is None or geom.geom_type == 'Point': continue
                folium.GeoJson(geom, style_function=lambda x: {'color': '#E63946', 'weight': 7, 'opacity': 0.9}).add_to(m)

    map_data = st_folium(m, width="100%", height=530, returned_objects=["last_object_clicked"])
    
    # Nearest-Neighbor matching handles coordinate synchronization instantly
    if map_data and map_data.get("last_object_clicked") and not df_incidents.empty:
        click_lat = map_data["last_object_clicked"]["lat"]
        click_lng = map_data["last_object_clicked"]["lng"]
        
        spatial_distances = ((df_incidents['lat'] - click_lat)**2 + (df_incidents['lng'] - click_lng)**2)
        true_closest_idx = spatial_distances.idxmin()
        
        if spatial_distances[true_closest_idx] < 0.0005:
            new_map_selection = df_incidents.loc[true_closest_idx, 'IncidentID']
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
