# traffic_processor.py
import xml.etree.ElementTree as ET
import re
import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, MultiPoint, shape, mapping
from shapely.ops import nearest_points

class TrafficIncidentEngine:
    def __init__(self, road_path, boundary_path, building_path, xml_path, intersection_path=None, distance_threshold=500):
        self.road_path = road_path
        self.boundary_path = boundary_path
        self.building_path = building_path  
        self.xml_path = xml_path
        self.intersection_path = intersection_path or "hki_intersection.geojson"
        self.distance_threshold = distance_threshold
        
        self.road_df = None
        self.intersection_df = None
        self.landmark_cache = {}
        self.road_names_cache = []
        
    def initialize_spatial_basemaps(self):
        """Loads GIS layers and sets up the intersection node graph structures."""
        self.road_df = gpd.read_file(f"zip://{self.road_path}")
        boundary_df = gpd.read_file(f"zip://{self.boundary_path}")
        building_df = gpd.read_file(f"zip://{self.building_path}") 
        
        try:
            import os
            base_dir = os.path.dirname(os.path.abspath(__file__))
            full_intersection_path = os.path.join(base_dir, self.intersection_path)
            
            self.intersection_df = gpd.read_file(full_intersection_path)
            self.intersection_df.columns = self.intersection_df.columns.str.upper()
            self.intersection_df = self.intersection_df.set_geometry("GEOMETRY").set_crs(epsg=2326, allow_override=True)
        except Exception as e:
            print(f"⚠️ Warning: Could not load intersection layer from {self.intersection_path}. Reason: {e}")
            self.intersection_df = gpd.GeoDataFrame(
                columns=['GEOMETRY', 'INT_ENAME'], 
                crs=2326, 
                geometry='GEOMETRY'
            )

        self.road_df.columns = self.road_df.columns.str.upper()
        boundary_df.columns = boundary_df.columns.str.upper()
        building_df.columns = building_df.columns.str.upper()
        
        self.road_df = self.road_df.set_geometry("GEOMETRY").set_crs(epsg=2326, allow_override=True)
        boundary_df = boundary_df.set_geometry("GEOMETRY").set_crs(epsg=2326, allow_override=True)
        building_df = building_df.set_geometry("GEOMETRY").set_crs(epsg=2326, allow_override=True)
            
        for _, feature in boundary_df.iterrows():
            geom = feature['GEOMETRY']
            if geom is None: continue
            centroid = geom.centroid
            for field in ['ENGLISHSITENAME', 'ENGLISHBUILDINGNAME']:
                if field in boundary_df.columns and pd.notna(feature[field]):
                    val = str(feature[field]).upper().strip()
                    if len(val) > 3 and val not in ["NAN", "<NA>"]: 
                        self.landmark_cache[val] = centroid

        for _, feature in building_df.iterrows():
            geom = feature['GEOMETRY']
            if geom is None: continue
            centroid = geom.centroid
            for field in ['BUILDINGNAME', 'ENGLISHBUILDINGNAME', 'BUILDING_EN']:
                if field in building_df.columns and pd.notna(feature[field]):
                    val = str(feature[field]).upper().strip()
                    if len(val) > 3 and val not in ["NAN", "<NA>"] and val not in self.landmark_cache: 
                        self.landmark_cache[val] = centroid

        if 'STREET_ENAME' in self.road_df.columns:
            raw_names = set(self.road_df['STREET_ENAME'].dropna().str.upper().str.strip())
            self.road_names_cache = sorted(list(raw_names), key=len, reverse=True)

    @staticmethod
    def _get_xml_text(message_element, tag_name):
        for child in message_element:
            if child.tag.split('}')[-1] == tag_name:
                return child.text.strip() if child.text else ""
        return ""

    # =========================================================================
    # PART 1A: THE LANGUAGE PROCESSING ENGINE (NLP COMPOUND EXPLODER)
    # =========================================================================
    def explode_and_analyze_message(self, message_block):
        """Identifies multiple incidents inside a single text block and explodes them into individual cases."""
        inc_id = self._get_xml_text(message_block, 'ID')
        status = self._get_xml_text(message_block, 'INCIDENT_STATUS_EN').upper().strip()
        location_en = self._get_xml_text(message_block, 'LOCATION_EN').upper().strip()
        content_en = self._get_xml_text(message_block, 'CONTENT_EN').strip()
        content_upper = content_en.upper()
        
        if status == "CLOSED" or not location_en:
            return []
            
        # 🎯 THE FIX: Do NOT split or truncate the text at "RESUMED NORMAL". Scan the entire text window.
        active_text_block = content_upper 
        
        # Discover where roads appear chronologically inside the text block
        road_mentions = []
        for road_name in self.road_names_cache:
            pattern = r'\b' + re.escape(road_name) + r'\b'
            for match in re.finditer(pattern, active_text_block):
                road_mentions.append({
                    "name": road_name,
                    "start": match.start(),
                    "end": match.end()
                })
                
        if not road_mentions:
            return []

        # Sort road mentions chronologically by their character position index in the text
        road_mentions = sorted(road_mentions, key=lambda x: x["start"])
        
        # Filter out nested substrings (e.g., prevent 'Hill Road' from hijacking 'Hammer Hill Road')
        clean_mentions = []
        for i, current in enumerate(road_mentions):
            is_sub_match = False
            for other in road_mentions:
                if current != other and other["start"] <= current["start"] and other["end"] >= current["end"]:
                    is_sub_match = True
                    break
            if not is_sub_match:
                clean_mentions.append(current)

        sub_cases_payloads = []
        for idx, mention in enumerate(clean_mentions):
            slice_start = mention["start"]
            # Slice up until the start of the next road text section, or the very end of the document block
            slice_end = clean_mentions[idx + 1]["start"] if idx + 1 < len(clean_mentions) else len(active_text_block)
            
            # Extract the pure structural sentence block dedicated to this road feature
            localized_clause = active_text_block[slice_start:slice_end].strip()
            raw_localized_clause_en = content_en[slice_start:slice_end].strip()
            
            # Reconstruct trailing context data words if a text split drops important context
            if idx == 0 and slice_start > 0:
                prefix_context = active_text_block[:slice_start]
                localized_clause = prefix_context + " " + localized_clause
                raw_localized_clause_en = content_en[:slice_start] + " " + raw_localized_clause_en

            # Parse Incident Type for this sub-clause dynamically
            incident_type = "General Alert"
            if any(kw in active_text_block for kw in ["ACCIDENT", "COLLISION", "CAR CRASH"]): incident_type = "Accident"
            elif any(kw in active_text_block for kw in ["ROAD WORKS", "ROADWORKS", "MAINTENANCE", "REPAIR"]): incident_type = "Road Works"
            elif any(kw in active_text_block for kw in ["TRAFFIC QUEUE", "CONGESTION", "BUSY", "SLOW TRAFFIC"]): incident_type = "Congestion"
            elif any(kw in active_text_block for kw in ["CLOSED", "POLICE INVESTIGATION", "BLOCKAGE"]): incident_type = "Road Closure"

            # Parse Scale / Severity status metrics for this sub-clause
            scale_extent = "Minor"
            if any(kw in localized_clause for kw in ["ALL LANES CLOSED", "SERIOUS", "COMPLETELY BLOCKED", "FATAL"]): scale_extent = "Major"
            elif any(kw in localized_clause for kw in ["PART OF THE LANES", "PART OF LANES", "SLOW TRAFFIC", "BUSY", "QUEUE"]): scale_extent = "Moderate"

            # Parse Format Layout configuration constraints matching this sub-clause text context
            format_layout = "GENERAL"
            if location_en == "BUSY ROAD SECTIONS": format_layout = "BROADCAST"
            elif "JUNCTION" in localized_clause or any(kw in localized_clause for kw in ["AND", "CORNER OF"]): format_layout = "JUNCTION"
            elif "BETWEEN" in localized_clause: format_layout = "BETWEEN"
            elif "NEAR" in localized_clause: format_layout = "NEAR"

            # Parse Localized Compass Target Direction flags
            bound_compass = None
            if any(kw in localized_clause for kw in ["WEST BOUND", "WESTBOUND", "CENTRAL BOUND", "CENTRAL-BOUND", "KENNEDY TOWN BOUND", "SHEUNG WAN BOUND", "TO CENTRAL", "TUEN MUN BOUND", "YUEN LONG BOUND", "TSUEN WAN BOUND"]): bound_compass = "WEST"
            elif any(kw in localized_clause for kw in ["EAST BOUND", "EASTBOUND", "CHAI WAN BOUND", "EASTERN BOUND", "CAUSEWAY BAY BOUND", "QUARRY BAY BOUND", "NORTH POINT BOUND", "TO CHAI WAN", "SAI KUNG BOUND", "MA ON SHAN BOUND", "TAI PO BOUND", "CROSS HARBOUR TUNNEL BOUND"]): bound_compass = "EAST"
            elif any(kw in localized_clause for kw in ["SOUTH BOUND", "SOUTHBOUND", "ABERDEEN BOUND", "STANLEY BOUND", "REPULSE BAY BOUND", "WONG CHUK HANG BOUND"]): bound_compass = "SOUTH"
            elif any(kw in localized_clause for kw in ["NORTH BOUND", "NORTHBOUND", "KOWLOON BOUND", "CROSS HARBOUR", "SHATIN BOUND", "SHA TIN BOUND", "FANLING BOUND", "SHEUNG SHUI BOUND", "KWUN TONG BOUND"]): bound_compass = "NORTH"

            # 🎯 THE FIX: Suffix the case ID with an explicit sub-incident flag if multi-containing news triggers
            if len(clean_mentions) > 1:
                sub_case_id = f"{inc_id}-{idx + 1}"
            else:
                sub_case_id = inc_id # Maintain perfect backward compatibility for single-incident logs
            
            sub_cases_payloads.append({
                "sub_case_id": sub_case_id,
                "parent_id": inc_id,
                "type": incident_type,
                "scale": scale_extent,
                "format": format_layout,
                "bound_compass": bound_compass,
                "target_road": mention["name"],
                "primary_location": mention["name"] if location_en == "BUSY ROAD SECTIONS" else location_en,
                "details": raw_localized_clause_en if len(raw_localized_clause_en) > 10 else content_en,
                "localized_clause_text": localized_clause
            })
            
        return sub_cases_payloads

    # =========================================================================
    # PART 1B: THE GEOSPATIAL PROCESSING ENGINE (SPATIAL ROUTING PIPELINE)
    # =========================================================================
    def execute_geospatial_processing(self, sub_case_payload):
        """Processes geometry operations matching the parameters of a single sub-case profile."""
        if not sub_case_payload:
            return None

        dir_col = None
        for col in self.road_df.columns:
            if any(k in col for k in ['TRAVEL', 'DIR', 'TRAFFIC']):
                dir_col = col
                break

        is_northing_first = False
        if self.road_df.crs and hasattr(self.road_df.crs, 'axis_info'):
            try:
                if self.road_df.crs.axis_info[0].direction.lower() == 'north':
                    is_northing_first = True
            except: pass

        def filter_segments(gdf, compass):
            if not compass or gdf.empty: return gdf
            valid_idx = []
            for idx, row in gdf.iterrows():
                geom = row['GEOMETRY']
                if geom is None: continue
                
                dir_val = "1"
                if dir_col and pd.notna(row[dir_col]):
                    dir_val = str(row[dir_col]).strip().split('.')[0]
                    
                if dir_val == '3':
                    coords = list(geom.coords) if geom.geom_type == 'LineString' else (list(max(geom.geoms, key=lambda l: l.length).coords) if not geom.is_empty else [])
                    if len(coords) < 2:
                        valid_idx.append(idx)
                        continue
                        
                    first, last = coords[0], coords[-1]
                    e_start, n_start = (first[1], first[0]) if is_northing_first else (first[0], first[1])
                    e_end, n_end = (last[1], last[0]) if is_northing_first else (last[0], last[1])
                    
                    dx, dy = e_end - e_start, n_end - n_start
                    keep = True
                    if compass == "WEST" and dx > 10: keep = False
                    if compass == "EAST" and dx < -10: keep = False
                    if compass == "SOUTH" and dy > 10: keep = False
                    if compass == "NORTH" and dy < -10: keep = False
                    if keep: valid_idx.append(idx)
                else:
                    valid_idx.append(idx)
            return gdf.loc[valid_idx] if 0 < len(valid_idx) < len(gdf) else gdf

        output_geometry = None
        spatial_features = []
        target_road_name = sub_case_payload["target_road"]
        
        r_segments = self.road_df[self.road_df['STREET_ENAME'] == target_road_name]
        if r_segments.empty:
            return None
            
        filtered_segments = filter_segments(r_segments, sub_case_payload["bound_compass"])
        target_road_geom = filtered_segments.geometry.unary_union

        # Format 1: Junction Node Layout Router using explicit Topology lookups
        if sub_case_payload["format"] == "JUNCTION":
            cross_road_name = None
            for road_name in self.road_names_cache:
                if road_name != target_road_name and road_name in sub_case_payload["localized_clause_text"]:
                    cross_road_name = road_name
                    break
            
            if cross_road_name:
                road_b_df = self.road_df[self.road_df['STREET_ENAME'] == cross_road_name]
                if not road_b_df.empty and not self.intersection_df.empty:
                    route_ids_a = set(filtered_segments['ROUTE_ID'].dropna().astype(str).str.split('.').str[0])
                    route_ids_b = set(road_b_df['ROUTE_ID'].dropna().astype(str).str.split('.').str[0])
                    rd_cols = [c for c in self.intersection_df.columns if c.startswith('RD_ID_')]
                    
                    for idx, node_row in self.intersection_df.iterrows():
                        node_links = set()
                        for col in rd_cols:
                            if pd.notna(node_row[col]): node_links.add(str(node_row[col]).split('.')[0])
                        if node_links.intersection(route_ids_a) and node_links.intersection(route_ids_b):
                            output_geometry = node_row['GEOMETRY']
                            break

                if output_geometry is None and not road_b_df.empty:
                    geom_b = road_b_df.geometry.unary_union
                    intersection = target_road_geom.intersection(geom_b)
                    output_geometry = intersection.centroid if not intersection.is_empty else nearest_points(target_road_geom, geom_b)[0]

            if output_geometry:
                spatial_features.append({
                    "type": "Feature",
                    "geometry": mapping(output_geometry),
                    "properties": {
                        "IncidentID": sub_case_payload["sub_case_id"],
                        "RoadName": target_road_name,
                        "RouteID": "INTERSECTION_NODE"
                    }
                })
                
        # Format 2: Segment Length Layouts (Near, Between, Broadcast, General)
        if not spatial_features:
            pts_context = []
            
            # Scan for landmark markers inside the database mapping parameters
            for landmark_key, landmark_pt in self.landmark_cache.items():
                if landmark_key in sub_case_payload["localized_clause_text"]:
                    pts_context.append(landmark_pt)
                    
            # Scan for intersecting street boundaries inside the clause text window
            for cross_road in self.road_names_cache:
                if cross_road == target_road_name: continue
                if cross_road in sub_case_payload["localized_clause_text"]:
                    cross_feats = self.road_df[self.road_df['STREET_ENAME'] == cross_road]
                    if not cross_feats.empty:
                        cross_geom = cross_feats.geometry.unary_union
                        intersection = target_road_geom.intersection(cross_geom)
                        pts_context.append(intersection.centroid if not intersection.is_empty else nearest_points(target_road_geom, cross_geom)[0])

            buffers = []
            if sub_case_payload["format"] == "BETWEEN" and len(pts_context) >= 2:
                buffers = [MultiPoint(pts_context).envelope.buffer(40)]
            elif pts_context:
                buffers = [pt.buffer(self.distance_threshold) for pt in pts_context]

            is_queueing = "TRAFFIC QUEUE" in sub_case_payload["localized_clause_text"]

            for _, row in filtered_segments.iterrows():
                geom = row['GEOMETRY']
                if geom is None: continue
                
                if not buffers or is_queueing or sub_case_payload["format"] == "BROADCAST":
                    spatial_features.append({
                        "type": "Feature",
                        "geometry": mapping(geom),
                        "properties": {
                            "IncidentID": sub_case_payload["sub_case_id"],
                            "RoadName": target_road_name,
                            "RouteID": str(row.get('ROUTE_ID', 'UNKNOWN'))
                        }
                    })
                else:
                    if any(geom.intersects(buf) for buf in buffers):
                        spatial_features.append({
                            "type": "Feature",
                            "geometry": mapping(geom),
                            "properties": {
                                "IncidentID": sub_case_payload["sub_case_id"],
                                "RoadName": target_road_name,
                                "RouteID": str(row.get('ROUTE_ID', 'UNKNOWN'))
                            }
                        })

        if not spatial_features:
            return None

        return {
            "incident_id": sub_case_payload["sub_case_id"],
            "type": sub_case_payload["type"],
            "scale": sub_case_payload["scale"],
            "format": sub_case_payload["format"],
            "location_text": f"{target_road_name} ({sub_case_payload['bound_compass'] or 'General'} Bound)" if sub_case_payload["bound_compass"] else target_road_name,
            "details": sub_case_payload["details"],
            "geojson": {
                "type": "FeatureCollection",
                "features": spatial_features
            }
        }

    def process_active_incidents(self):
        """Unified runner script coordinating Part 1A and Part 1B pipelines."""
        if self.road_df is None:
            self.initialize_spatial_basemaps()

        with open(self.xml_path, 'r', encoding='utf-8') as f:
            full_text = f.read()

        message_blocks = re.findall(r'<message>.*?</message>', full_text, re.DOTALL)
        
        incident_records = []
        master_spatial_features = []

        for block in message_blocks:
            try:
                root_snippet = ET.fromstring(f"<root>{block}</root>")
                message_element = root_snippet.find('message')
            except: continue
            if message_element is None: continue

            # RUN PART 1A EXPLODER: Divides 1 message into an array of sub-cases
            sub_cases = self.explode_and_analyze_message(message_element)
            
            # Process each individual separated sub-case completely independently
            for sub_case in sub_cases:
                # Run Part 1b Pipeline: Geospatial Routing Engine
                spatial_output = self.execute_geospatial_processing(sub_case)
                if not spatial_output: continue

                incident_records.append({
                    'IncidentID': spatial_output["incident_id"],
                    'Category': spatial_output["type"],
                    'Scale': spatial_output["scale"],
                    'Location': spatial_output["location_text"],
                    'Details': spatial_output["details"]
                })
                
                master_spatial_features.extend(spatial_output["geojson"]["features"])

        df_incidents = pd.DataFrame(incident_records).drop_duplicates(subset=['IncidentID'])
        
        if master_spatial_features:
            features_gdf_list = []
            for feat in master_spatial_features:
                geom_obj = shape(feat["geometry"])
                features_gdf_list.append({
                    'IncidentID': feat["properties"]["IncidentID"],
                    'RoadName': feat["properties"]["RoadName"],
                    'RouteID': feat["properties"]["RouteID"],
                    'geometry': geom_obj
                })
            gdf_spatial = gpd.GeoDataFrame(features_gdf_list, crs=self.road_df.crs)
        else:
            gdf_spatial = gpd.GeoDataFrame(columns=['IncidentID', 'RoadName', 'RouteID', 'geometry'], crs=self.road_df.crs)

        active_ids = gdf_spatial['IncidentID'].unique()
        df_incidents = df_incidents[df_incidents['IncidentID'].isin(active_ids)].reset_index(drop=True)
        
        return df_incidents, gdf_spatial
