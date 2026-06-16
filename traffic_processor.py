# traffic_processor.py
import xml.etree.ElementTree as ET
import re
import json
import pandas as pd
import geopandas as gpd  # 🎯 FIXED SYNTAX TYPO HERE
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
        
        # Load the network intersection geojson node map dataset
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
    # PART 1A: THE LANGUAGE PROCESSING ENGINE (NLP PIPELINE)
    # =========================================================================
    def analyze_incident_language(self, message_block):
        """Parses text narratives into structured, decoupled data packages."""
        inc_id = self._get_xml_text(message_block, 'ID')
        status = self._get_xml_text(message_block, 'INCIDENT_STATUS_EN').upper().strip()
        location_en = self._get_xml_text(message_block, 'LOCATION_EN').upper().strip()
        landmark_en = self._get_xml_text(message_block, 'NEAR_LANDMARK_EN').upper().strip()
        between_en = self._get_xml_text(message_block, 'BETWEEN_LANDMARK_EN').upper().strip()
        content_en = self._get_xml_text(message_block, 'CONTENT_EN').strip()
        content_upper = content_en.upper()
        
        if status == "CLOSED" or not location_en:
            return None
            
        active_text_block = content_upper.split("RESUMED NORMAL")[0]
        text_pool = active_text_block + " " + location_en

        # Task A1: Identify Incident Type
        incident_type = "General Alert"
        if any(kw in text_pool for kw in ["ACCIDENT", "COLLISION", "CAR CRASH"]):
            incident_type = "Accident"
        elif any(kw in text_pool for kw in ["ROAD WORKS", "ROADWORKS", "MAINTENANCE", "REPAIR"]):
            incident_type = "Road Works"
        elif any(kw in text_pool for kw in ["TRAFFIC QUEUE", "CONGESTION", "BUSY", "SLOW TRAFFIC"]):
            incident_type = "Congestion"
        elif any(kw in text_pool for kw in ["CLOSED", "POLICE INVESTIGATION", "BLOCKAGE"]):
            incident_type = "Road Closure"

        # Task A2: Identify Scale / Extent / Seriousness
        scale_extent = "Minor"
        if any(kw in text_pool for kw in ["ALL LANES CLOSED", "SERIOUS", "COMPLETELY BLOCKED", "FATAL"]):
            scale_extent = "Major"
        elif any(kw in text_pool for kw in ["PART OF THE LANES", "PART OF LANES", "SLOW TRAFFIC", "BUSY", "QUEUE"]):
            scale_extent = "Moderate"

        # Task A4: Accurate Location Extraction via Sequential Substring Masking
        extracted_roads = []
        working_text = text_pool
        
        for road_name in self.road_names_cache:
            pattern = r'\b' + re.escape(road_name) + r'\b'
            if re.search(pattern, working_text):
                extracted_roads.append(road_name)
                working_text = re.sub(pattern, "X" * len(road_name), working_text)

        # Task A3: Identify Format Layout Configuration
        format_layout = "GENERAL"
        if location_en == "BUSY ROAD SECTIONS":
            format_layout = "BROADCAST"
        elif "JUNCTION" in text_pool or (len(extracted_roads) >= 2 and any(kw in text_pool for kw in ["AND", "CORNER OF"])):
            format_layout = "JUNCTION"
        elif "BETWEEN" in text_pool or between_en:
            format_layout = "BETWEEN"
        elif "NEAR" in text_pool or landmark_en:
            format_layout = "NEAR"

        # Compass Target Direction Extractor
        bound_compass = None
        if any(kw in text_pool for kw in ["WEST BOUND", "WESTBOUND", "CENTRAL BOUND", "CENTRAL-BOUND", "KENNEDY TOWN BOUND", "SHEUNG WAN BOUND", "TO CENTRAL", "TUEN MUN BOUND", "YUEN LONG BOUND", "TSUEN WAN BOUND"]):
            bound_compass = "WEST"
        elif any(kw in text_pool for kw in ["EAST BOUND", "EASTBOUND", "CHAI WAN BOUND", "EASTERN BOUND", "CAUSEWAY BAY BOUND", "QUARRY BAY BOUND", "NORTH POINT BOUND", "TO CHAI WAN", "SAI KUNG BOUND", "MA ON SHAN BOUND", "TAI PO BOUND"]):
            bound_compass = "EAST"
        elif any(kw in text_pool for kw in ["SOUTH BOUND", "SOUTHBOUND", "ABERDEEN BOUND", "STANLEY BOUND", "REPULSE BAY BOUND", "WONG CHUK HANG BOUND"]):
            bound_compass = "SOUTH"
        elif any(kw in text_pool for kw in ["NORTH BOUND", "NORTHBOUND", "KOWLOON BOUND", "CROSS HARBOUR", "SHATIN BOUND", "SHA TIN BOUND", "FANLING BOUND", "SHEUNG SHUI BOUND", "KWUN TONG BOUND"]):
            bound_compass = "NORTH"

        return {
            "incident_id": inc_id,
            "type": incident_type,
            "scale": scale_extent,
            "format": format_layout,
            "bound_compass": bound_compass,
            "extracted_roads": extracted_roads,
            "primary_location": location_en,
            "near_landmark": landmark_en,
            "between_landmark": between_en,
            "details": content_en,
            "active_text_block": active_text_block
        }

    # =========================================================================
    # PART 1B: THE GEOSPATIAL PROCESSING ENGINE (SPATIAL ROUTING PIPELINE)
    # =========================================================================
    def execute_geospatial_processing(self, nlp_payload):
        """Processes geometries using explicit intersection topology lookups and trajectory validation."""
        if not nlp_payload or not nlp_payload["extracted_roads"]:
            return None

        # Discover Travel Direction database attributes
        dir_col = None
        for col in self.road_df.columns:
            if any(k in col for k in ['TRAVEL', 'DIR', 'TRAFFIC']):
                dir_col = col
                break

        # Dynamic Environment CRS Axis-Info Evaluation
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

        # DETERMINISTIC LINK-NODE TOPOLOGY JUNCTION ROUTER
        if nlp_payload["format"] == "JUNCTION" and len(nlp_payload["extracted_roads"]) >= 2:
            road_a_name = nlp_payload["extracted_roads"][0]
            road_b_name = nlp_payload["extracted_roads"][1]
            
            road_a_df = self.road_df[self.road_df['STREET_ENAME'] == road_a_name]
            road_b_df = self.road_df[self.road_df['STREET_ENAME'] == road_b_name]
            
            if not road_a_df.empty and not road_b_df.empty and not self.intersection_df.empty:
                route_ids_a = set(road_a_df['ROUTE_ID'].dropna().astype(str).str.split('.').str[0])
                route_ids_b = set(road_b_df['ROUTE_ID'].dropna().astype(str).str.split('.').str[0])
                
                rd_cols = [c for c in self.intersection_df.columns if c.startswith('RD_ID_')]
                
                for idx, node_row in self.intersection_df.iterrows():
                    node_links = set()
                    for col in rd_cols:
                        if pd.notna(node_row[col]):
                            node_links.add(str(node_row[col]).split('.')[0])
                    
                    if node_links.intersection(route_ids_a) and node_links.intersection(route_ids_b):
                        output_geometry = node_row['GEOMETRY']
                        break
                
                if output_geometry is None:
                    name_matches = self.intersection_df[
                        self.intersection_df['INT_ENAME'].str.contains(road_a_name, na=False) &
                        self.intersection_df['INT_ENAME'].str.contains(road_b_name, na=False)
                    ]
                    if not name_matches.empty:
                        output_geometry = name_matches.iloc[0]['GEOMETRY']

            if output_geometry is None and not road_a_df.empty and not road_b_df.empty:
                geom_a = road_a_df.geometry.unary_union
                geom_b = road_b_df.geometry.unary_union
                intersection = geom_a.intersection(geom_b)
                output_geometry = intersection.centroid if not intersection.is_empty else nearest_points(geom_a, geom_b)[0]
                
            if output_geometry:
                spatial_features.append({
                    "type": "Feature",
                    "geometry": mapping(output_geometry),
                    "properties": {
                        "IncidentID": nlp_payload["incident_id"],
                        "RoadName": road_a_name,
                        "RouteID": "INTERSECTION_NODE"
                    }
                })

        # Format 2: Continuous Line Segment Layouts (Near, Between, Broadcast, General)
        else:
            combined_gdf_list = []
            for r_name in nlp_payload["extracted_roads"]:
                r_segments = self.road_df[self.road_df['STREET_ENAME'] == r_name]
                if not r_segments.empty:
                    filtered_segments = filter_segments(r_segments, nlp_payload["bound_compass"])
                    combined_gdf_list.append(filtered_segments)
                    
            if combined_gdf_list:
                merged_roads_gdf = pd.concat(combined_gdf_list)
                target_road_geom = merged_roads_gdf.geometry.unary_union
                
                pts_context = []
                if nlp_payload["near_landmark"] in self.landmark_cache:
                    pts_context.append(self.landmark_cache[nlp_payload["near_landmark"]])
                if nlp_payload["between_landmark"] in self.landmark_cache:
                    pts_context.append(self.landmark_cache[nlp_payload["between_landmark"]])
                    
                for cross_road in self.road_names_cache:
                    if cross_road in nlp_payload["extracted_roads"]: continue
                    if cross_road in nlp_payload["active_text_block"]:
                        cross_feats = self.road_df[self.road_df['STREET_ENAME'] == cross_road]
                        if not cross_feats.empty:
                            cross_geom = cross_feats.geometry.unary_union
                            intersection = target_road_geom.intersection(cross_geom)
                            pts_context.append(intersection.centroid if not intersection.is_empty else nearest_points(target_road_geom, cross_geom)[0])

                buffers = []
                if nlp_payload["format"] == "BETWEEN" and len(pts_context) >= 2:
                    buffers = [MultiPoint(pts_context).envelope.buffer(40)]
                elif pts_context:
                    buffers = [pt.buffer(self.distance_threshold) for pt in pts_context]

                is_queueing = "TRAFFIC QUEUE" in nlp_payload["active_text_block"]

                for _, row in merged_roads_gdf.iterrows():
                    geom = row['GEOMETRY']
                    if geom is None: continue
                    
                    if not buffers or is_queueing or nlp_payload["format"] == "BROADCAST":
                        spatial_features.append({
                            "type": "Feature",
                            "geometry": mapping(geom),
                            "properties": {
                                "IncidentID": nlp_payload["incident_id"],
                                "RoadName": row.get('STREET_ENAME', 'UNKNOWN'),
                                "RouteID": str(row.get('ROUTE_ID', 'UNKNOWN'))
                            }
                        })
                    else:
                        if any(geom.intersects(buf) for buf in buffers):
                            spatial_features.append({
                                "type": "Feature",
                                "geometry": mapping(geom),
                                "properties": {
                                    "IncidentID": nlp_payload["incident_id"],
                                    "RoadName": row.get('STREET_ENAME', 'UNKNOWN'),
                                    "RouteID": str(row.get('ROUTE_ID', 'UNKNOWN'))
                                }
                            })

        if not spatial_features:
            return None

        # Task B3: Structure final configurations into a standardized exchange dictionary
        spatial_json_payload = {
            "incident_id": nlp_payload["incident_id"],
            "type": nlp_payload["type"],
            "scale": nlp_payload["scale"],
            "format": nlp_payload["format"],
            "location_text": nlp_payload["primary_location"],
            "details": nlp_payload["details"],
            "geojson": {
                "type": "FeatureCollection",
                "features": spatial_features
            }
        }
        return spatial_json_payload

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

            # Run Part 1a Pipeline: Language Entity Processing
            nlp_payload = self.analyze_incident_language(message_element)
            if not nlp_payload: continue

            # Run Part 1b Pipeline: Geospatial Routing Engine
            spatial_output = self.execute_geospatial_processing(nlp_payload)
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
