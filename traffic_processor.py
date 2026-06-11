# traffic_processor.py
import xml.etree.ElementTree as ET
import re
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, MultiPoint
from shapely.ops import nearest_points

# Hardcoded anchors for major bounds to ensure the math always has a target destination (HK80 EPSG:2326 coordinates)
MAJOR_DISTRICTS_HK80 = {
    "CENTRAL": (833500, 816000),
    "CHAI WAN": (841000, 812500),
    "EASTERN": (841000, 812500),
    "ABERDEEN": (834000, 812000),
    "KENNEDY TOWN": (831000, 816000),
    "WAN CHAI": (835500, 815500),
    "CAUSEWAY BAY": (837000, 815500),
    "NORTH POINT": (839000, 817000),
    "QUARRY BAY": (840000, 816000),
    "WONG CHUK HANG": (835000, 812000),
    "STANLEY": (839500, 808500),
    "MONG KOK": (835500, 820000), # Cross harbour reference
    "KOWLOON": (836000, 819000)
}

class TrafficIncidentEngine:
    def __init__(self, road_path, boundary_path, building_path, xml_path, distance_threshold=500):
        self.road_path = road_path
        self.boundary_path = boundary_path
        self.building_path = building_path  
        self.xml_path = xml_path
        self.distance_threshold = distance_threshold
        
        self.road_df = None
        self.landmark_cache = {}
        self.road_names_cache = set()
        
    def initialize_spatial_basemaps(self):
        """Loads GIS layers directly out of compressed zip files."""
        self.road_df = gpd.read_file(f"zip://{self.road_path}")
        boundary_df = gpd.read_file(f"zip://{self.boundary_path}")
        building_df = gpd.read_file(f"zip://{self.building_path}") 
        
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
            self.road_names_cache = set(self.road_df['STREET_ENAME'].dropna().str.upper().str.strip())

    def classify_incident(self, content_upper):
        if any(kw in content_upper for kw in ["ACCIDENT", "COLLISION", "CAR CRASH"]):
            return "Accident"
        elif any(kw in content_upper for kw in ["ROAD WORKS", "ROADWORKS", "MAINTENANCE", "REPAIR"]):
            return "Road Works"
        elif any(kw in content_upper for kw in ["TRAFFIC QUEUE", "CONGESTION", "BUSY", "SLOW TRAFFIC"]):
            return "Congestion"
        elif any(kw in content_upper for kw in ["CLOSED", "POLICE INVESTIGATION", "BLOCKAGE"]):
            return "Road Closure"
        else:
            return "General Alert"

    def is_correct_direction_vector(self, geom, target_geom):
        """
        Your Step 3 Math: Computes the Dot Product between the segment vector and the destination vector.
        If > 0, the segment is pointing generally towards the target bound.
        """
        if geom is None or target_geom is None:
            return True # Keep if we can't mathematically determine
            
        if geom.geom_type == 'LineString':
            coords = list(geom.coords)
        elif geom.geom_type == 'MultiLineString' and not geom.is_empty:
            coords = list(max(geom.geoms, key=lambda l: l.length).coords)
        else:
            return True
            
        if len(coords) < 2: 
            return True
            
        start_pt = coords[0]
        end_pt = coords[-1]
        
        # Vector 1: The direction the road segment is physically digitized
        v_seg_x = end_pt[0] - start_pt[0]
        v_seg_y = end_pt[1] - start_pt[1]
        
        # Vector 2: The direction from the start of the road segment to the final destination
        v_dest_x = target_geom.x - start_pt[0]
        v_dest_y = target_geom.y - start_pt[1]
        
        # Dot Product (A * B)
        dot_product = (v_seg_x * v_dest_x) + (v_seg_y * v_dest_y)
        
        # Positive dot product means the angle between the segment and destination is < 90 degrees (Same general direction)
        return dot_product > 0

    @staticmethod
    def _get_xml_text(message_element, tag_name):
        for child in message_element:
            if child.tag.split('}')[-1] == tag_name:
                return child.text.strip() if child.text else ""
        return ""

    def process_active_incidents(self):
        if self.road_df is None:
            self.initialize_spatial_basemaps()

        with open(self.xml_path, 'r', encoding='utf-8') as f:
            full_text = f.read()

        message_blocks = re.findall(r'<message>.*?</message>', full_text, re.DOTALL)
        incident_records = []
        spatial_features_list = []

        for block in message_blocks:
            try:
                root_snippet = ET.fromstring(f"<root>{block}</root>")
                message = root_snippet.find('message')
            except: continue
            if message is None: continue

            inc_id = self._get_xml_text(message, 'ID')
            status = self._get_xml_text(message, 'INCIDENT_STATUS_EN').upper().strip()
            location_en = self._get_xml_text(message, 'LOCATION_EN').upper().strip()
            landmark_en = self._get_xml_text(message, 'NEAR_LANDMARK_EN').upper().strip()
            between_en = self._get_xml_text(message, 'BETWEEN_LANDMARK_EN').upper().strip()
            content_en = self._get_xml_text(message, 'CONTENT_EN').strip()
            content_upper = content_en.upper()

            if status == "CLOSED" or not location_en: 
                continue
                
            active_text_block = content_upper.split("RESUMED NORMAL")[0]
            category = self.classify_incident(content_upper)
            text_pool = content_upper + " " + location_en

            # ==========================================================
            # YOUR STEP 2: Identify direction between road and bound
            # ==========================================================
            target_geom = None
            extracted_target_name = None
            
            # Find "XXX BOUND" or "TOWARDS XXX"
            bound_match = re.search(r'([A-Z\s]+)\s+BOUND', text_pool)
            if bound_match:
                extracted_target_name = bound_match.group(1).strip()
            else:
                towards_match = re.search(r'(?:TOWARDS|HEADING TO|LEADING TO)\s+([A-Z0-9\s\-]+)', content_upper)
                if towards_match:
                    extracted_target_name = towards_match.group(1).strip()

            if extracted_target_name:
                clean_target = re.split(r'\b(IS|NEAR|BETWEEN|AND|PART|THE)\b', extracted_target_name)[0].strip()
                
                # Check major districts first (Central, Chai Wan, etc.)
                for dist_name, coords in MAJOR_DISTRICTS_HK80.items():
                    if dist_name in clean_target:
                        target_geom = Point(coords)
                        break
                        
                # Fallback to landmarks/roads if not a major district
                if target_geom is None:
                    if clean_target in self.landmark_cache:
                        target_geom = self.landmark_cache[clean_target]
                    else:
                        for r in sorted(list(self.road_names_cache), key=len, reverse=True):
                            if r in clean_target:
                                target_geom = self.road_df[self.road_df['STREET_ENAME'] == r].geometry.unary_union.centroid
                                break

            # Find the main road where the incident occurred
            main_road = None
            if location_en in self.road_names_cache:
                main_road = location_en
            else:
                for r in sorted(list(self.road_names_cache), key=len, reverse=True):
                    if r in location_en:
                        main_road = r
                        break

            if location_en != "BUSY ROAD SECTIONS" and not main_road:
                continue

            # Identify the Travel Direction column (handling Shapefile truncations safely)
            dir_col = None
            for col in self.road_df.columns:
                if col in ['TRAVEL_DIR', 'TRAVEL_DIRECTION', 'TRAFFIC_DIR', 'TRAFFIC_DIRECTION', 'DIR_CODE', 'DIRECTION']:
                    dir_col = col
                    break

            # TRACK B: BROADCAST MODE
            if location_en == "BUSY ROAD SECTIONS":
                sorted_roads = sorted(list(self.road_names_cache), key=len, reverse=True)
                text_to_scan = active_text_block
                matched_any_hki_road = False
                
                for cached_road in sorted_roads:
                    if re.search(r'\b' + re.escape(cached_road) + r'\b', text_to_scan):
                        text_to_scan = text_to_scan.replace(cached_road, " __SPATIAL_MATCH__ ")
                        matched_roads = self.road_df[self.road_df['STREET_ENAME'] == cached_road]
                        
                        valid_indices = []
                        for idx, road_feat in matched_roads.iterrows():
                            geom = road_feat['GEOMETRY']
                            if geom is None: continue
                            
                            dir_val = "1"
                            if dir_col and pd.notna(road_feat[dir_col]):
                                dir_val = str(road_feat[dir_col]).strip().split('.')[0]
                                
                            # Value 1: Combined two-way road. Cannot filter, must keep.
                            if dir_val == '1':
                                valid_indices.append(idx)
                                continue
                                
                            # Value 3: One-way. Apply your Step 3 Dot Product logic!
                            if dir_val == '3' and self.is_correct_direction_vector(geom, target_geom):
                                valid_indices.append(idx)
                                
                        # Fallback: if we accidentally filtered out everything, keep all segments to be safe
                        if len(valid_indices) == 0 and not matched_roads.empty:
                            valid_indices = matched_roads.index.tolist()
                                    
                        matched_roads = matched_roads.loc[valid_indices]
                        if not matched_roads.empty:
                            for _, road_feat in matched_roads.iterrows():
                                matched_any_hki_road = True
                                spatial_features_list.append({
                                    'IncidentID': inc_id, 'RoadName': cached_road, 
                                    'RouteID': str(road_feat.get('ROUTE_ID', 'UNKNOWN')), 'geometry': road_feat['GEOMETRY']
                                })
                
                if matched_any_hki_road:
                    incident_records.append({'IncidentID': inc_id, 'Category': category, 'Location': location_en, 'Details': content_en})
                continue

            # TRACK A: LOCALIZED MODE
            matched_roads = self.road_df[self.road_df['STREET_ENAME'] == main_road]
            if matched_roads.empty:
                continue

            valid_indices = []
            for idx, road_feat in matched_roads.iterrows():
                geom = road_feat['GEOMETRY']
                if geom is None: continue
                
                dir_val = "1"
                if dir_col and pd.notna(road_feat[dir_col]):
                    dir_val = str(road_feat[dir_col]).strip().split('.')[0]
                
                if dir_val == '1':
                    valid_indices.append(idx)
                    continue
                    
                # YOUR STEP 3: Apply Dot Product logic
                if dir_val == '3' and self.is_correct_direction_vector(geom, target_geom):
                    valid_indices.append(idx)
                        
            # Prevent 100% loss. If the math dropped everything, it was likely bad data. Restore it.
            if len(valid_indices) == 0 and not matched_roads.empty:
                valid_indices = matched_roads.index.tolist()

            matched_roads = matched_roads.loc[valid_indices]

            target_road_geom = matched_roads.geometry.unary_union
            pts_to_check = []

            sorted_roads = sorted(list(self.road_names_cache), key=len, reverse=True)
            for cross_road in sorted_roads:
                if cross_road == main_road: continue
                if cross_road in active_text_block or cross_road in location_en:
                    cross_feats = self.road_df[self.road_df['STREET_ENAME'] == cross_road]
                    if not cross_feats.empty:
                        cross_geom = cross_feats.geometry.unary_union
                        intersection = target_road_geom.intersection(cross_geom)
                        if not intersection.is_empty:
                            pts_to_check.append(intersection.centroid)
                        else:
                            p1, p2 = nearest_points(target_road_geom, cross_geom)
                            pts_to_check.append(p1)

            is_junction = "JUNCTION" in content_upper or "JUNCTION" in location_en
            if is_junction and pts_to_check:
                spatial_features_list.append({
                    'IncidentID': inc_id, 'RoadName': main_road, 
                    'RouteID': 'INTERSECTION_NODE', 'geometry': pts_to_check[0]
                })
                incident_records.append({'IncidentID': inc_id, 'Category': category, 'Location': location_en, 'Details': content_en})
                continue

            if landmark_en in self.landmark_cache: pts_to_check.append(self.landmark_cache[landmark_en])
            if between_en in self.landmark_cache: pts_to_check.append(self.landmark_cache[between_en])

            is_between_incident = "BETWEEN" in content_upper or "BETWEEN" in location_en
            if is_between_incident and len(pts_to_check) >= 2:
                bounding_buffers = [MultiPoint(pts_to_check).envelope.buffer(40)]
            else:
                bounding_buffers = [pt.buffer(self.distance_threshold) for pt in pts_to_check]

            is_queueing = "TRAFFIC QUEUE" in content_upper
            spatial_match_count = 0
            
            for _, road_feat in matched_roads.iterrows():
                geom = road_feat['GEOMETRY']
                if geom is None: continue
                route_id_val = str(road_feat.get('ROUTE_ID', 'UNKNOWN'))

                def append_feat():
                    spatial_features_list.append({'IncidentID': inc_id, 'RoadName': main_road, 'RouteID': route_id_val, 'geometry': geom})

                if not bounding_buffers or is_queueing:
                    append_feat()
                    spatial_match_count += 1
                else:
                    if any(geom.intersects(buf) for buf in bounding_buffers):
                        append_feat()
                        spatial_match_count += 1
                        
            if spatial_match_count > 0:
                incident_records.append({'IncidentID': inc_id, 'Category': category, 'Location': location_en, 'Details': content_en})

        df_incidents = pd.DataFrame(incident_records).drop_duplicates(subset=['IncidentID'])
        if spatial_features_list:
            gdf_spatial = gpd.GeoDataFrame(spatial_features_list, crs=self.road_df.crs)
        else:
            gdf_spatial = gpd.GeoDataFrame(columns=['IncidentID', 'RoadName', 'RouteID', 'geometry'], crs=self.road_df.crs)
            
        active_ids = gdf_spatial['IncidentID'].unique()
        df_incidents = df_incidents[df_incidents['IncidentID'].isin(active_ids)].reset_index(drop=True)
        
        return df_incidents, gdf_spatial
