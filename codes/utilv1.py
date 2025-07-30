import sys
import os
import re
import shutil
import threading
import tempfile
import pandas as pd
import geopandas as gpd
import folium
import cv2
import rasterio
import numpy as np
import PIL
import base64

from shapely.geometry import box
from shapely.ops import transform as shapely_transform
from pyproj import Transformer
from datetime import datetime
from rasterio.transform import xy  # Make sure this is imported at the top

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QInputDialog
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from waitress import serve
from io import BytesIO

from utilv1 import Preprocessing, Filters, Morph, MeasureWidth, measure_line, Interaction, export

import matplotlib
matplotlib.use("Agg")
# ==========================================================
# Paths
shapefile_path = r"C:\Users\user-307E123400\OneDrive - Philippine Space Agency\SDMAD_SHARED\PROJECTS\SAKA\FMR\GUI\Master FMR\NE_master_fmr.shp"
bsg_folder = r"C:\Users\user-307E123400\OneDrive - Philippine Space Agency\SDMAD_SHARED\PROJECTS\SAKA\FMR\GUI\Raster images"
pneo_folder = bsg_folder # palceholder na lang muna

# ==========================================================
# Flask Setup
app = Flask(__name__)
CORS(app)
selected_features = []
current_fmr_data = None
current_image_data = None
lock = threading.Lock()
gdf = gpd.read_file(shapefile_path).to_crs(epsg=4326)  # Reproject to WGS84
filtered_gdf = gdf.copy()

# ==========================================================
# Processing Functions
# not yet finished, care of aina

@app.route('/process_fmr', methods=['POST'])
def process_fmr():
    """Process the selected FMR with the chosen workflow"""
    data = request.json
    fmr_id = data.get("fmr_id")
    image_path = data.get("image_path")
    workflow_type = data.get("workflow_type") # manual or automatic
    image_type = data.get("image_type")  
    fmr_db_file = os.path.join(os.path.dirname(shapefile_path), "fmr_database.csv")

    global selected_features, gdf

    if fmr_id is None or image_path is None or workflow_type is None:
        missing = []
        if fmr_id is None:
            missing.append("fmr_id")
        if not image_path:
            missing.append("image_path")
        if not workflow_type:
            missing.append("workflow_type")
        return jsonify({
            "status": "error",
            "message": f"Missing required parameter(s): {', '.join(missing)}"
        }), 400
    
    try:
        # Validate that the FMR_ID is in selected_features
        if fmr_id not in selected_features:
            return jsonify({"status": "error", "message": f"FMR ID {fmr_id} is not selected"}), 400
        
        # Get image path from FMR database if not provided or validate existing path
        if not image_path:
            # Try to recover image path from database if missing
            if os.path.exists(fmr_db_file):
                fmr_database = pd.read_csv(fmr_db_file)
                fmr_name = str(gdf.loc[fmr_id].get("name", f"FMR_{fmr_id}"))
                fmr_entry = fmr_database[fmr_database["FMR"] == fmr_name]
                if not fmr_entry.empty and pd.notna(fmr_entry.iloc[0].get("Image Path")):
                    image_paths = fmr_entry.iloc[0]["Image Path"].split(", ")
                    if image_paths:
                        image_path = image_paths[0]  # Use first available image

            if not image_path or not os.path.exists(image_path):
                return jsonify({"status": "error", "message": "No valid image path found for this FMR"}), 400

        elif not os.path.exists(image_path):
            # Provided image_path is invalid
            return jsonify({"status": "error", "message": "Provided image path does not exist"}), 400

        fmr_geom = gdf.loc[fmr_id].geometry
        
        # FIX: Create GeoDataFrame with proper CRS
        fmr_gdf = gpd.GeoDataFrame({'geometry': [fmr_geom]}, crs=gdf.crs)
        
        # Alternative: If gdf doesn't have CRS, set it explicitly
        # Assuming your data is in WGS84 (EPSG:4326) - adjust as needed
        if fmr_gdf.crs is None:
            fmr_gdf = fmr_gdf.set_crs('EPSG:4326')  # or whatever your original CRS is

        if workflow_type == 'manual':
            manual_fmrs = data.get("fmr_gdf", [])
            image_type = data.get("image_type", "BSG")
            return jsonify(*manual_processing(fmr_gdf, image_type))


        elif workflow_type == 'automatic':
            processing_result = automatic_processing(fmr_gdf, image_path, image_type)
            return jsonify(processing_result)
        
        else:
            return jsonify({"status": "error", "message": "Invalid workflow type"}), 400
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

## processing function
def automatic_processing(vector_gdf, raster_path, image_type):
    results = {}
    raster_directory = os.path.dirname(raster_path)
    master_directory = os.path.dirname(raster_directory)
    output_folder = os.path.join(os.path.dirname(os.path.dirname(master_directory)), "Outputs")

    try:
        # FIX: Ensure vector_gdf has a CRS before any operations
        if vector_gdf.crs is None:
            # Set appropriate CRS - adjust EPSG code based on your data
            vector_gdf = vector_gdf.set_crs('EPSG:32651')  # WGS84 as default
            print("Warning: No CRS found, setting to EPSG:32651")
        
        preprocessor = Preprocessing()
        preprocessor.reproject(raster_path, vector_gdf)
        
        if image_type == 'BSG':
            int, tol, res = 3, 0.15, 0.3

            clipped_data, clipped_transform = preprocessor.clipraster(vector_data=vector_gdf, buffer_dist=25) #bbox=False

            filter = Filters()
            warm_raster = filter.enhance_image_warmth(clipped_data)
            stretch_raster = filter.enhance_linear_stretch(clipped_data)

            morph = Morph()
            morph_warm = morph.process(warm_raster, a=4, b=3, ite1=2, ite2=3)
            morph_stretch = morph.process(stretch_raster, a=4, b=3, ite1=2, ite2=3)

            merged_or = np.logical_or(morph_warm, morph_stretch)
            initial_binary_raster = merged_or

        if image_type == 'PNEO':
            int, tol, res = 3, 0.15, 0.3

            clipped_data, clipped_transform = preprocessor.clipraster(vector_data=vector_gdf, bbox=True)

            filter = Filters()
            cielab = filter.cielab(clipped_data)
            
            morph = Morph()
            initial_binary_raster = morph.threshold_cielab(cielab)

        final_binary_transform = clipped_transform

        # plt.imshow(final_clipped_data, cmap="gray")
        final_binary_raster = morph.remove_small_islands(initial_binary_raster, min_size=500)
        final_binary_raster = np.squeeze(final_binary_raster)
        final_binary_raster = cv2.morphologyEx(final_binary_raster.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=3)

        measure = MeasureWidth(final_binary_raster, final_binary_transform, vector_gdf)
        transects = measure.process(int=int, tol=tol, res=res)
        road_polygon = measure.generate_polygon() #export??

        ## NEED TO ADD EXPORT lines here on designated folder...

        final_binary_raster, final_binary_transform = preprocessor.clipraster(
                            raster_data = final_binary_raster.astype(np.uint8),
                            vector_data = vector_gdf,
                            transform = clipped_transform,
                            buffer_dist = 3)
        final_binary_raster = np.squeeze(final_binary_raster)

        final_line = measure_line(final_binary_raster, final_binary_transform, spacing=3)

        if final_line is not None and not final_line.empty:
            final_line_length = final_line.length.values[0]
            
            # FIX: Ensure CRS is set before transformation
            if vector_gdf.crs is None:
                vector_gdf = vector_gdf.set_crs('EPSG:32651')
            
            vector_length = vector_gdf.to_crs("EPSG:32651").length.sum()
            
            results['Actual Length'] = float(final_line_length)
            results['Planned Length'] = float(vector_length)
            results['FMR progress'] = (final_line_length / vector_length) * 100
            results['Average Road Width'] = float(transects['width'].mean())

        else:
            results['Actual Length'] = None
            results['Planned Length'] = None
            results['FMR Progress'] = None
            results['Average Road Width'] = None
            results['message'] = 'No road line detected'

        #add export lines here later 

        return {
            "status": "success",
            "results": results
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

def manual_processing(manual_fmrs, image_type):
    raster_folder = bsg_folder if image_type == "BSG" else pneo_folder

    if not manual_fmrs:
        return {"status": "error", "message": "No manual FMRs provided"}, 400

    results = []

    for item in manual_fmrs:
        name = item.get("name")
        geometry = item.get("geometry")
        if not name or not geometry:
            results.append({
                "fmr_name": name or "Unnamed",
                "status": "error",
                "message": "Missing name or geometry."
            })
            continue

        drawn_gdf = gpd.GeoDataFrame.from_features([{
            "type": "Feature",
            "properties": {},
            "geometry": geometry
        }], crs="EPSG:4326").to_crs("EPSG:32651")

        matched_image = None
        for tif_file in os.listdir(raster_folder):
            if not tif_file.endswith(".tif"):
                continue
            tif_path = os.path.join(raster_folder, tif_file)
            with rasterio.open(tif_path) as src:
                bounds = box(*src.bounds)
                bounds = gpd.GeoSeries([bounds], crs=src.crs).to_crs("EPSG:32651").iloc[0]
                if drawn_gdf.geometry.iloc[0].intersects(bounds):
                    matched_image = tif_path
                    break

        if not matched_image:
            results.append({
                "fmr_name": name,
                "status": "error",
                "message": "No matching image found"
            })
            continue

        try:
            fmr_id = f"manual_{name.replace(' ', '_')}"
            province = "Manual"
            process_result = processing(
                drawn_gdf,
                matched_image,
                image_type,
                fmr_id,
                province,
                name
            )
            process_result["fmr_name"] = name
            results.append(process_result)

        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({
                "fmr_name": name,
                "status": "error",
                "message": str(e)
            })

    return {"status": "success", "results": results}, 200

# ==========================================================
# Original Flask Routes
# ==========================================================
def getDatabase():
    """Efficiently scan FMR and BSG images, log all raster-FMR matches (1 row per match), sorted numerically by FMR index and date. Skips entries that are already in the database."""

    master_fmr = shapefile_path
    bsg_folder_path = bsg_folder
    fmr_db_file = os.path.join(os.path.dirname(master_fmr), "fmr_database.csv")

    # Load FMRs in EPSG:32651
    fmr_gdf = gpd.read_file(master_fmr).to_crs("EPSG:32651")

    # Transformer from EPSG:4326 (raster bounds) to EPSG:32651 (FMR geometries)
    raster_to_fmr_crs = Transformer.from_crs("EPSG:4326", "EPSG:32651", always_xy=True)

    # Load existing DB (if exists)
    if os.path.exists(fmr_db_file):
        existing_df = pd.read_csv(fmr_db_file)
        existing_keys = set(
            zip(existing_df["FMR"], existing_df["BSG"], existing_df["Date"])
        )
    else:
        existing_df = pd.DataFrame()
        existing_keys = set()

    # === Part 1: Preload raster bounds and reproject to EPSG:32651 ===
    raster_bounds_dict = {}
    for tif_file in os.listdir(bsg_folder_path):
        if not tif_file.endswith("Tiff.tif"):
            continue
        tif_path = os.path.join(bsg_folder_path, tif_file)
        try:
            with rasterio.open(tif_path) as src:
                minx, miny, maxx, maxy = src.bounds
                minx_t, miny_t = raster_to_fmr_crs.transform(minx, miny)
                maxx_t, maxy_t = raster_to_fmr_crs.transform(maxx, maxy)
                reprojected_bounds = box(minx_t, miny_t, maxx_t, maxy_t)
                raster_bounds_dict[tif_file] = {
                    "path": tif_path,
                    "bounds_geom": reprojected_bounds
                }
        except Exception as e:
            print(f"Error reading {tif_file}: {e}")
            continue

    # === Part 2: For each FMR, log all raster matches ===
    results = []
    for idx, row in fmr_gdf.iterrows():
        fmr_name = str(row.get("name", f"FMR_{idx}"))
        fmr_geom = row.geometry
        planned_length = fmr_geom.length

        matched = False

        for tif_file, data in raster_bounds_dict.items():
            if fmr_geom.intersects(data["bounds_geom"]):
                matched = True
                match = re.search(r"(\d{8})-(\d{6})", tif_file)
                if match:
                    raw_date, raw_time = match.groups()
                    try:
                        dt = datetime.strptime(raw_date + raw_time, "%Y%m%d%H%M%S")
                        formatted_date = dt.strftime("%Y-%m-%d")
                        formatted_time = dt.strftime("%H:%M:%S")
                    except ValueError:
                        formatted_date = ""
                        formatted_time = ""
                else:
                    formatted_date = ""
                    formatted_time = ""

                # Skip duplicates before appending
                if (fmr_name, tif_file, formatted_date) in existing_keys:
                    continue

                results.append({
                    "FMR": fmr_name,
                    "BSG": tif_file,
                    "Date": formatted_date,
                    "Time": formatted_time,
                    "Planned FMR Length": planned_length,
                    "Current FMR Length": "",
                    "FMR Progress": "",
                    "Image Path": data["path"]
                })

        if not matched:
            # Check if this FMR already exists in DB with BSG=None
            already_exists_blank = any(
                (fmr_name == existing_fmr and pd.isna(existing_bsg))
                for existing_fmr, existing_bsg, _ in existing_keys
            )
            if not already_exists_blank:
                results.append({
                    "FMR": fmr_name,
                    "BSG": None,
                    "Date": None,
                    "Time": None,
                    "Planned FMR Length": planned_length,
                    "Current FMR Length": "",
                    "FMR Progress": "",
                    "Image Path": ""
                })

    # === Part 3: Create DataFrame and sort ===
    results_df = pd.DataFrame(results)
    
    if results_df.empty:
        print("No new FMR/BSG matches found. Skipping database update.")
        return

    # Extract numeric index from FMR names (e.g., FMR_0, FMR_10 → 0, 10)
    results_df["FMR_INDEX"] = results_df["FMR"].str.extract(r"(\d+)", expand=False).astype(int)

    # Ensure 'Date' is datetime for proper sorting
    results_df["Date"] = pd.to_datetime(results_df["Date"], errors="coerce")

    # === Part 4: Append and sort ===
    if not existing_df.empty:
        final_df = pd.concat([existing_df, results_df], ignore_index=True)
    else:
        final_df = results_df

    final_df["FMR_INDEX"] = final_df["FMR"].str.extract(r"(\d+)", expand=False).astype(int)
    final_df["Date"] = pd.to_datetime(final_df["Date"], errors="coerce")
    final_df = final_df.sort_values(by=["FMR_INDEX", "Date"])
    final_df = final_df.drop(columns=["FMR_INDEX"])

    final_df.to_csv(fmr_db_file, index=False)
    print(f"Done! FMR database saved to:\n{fmr_db_file}")


def updateFMRs(master_path):
    """Merge other FMR shapefiles into the master FMR shapefile."""
    master_dir = os.path.dirname(master_path)
    master_name = os.path.splitext(os.path.basename(master_path))[0]

    master_gdf = gpd.read_file(master_path)
    master_crs = master_gdf.crs
    gdfs_to_merge = []
    for file in os.listdir(master_dir):
        if file.endswith('.shp'):
            base_name = os.path.splitext(file)[0]
            if base_name != master_name:
                file_path = os.path.join(master_dir, file)
                try:
                    gdf = gpd.read_file(file_path)
                    if gdf.crs != master_crs:
                        gdf = gdf.to_crs(master_crs)
                    gdfs_to_merge.append(gdf)
                except Exception as e:
                    print(f"Could not read {file_path}: {e}")

    if gdfs_to_merge:
        merged_gdf = pd.concat([master_gdf] + gdfs_to_merge, ignore_index=True)
        merged_gdf.to_file(master_path)
    else:
        merged_gdf = master_gdf

    merged_folder = os.path.join(master_dir, "merged")
    os.makedirs(merged_folder, exist_ok=True)

    for file in os.listdir(master_dir):
        base_name, ext = os.path.splitext(file)
        if base_name != master_name and ext.lower() in ['.shp', '.shx', '.dbf', '.prj', '.cpg', '.qix']:
            full_file = os.path.join(master_dir, file)
            if os.path.isfile(full_file):
                shutil.move(full_file, os.path.join(merged_folder, file))

## ================= DISPLAY FUNCTIONS =============== ##

def stretch_band(band, lower_percent=2, upper_percent=98):
    lower = np.percentile(band, lower_percent)
    upper = np.percentile(band, upper_percent)

    # Prevent divide-by-zero error
    if upper == lower:
        return np.zeros_like(band, dtype=np.float32)

    stretched = np.clip((band - lower) / (upper - lower), 0, 1)
    return stretched


def create_image_preview(image_path, fmr_gdf):
    try:
        preprocessor = Preprocessing()
        rep_data, rep_transform, rep_crs = preprocessor.reproject(image_path, fmr_gdf)
        clipped_data, clipped_transform = preprocessor.clipraster(vector_data=fmr_gdf, buffer_dist=25, bbox=True)
        
        height, width = clipped_data.shape[1:]
        top_left = xy(clipped_transform, 1, 0, offset='ul')
        bottom_right = xy(clipped_transform, height - 1, width - 1, offset='lr')

        transformer = Transformer.from_crs(rep_crs, "EPSG:4326", always_xy=True)
        minx, miny = transformer.transform(*top_left)
        maxx, maxy = transformer.transform(*bottom_right)
        image_bounds = [[miny, minx], [maxy, maxx]]

        rgb = np.stack([
            np.clip(clipped_data[0], 0, 255) / 255,
            np.clip(clipped_data[1], 0, 255) / 255,
            np.clip(clipped_data[2], 0, 255) / 255
        ], axis=-1)

        rgb_uint8 = (rgb * 255).astype(np.uint8)
        image = PIL.Image.fromarray(rgb_uint8)
        buf = BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)

        image_base64 = base64.b64encode(buf.read()).decode('utf-8')

        return {
            "base64": image_base64,
            "bounds": image_bounds
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"create_image_preview failed: {str(e)}")
    
## ==========================================================

@app.route('/get_matching_images', methods=['POST'])
def get_matching_images():
    data = request.json
    fmr_id = data.get("fmr_id")
    fmr_name = str(gdf.loc[fmr_id].get("name", f"FMR_{fmr_id}"))
    fmr_db_file = os.path.join(os.path.dirname(shapefile_path), "fmr_database.csv")

    if not os.path.exists(fmr_db_file):
        return jsonify({"status": "error", "message": "FMR database not found"}), 404

    fmr_database = pd.read_csv(fmr_db_file)
    rows = fmr_database[fmr_database["FMR"] == fmr_name]
    if rows.empty:
        return jsonify({"status": "error", "message": "No image found for FMR"}), 404

    images = []
    for _, row in rows.iterrows():
        # Prefer "Image Path" if present, else fallback to "BSG"
        image_paths = []
        if pd.notna(row.get("Image Path", None)) and row["Image Path"]:
            image_paths = [p.strip() for p in str(row["Image Path"]).split(",") if p.strip()]
        elif pd.notna(row.get("BSG", None)) and row["BSG"]:
            image_paths = [row["BSG"]]

        for p in image_paths:
            if os.path.exists(p):
                images.append({"filename": os.path.basename(p), "path": p})

    if not images:
        return jsonify({"status": "error", "message": "No valid image files found for FMR"}), 404

    return jsonify({"status": "success", "images": images})

## Added 07/28 2:04; for image-available FMR visibility
@app.route('/get_fmrs_with_images', methods=['GET'])
def get_fmrs_with_images():
    fmr_db_file = os.path.join(os.path.dirname(shapefile_path), "fmr_database.csv")

    if not os.path.exists(fmr_db_file):
        return jsonify({"status": "error", "message": "FMR database not found"}), 404

    df = pd.read_csv(fmr_db_file)
    df = df[df["Image Path"].notna() & df["Image Path"].astype(str).str.strip().ne("")]

    # Extract numeric index from "FMR" column like "FMR_0"
    fmr_ids = df["FMR"].str.extract(r"FMR_(\d+)", expand=False).dropna().astype(int).unique().tolist()

    return jsonify({"status": "success", "fmr_ids": fmr_ids})

@app.route('/')
def serve_map():
    return send_file(r"C:\Users\user-307E123400\Desktop\BAFE FMR\fmr_interactive_map.html")  # Path changed aina


@app.route('/select', methods=['POST'])
def select_fmr():
    fmr_id = request.json.get("fmr_id")
    with lock:
        if fmr_id is not None and fmr_id not in selected_features:
            selected_features.append(fmr_id)
            return jsonify({"status": "selected", "selected": selected_features})
    return jsonify({"status": "error"}), 400


@app.route('/deselect', methods=['POST'])
def deselect_fmr():
    global selected_fmrs
    data = request.get_json()
    fmr_id = data.get('fmr_id')
    
    if fmr_id in selected_features:
        selected_features.remove(fmr_id)
        return jsonify({"status": "deselected"})
    else:
        return jsonify({"status": "not_selected"})
    
    
@app.route('/clear', methods=['POST'])
def clear_selections():
    with lock:
        selected_features.clear()
    return jsonify({"status": "cleared"})


@app.route('/filter', methods=['POST'])
def filter_by_province():
    province = request.json.get("province")
    global filtered_gdf
    with lock:
        if province == "All":
            filtered_gdf = gdf.copy()
        else:
            filtered_gdf = gdf[gdf["PROV_NAME"].str.lower() == province.lower()].copy()
        create_fmr_map(filtered_gdf)
        return jsonify({"status": "filtered", "count": len(filtered_gdf)})


@app.route('/update_fmr', methods=['POST'])
def update_fmr_route():
    try:
        updateFMRs(shapefile_path)
        getDatabase()
        global gdf, filtered_gdf
        gdf = gpd.read_file(shapefile_path).to_crs(epsg=32651)
        filtered_gdf = gdf.copy()
        create_fmr_map(gdf)
        return jsonify({"status": "success", "message": "FMR updated successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/export', methods=['POST'])
def export_selected():
    """
    Export selected FMRs as a zip containing GeoJSON, Shapefile, and CSV.
    The zip and files are named based on the FMR ID(s).
    Expects JSON: { "selected_ids": [list of indices] }
    """
    import zipfile
    from flask import after_this_request
    data = request.get_json()
    ids = data.get("selected_ids", []) if data else []
    print("DEBUG /export called, ids:", ids)
    if not ids:
        print("DEBUG: No FMRs selected")
        return jsonify({"status": "error", "message": "No FMR(s) selected."}), 400

    try:
        master_gdf = gpd.read_file(shapefile_path)
        ids = [int(i) for i in ids]
        ids = [i for i in ids if 0 <= i < len(master_gdf)]
        if not ids:
            print("DEBUG: Invalid FMR indices")
            return jsonify({"status": "error", "message": "Invalid FMR indices"}), 400
        selected = master_gdf.iloc[ids]

        # Determine export base name
        if len(ids) == 1:
            fmr_id = str(selected.iloc[0].get("FMR_ID", ids[0])) if "FMR_ID" in selected.columns else str(ids[0])
            base_name = f"FMR_{fmr_id}"
        else:
            if "FMR_ID" in selected.columns:
                id_list = [str(row["FMR_ID"]) for _, row in selected.iterrows()]
            else:
                id_list = [str(i) for i in ids]
            base_name = f"multiFMR_{'_'.join(id_list)}"

        export_dir = os.path.dirname(shapefile_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tempdir = tempfile.mkdtemp()

        # GeoJSON
        geojson_path = os.path.join(tempdir, f"{base_name}.geojson")
        selected.to_file(geojson_path, driver="GeoJSON")
        # CSV
        csv_path = os.path.join(tempdir, f"{base_name}.csv")
        selected.drop(columns="geometry").to_csv(csv_path, index=False)
        # Shapefile (multiple files)
        shp_base = os.path.join(tempdir, base_name)
        selected.to_file(f"{shp_base}.shp")
        # Gather all files
        files_to_zip = [geojson_path, csv_path]
        for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
            f = f"{shp_base}{ext}"
            if os.path.exists(f):
                files_to_zip.append(f)
        # Zip
        zip_path = os.path.join(export_dir, f"{base_name}_{timestamp}.zip")
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for f in files_to_zip:
                zipf.write(f, os.path.basename(f))
        print("DEBUG: Exported zip to", zip_path)

        message = f"FMR(s) exported successfully to:\n{zip_path}"

        @after_this_request
        def add_export_message_header(response):
            # Only add header if not streaming (send_file disables custom headers for streamed files)
            try:
                response.headers.add("X-Export-Message", message)
            except Exception:
                pass
            return response

        return send_file(
            zip_path,
            as_attachment=True,
            download_name=os.path.basename(zip_path),
            mimetype="application/zip"
        )
    except Exception as e:
        print("DEBUG: Exception in /export:", e)
        return jsonify({"status": "error", "message": str(e)}), 500

# ==========================================================
# New Processing Routes
# ==========================================================

@app.route('/display_selected_image', methods=['POST'])
def display_selected_image():
    data = request.get_json()
    fmr_id = data.get("fmr_id")
    image_path = data.get("image_path")

    if not os.path.exists(image_path):
        return jsonify({"status": "error", "message": "Image file not found."}), 404

    try:
        if fmr_id not in gdf.index:
            return jsonify({"status": "error", "message": f"FMR ID {fmr_id} not found"}), 400

        fmr_geometry = gdf.loc[fmr_id].geometry
        fmr_gdf = gpd.GeoDataFrame({"geometry": [fmr_geometry]}, crs="EPSG:4326")

        # Call preview function
        preview = create_image_preview(image_path, fmr_gdf)

        # Handle tuple (error) case
        if isinstance(preview, tuple):
            return preview  # Already a response

        return jsonify({
            "status": "success",
            "image_data": preview["base64"],
            "bounds": preview["bounds"],
            "fmr_id": fmr_id,
            "image_path": image_path
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

# @app.route('/get_fmr_metadata', methods=['POST'])
# def get_fmr_metadata():
#     data = request.get_json()
#     fmr_id = data.get("fmr_id")

#     if fmr_id is None or fmr_id not in gdf.index:
#         return jsonify({"status": "error", "message": "Invalid FMR ID"}), 400

#     row = gdf.loc[fmr_id]
#     return jsonify({
#         "status": "success",
#         "fmr_id": fmr_id,
#         "name": row.get("name", f"FMR_{fmr_id}"),
#         "barangay": row.get("BRGY_NAME", "N/A"),
#         "municipality": row.get("MUN_NAME", "N/A"),
#         "province": row.get("PROV_NAME", "N/A")
#     })

def run_flask():
    """Run the Flask app using Waitress."""
    serve(app, host="127.0.0.1", port=5000)

def create_fmr_map(input_gdf=None):
    map_gdf = input_gdf if input_gdf is not None else gdf
    if map_gdf.empty:
        print("Shapefile is empty!")
        return ""

    fmr_db_file = os.path.join(os.path.dirname(shapefile_path), "fmr_database.csv")
    fmr_database = None
    if os.path.exists(fmr_db_file):
        try:
            fmr_database = pd.read_csv(fmr_db_file)
        except Exception as e:
            print(f"Error loading FMR database: {e}")

    center = map_gdf.unary_union.centroid
    fmap = folium.Map(location=[center.y, center.x], zoom_start=10, tiles="Esri.WorldImagery")

    geo_layer_var_lines = []
    for idx, row in map_gdf.iterrows():
        layer_name = f"geoLayer_{idx}"
        brgy = row.get("BRGY_NAME", "N/A")
        mun = row.get("MUN_NAME", "N/A")
        prov = row.get("PROV_NAME", "N/A")
        fmr_name = str(row.get("name", f"FMR_{idx}"))

        bsg_info = ""
        if fmr_database is not None:
            fmr_entries = fmr_database[(fmr_database["FMR"] == fmr_name) & (fmr_database["BSG"].notna()) & (fmr_database["BSG"] != "")]
            if not fmr_entries.empty:
                bsg_info = "<b>Available BSG Images:</b><br>"
                for _, db_row in fmr_entries.iterrows():
                    if pd.notna(db_row.get("BSG")):
                        bsg_file = db_row["BSG"]
                        date_str = db_row.get("Date", "N/A")
                        bsg_info += f"> {bsg_file}<br>"
                        if date_str != "N/A":
                            bsg_info += f"Date: {date_str}<br>"
                        bsg_info += "<br>"
            else:
                bsg_info = "<b>BSG Images:</b> No matching images found<br><br>"

        popup_html = f"""
        <div style='word-wrap: break-word; max-width: 350px;'>
            <b>FMR ID:</b> {idx}<br>
            <b>FMR Name:</b> {fmr_name}<br>
            <b>Barangay:</b> {brgy}<br>
            <b>Municipality:</b> {mun}<br>
            <b>Province:</b> {prov}<br><br>
            {bsg_info}
            <div style="display: flex; gap: 8px; margin-top: 5px;">
                <button onclick="selectFMR({idx})">Select FMR</button>
                <button onclick="deselectFMR({idx})">Deselect FMR</button>
            </div>
        </div>
        """

        geojson = folium.GeoJson(
            row.geometry,
            name=layer_name,
            tooltip=f"FMR ID: {idx}",
            style_function=lambda feature: {"color": "yellow", "weight": 3.5},
        )
        geojson.add_child(folium.Popup(popup_html, max_width=400))
        geojson.add_to(fmap)

        geojson_js_var = geojson.get_name()
        geo_layer_var_lines.append(f"geoLayers['{layer_name}'] = {geojson_js_var};")

    geo_layer_script = "\n".join(geo_layer_var_lines)
    provinces = sorted(set(p.title() for p in gdf["PROV_NAME"].dropna()))
    province_options = "".join([f"<option value='{p}'>{p}</option>" for p in provinces])

    js_ui = f"""
        <link rel=\"stylesheet\" href=\"https://unpkg.com/leaflet-draw/dist/leaflet.draw.css\" />
        <script src=\"https://unpkg.com/leaflet-draw/dist/leaflet.draw.js\"></script>
        <link rel=\"stylesheet\" href=\"https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css\" integrity=\"sha512-papER...==\"
            crossorigin=\"anonymous\" referrerpolicy=\"no-referrer\" />
        <script src=\"/static/fmr_ui_script.js\"></script>
        
        <style>
            #selection-panel {{
                position: fixed;
                bottom: 5px;
                left: 5px;
                background: rgba(255,255,255,0.95);
                padding: 10px;
                border-radius: 8px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.3);
                z-index: 9999;
                max-width: 300px;
                overflow-x: auto;
            }}
            #selection-panel ul {{
                max-height: 100px;
                overflow-y: auto;
                padding-left: 20px;
            }}
            #selection-panel select,
            #selection-panel button {{
                width: 100%;
                margin-top: 6px;
            }}
            .clear-btn {{
                background-color: #dc3545;
                color: white;
            }}
            .clear-btn:hover {{
                background-color: #a71d2a;
            }}
            #processFMRBtn:disabled {{
                background-color: #e0e0e0;
                color: #777777;
                cursor: not-allowed;
            }}
            .image-preview {{
                max-width: 300px;
                max-height: 200px;
                margin-top: 10px;
            }}
            .image-option input[type='checkbox'][disabled] + label {{
                color: #999;
                cursor: not-allowed;
            }}

            /* ========== Modal Styling ========== */
            #processing-modal {{
                display: none;
                position: fixed;
                top: 0; left: 0;
                width: 100vw;
                height: 100vh;
                background-color: rgba(0, 0, 0, 0.4);
                z-index: 10000;
                justify-content: center;
                align-items: center;
            }}
            #processing-modal-content {{
                background: white;
                padding: 20px 25px;
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(0,0,0,0.3);
                width: 90%;
                max-width: 400px;
            }}
            #processing-modal h3 {{
                margin-top: 0;
                text-align: center;
            }}
            #processing-modal .option-group {{
                margin: 15px 0;
            }}
            #processing-modal label {{
                display: block;
                margin: 5px 0;
            }}
            #processing-modal select {{
                width: 100%;
                padding: 6px;
            }}
            #processing-modal .modal-buttons {{
                display: flex;
                justify-content: flex-end;
                gap: 10px;
                margin-top: 20px;
            }}
            #processing-modal .modal-buttons button {{
                padding: 8px 15px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }}
            #processing-modal #run-processing {{
                background-color: #28a745;
                color: white;
            }}
            #processing-modal #cancel-processing {{
                background-color: #dc3545;
                color: white;
            }}

            /* ========== Image Toggle Button ========== */
            .leaflet-top.leaflet-right .leaflet-control-image-toggle {{
                background-color: #fff;
                width: 30px;
                height: 30px;
                line-height: 30px;
                text-align: center;
                cursor: pointer;
                box-shadow: 0 1px 5px rgba(0,0,0,0.65);
                border-radius: 4px;
                margin: 10px;
                font-size: 16px;
                transition: background-color 0.2s ease;
            }}
            .leaflet-control-image-toggle:hover {{
                background-color: #f0f0f0;
            }}
            .leaflet-control-image-toggle.active {{
                background-color: #4285f4;
                color: white;
            }}
        </style>

        <!------------ Selection Panel ------------>
        <div id=\"selection-panel\">
            <b>Province Filter:</b>
            <select id=\"provinceSelect\" onchange=\"filterByProvince()\">
                <option value=\"All\">All</option>
                {province_options}
            </select>
            <b>Selected FMR(s):</b>
            <ul id=\"fmr-list\"></ul>
            <button id=\"runBtn\" onclick=\"showProcessingModal()\" disabled>Run</button>
            <button onclick=\"downloadSelected()\">Export Selected</button>
            <button class=\"clear-btn\" onclick=\"clearSelections()\">Clear</button>
            <button onclick=\"updateFMRs()\">Update FMR</button>
            <div id=\"dynamic-processing-panel\" style=\"margin-top: 20px;\"></div>
        </div>

        <!-- Processing Modal -->
        <div id=\"processing-modal\">
            <div id=\"processing-modal-content\">
                <h3>Processing Options</h3>

                <div class=\"option-group\">
                    <strong>Process:</strong>
                    <label><input type=\"radio\" name=\"process-type\" value=\"selected\" checked> Selected images only</label>
                    <label><input type=\"radio\" name=\"process-type\" value=\"all\"> All images</label>
                </div>

                <div class=\"option-group\">
                    <strong>Workflow Type:</strong>
                    <label><input type=\"radio\" name=\"workflow-type\" value=\"manual\" onchange=\"toggleManualSection()\"> Manual</label>
                    <label><input type=\"radio\" name=\"workflow-type\" value=\"automatic\" onchange=\"toggleManualSection()\" checked> Automatic</label>
                </div>

                <!-- Manual Drawing UI -->
                <div id=\"manual-fmr-section\" style=\"display: none; margin-top: 10px;\">
                    <strong>Draw FMR Centerlines:</strong>
                    <div id=\"manual-fmr-container\" style=\"margin-bottom: 10px;\"></div>
                    <button type=\"button\" onclick=\"addManualFMRRow()\">+ Draw additional FMR</button>
                </div>

                <div class=\"option-group\">
                    <strong>Image Type:</strong>
                    <select id=\"image-type\">
                        <option value=\"BSG\">BSG</option>
                        <option value=\"PNEO\">PNEO</option>
                    </select>
                </div>

                <div class=\"modal-buttons\">
                    <button id=\"cancel-processing\" onclick=\"hideProcessingModal()\">Cancel</button>
                    <button id=\"run-processing\" onclick=\"runProcessing()\">Run</button>
                </div>
            </div>
        </div>

        <!-- Image Toggle Button (Leaflet top-right) -->
        <div class=\"leaflet-top leaflet-right\">
            <div class=\"leaflet-control leaflet-bar leaflet-control-image-toggle\" title=\"Show FMRs with Satellite Images\" onclick=\"toggleImageVisibility(this)\">
                <i class=\"fas fa-image\"></i>
            </div>
        </div>
    """


    fmap.get_root().html.add_child(folium.Element(js_ui))

    fmap.get_root().html.add_child(folium.Element(f"""
        <script>
            L.Map.addInitHook(function () {{
                setTimeout(function () {{
                    {geo_layer_script}
                }}, 0);
            }});
        </script>
    """))

    fmap.get_root().html.add_child(folium.Element("""
        <script>
            L.Map.addInitHook(function () {
                window._map = this;
                console.log("Leaflet map initialized and exposed as window._map");
            });
        </script>
    """))

    html_path = "C:/Users/user-307E123400/Desktop/BAFE FMR/fmr_interactive_map.html"
    fmap.save(html_path)
    print("Interactive FMR map created: fmr_interactive_map.html")
    return os.path.abspath(html_path)

# ==========================================================
# PyQt5 GUI Application
# ==========================================================

class FMRMainWindow(QMainWindow):
    """Main window for the FMR GUI application."""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.flask_thread = None
        self.start_flask_server()

        # self.start_workflow()
    
    ## Workflow selection, 
    # def start_workflow(self):
    #     """Start the main workflow"""
    #     # Get user inputs
    #     new_ex, ok = QInputDialog.getItem(self, "Workflow Selection", 
    #                                      "Do you want to process a New or Existing Project?", 
    #                                      ["New", "Existing"], 0, False)
    #     if not ok:
    #         self.close()
    #         return

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("FMR Processing GUI")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create web view
        self.web_view = QWebEngineView()
        layout.addWidget(self.web_view)
        
        # Load the map
        self.load_map()
        
    def start_flask_server(self):
        """Start the Flask server in a separate thread."""
        if self.flask_thread is None:
            self.flask_thread = threading.Thread(target=run_flask, daemon=True)
            self.flask_thread.start()
            print("✅ Flask server started on http://127.0.0.1:5000")
        
    def load_map(self):
        """Load the FMR map in the web view."""
        # Create the initial map
        create_fmr_map()
        
        # Load the map in the web view
        map_url = QUrl("http://127.0.0.1:5000/")
        self.web_view.load(map_url)
        
    def closeEvent(self, event):
        """Handle application close event."""
        print("Closing FMR GUI application...")
        event.accept()


def main():
    """Main function to run the FMR GUI application."""
    print("Starting FMR Processing GUI...")
    
    # Initialize the database and create initial map
    print("Initializing FMR database...")
    getDatabase()
    
    print("🗺️ Creating initial FMR map...")
    create_fmr_map()
    
    # Create and run the GUI application
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("FMR Processing GUI")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Philippine Space Agency")
    
    # Create and show main window
    main_window = FMRMainWindow()
    main_window.show()
    
    print("✅ FMR GUI application ready!")
    print("📍 Access the web interface at: http://127.0.0.1:5000")
    
    # Run the application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
