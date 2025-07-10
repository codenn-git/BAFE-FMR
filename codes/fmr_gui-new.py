## removed get-matching-images function
## Available BSG images are now displayed in the popup
## July 9, adapted new display route from andrei, though I retained the create_image_function for cleanliness

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
import matplotlib.pyplot as plt

from shapely.geometry import box
from shapely.ops import transform as shapely_transform
from pyproj import Transformer
from datetime import datetime
from PIL import Image
from rasterio.transform import xy

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from waitress import serve
import base64
from io import BytesIO

from utilv1 import Preprocessing, Filters, Morph, MeasureWidth, measure_line, Interaction, export

import matplotlib
matplotlib.use("Agg")
# ==========================================================
# Paths
shapefile_path = r"C:\Users\user-307E4B3400\OneDrive - Philippine Space Agency\SDMAD_SHARED\PROJECTS\SAKA\FMR\GUI\Master FMR\NE_master_fmr.shp"
bsg_folder = r"C:\Users\user-307E4B3400\OneDrive - Philippine Space Agency\SDMAD_SHARED\PROJECTS\SAKA\FMR\GUI\Raster images"

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

def process_tracking(fmr_id, image_path, mode='automatic'):
    """Process FMR tracking (adapted from your tracking workflow)"""
    try:
        # Get the FMR feature using the FMR_ID
        if fmr_id not in gdf.index:
            return {
                'status': 'error',
                'message': f'FMR ID {fmr_id} not found in shapefile'
            }
        
        # Get the FMR geometry from the GeoDataFrame
        fmr_geometry = gdf.loc[fmr_id].geometry
        fmr_gdf = gpd.GeoDataFrame({'geometry': fmr_geometry}, crs=gdf.crs)
        fmr_name = str(gdf.loc[fmr_id].get("name", f"FMR_{fmr_id}"))
        
        # Validate image path exists
        if not os.path.exists(image_path):
            return {
                'status': 'error',
                'message': f'Image file not found: {image_path}'
            }
        
        # Initialize preprocessing with the specific image and FMR geometry
        preprocessor = Preprocessing()
        preprocessor.reproject(image_path)
        clipped_data, clipped_transform = preprocessor.clipraster(vector_data=fmr_gdf, buffer_dist=25, bbox=True)
        
        results = {
            'status': 'success',
            'fmr_id': fmr_id,
            'fmr_name': fmr_name,
            'mode': mode,
            'image_path': image_path,
            'message': f'Tracking completed for {fmr_name} in {mode} mode'
        }
        
        if mode == 'automatic':
            # Apply your automatic processing pipeline
            filter = Filters()
            warm_raster = filter.enhance_image_warmth(clipped_data)
            stretch_raster = filter.enhance_linear_stretch(clipped_data)

            morph = Morph()
            morph_warm = morph.process(warm_raster)
            morph_stretch = morph.process(stretch_raster)                                                                  

            # Merges the applied morphed warmth and stretch function
            final_binary_raster = np.logical_or(morph_warm, morph_stretch)
            final_binary_raster = cv2.morphologyEx(final_binary_raster.astype(np.uint8), 
                                                   cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=3)

            final_clipped_data, final_clipped_transform = preprocessor.clipraster(
                                    raster_data=final_binary_raster.astype(np.uint8),
                                    transform=clipped_transform,
                                    buffer_dist=1)
            
            final_line = measure_line(final_clipped_data, final_clipped_transform, spacing=3)
            
            if final_line is not None and not final_line.empty:
                final_line_length = final_line.length.values[0]
                vector_length = fmr_geometry.length
                
                results['Current FMR Length'] = float(final_line_length)
                results['vector_length'] = float(vector_length)
                results['FMR progress'] = float((final_line_length / vector_length) * 100)
            else:
                results['Current FMR Length'] = None
                results['FMR progress'] = None
                results['message'] += ' - No road line detected'
            
        elif mode == 'manual':
            # Set up for manual interaction
            interaction = Interaction(clipped_data, fmr_gdf)
            results['message'] = f'Manual tracking mode initialized for {fmr_name}'
            results['requires_interaction'] = True
        
        return results
        
    except Exception as e:
        return {
            'status': 'error',
            'fmr_id': fmr_id if 'fmr_id' in locals() else None,
            'message': f'Error in tracking: {str(e)}'
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Error in tracking: {str(e)}'
        }

# def process_extraction(image_path, fmr_geometry, image_type='BSG'):
#     """Process road width extraction (adapted from your extraction workflow)"""
#     try:
#         # This would use your processing pipeline
#         preprocessor = Preprocessing()
#         preprocessor.reproject(image_path, fmr_geometry)
        
#         results = {
#             'status': 'success',
#             'image_type': image_type,
#             'message': f'Width extraction completed for {image_type} image'
#         }
        
#         if image_type == 'PNEO':
#             # Apply PNEO-specific processing
#             # int, tol, res = 3, 0.15, 0.3
#             # clipped_data, clipped_transform = preprocessor.clipraster(bbox=True)
#             # filter = Filters(clipped_data)
#             # cielab = filter.cielab()
#             # ... rest of PNEO processing
            
#             results['mean_width'] = 'PNEO width calculation would go here'
            
#         elif image_type == 'BSG':
#             # Apply BSG-specific processing
#             int, tol, res = 3, 0.4, 0.3
#             raster_data, _ = preprocessor.clipraster(bbox=True)
#             clipped_data, clipped_transform = preprocessor.clipraster(buffer_dist=25)

#             #Filters: warmth and linear stretch
#             filter = Filters() #NEW: filter = Filters()
#             warm_raster = filter.enhance_image_warmth(clipped_data) #NEW: warm_raster  = filter.enhance_image_warmth(clipped_data)
#             stretch_raster = filter.enhance_linear_stretch(clipped_data)  #NEW: warm_raster  = filter.enhance_linear_stretch(clipped_data)

#             #Apply Morphological Operations
#             morph = Morph()
#             morph_warm = morph.process(warm_raster)
#             morph_stretch = morph.process(stretch_raster)

#             #merges the applied morphed warmth and stretch function
#             merged_or = np.logical_or(morph_warm, morph_stretch)
#             initial_binary_raster = merged_or

#             final_binary_transform = clipped_transform
 
#             # plt.imshow(final_clipped_data, cmap="gray")
#             final_binary_raster = morph.remove_small_islands(initial_binary_raster, min_size=1000)
#             final_binary_raster = cv2.morphologyEx(final_binary_raster.astype(np.uint8), 
#                                                    cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=3)

#             measure = MeasureWidth(final_binary_raster, final_binary_transform, fmr_geometry)
#             measure.process(int=int, tol=tol, res=res)

#             results['mean_width'] = measure.clipped_transects['width'].mean()
        
#         return results
        
#     except Exception as e:
#         return {
#             'status': 'error',
#             'message': f'Error in extraction: {str(e)}'
#         }

# ==========================================================
# Original Flask Routes
# ==========================================================

"""Scan FMR and BSG images, extracting match information and save results to 'fmr_database_aina.csv'."""
# edit: if fmr_db_file exists, it shouldn't iterate over ALL the FMR features again.
# Instead compare the existing FMRs with the new ones and only iterate over the new ones.

def getDatabase():
    """Efficiently scan FMR and BSG images, log all raster-FMR matches (1 row per match), sorted numerically by FMR index and date. Skips entries that are already in the database."""

    master_fmr = shapefile_path
    bsg_folder_path = bsg_folder
    fmr_db_file = os.path.join(os.path.dirname(master_fmr), "fmr_database_aina.csv")

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

                # ✅ Skip duplicates before appending
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
    """Stretch the bands of the image for display purposes"""
    lower = np.percentile(band, lower_percent)
    upper = np.percentile(band, upper_percent)
    stretched = np.clip((band - lower) / (upper - lower), 0, 1)
    return stretched

def create_image_preview(image_path, fmr_gdf):
    try:
        preprocessor = Preprocessing()
        _,_, rep_crs, _ = preprocessor.reproject(image_path)
        clipped_data, clipped_transform = preprocessor.clipraster(vector_data=fmr_gdf, bbox=True)
        
        height, width = clipped_data.shape[1:]
        top_left = xy(clipped_transform, 1, 0, offset='ul')  # Upper-left corner
        bottom_right = xy(clipped_transform, height - 1, width - 1, offset='lr')  # Lower-right corner

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
        image = Image.fromarray(rgb_uint8)
        buf = BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)

        image_base64 = base64.b64encode(buf.read()).decode('utf-8')

        return {
            "base64": image_base64, 
            "bounds": image_bounds
        }
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
## ==========================================================

@app.route('/get_matching_images', methods=['POST'])
def get_matching_images():
    data = request.json
    fmr_id = data.get("fmr_id")
    fmr_name = str(gdf.loc[fmr_id].get("name", f"FMR_{fmr_id}"))
    fmr_db_file = os.path.join(os.path.dirname(shapefile_path), "fmr_database_aina.csv")

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

@app.route('/')
def serve_map():
    return send_file(r"C:\Users\user-307E4B3400\Desktop\BAFE FMR\fmr_interactive_map.html")  # Path changed aina


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

@app.route('/display_image', methods=['POST'])
def display_image():
    data = request.json
    fmr_id = data.get("fmr_id")
    image_path = data.get("image_path")
    
    if not fmr_id or not image_path:
        return jsonify({"status": "error", "message": "Missing FMR ID or image path"}), 400
    
    if not os.path.exists(image_path):
        return jsonify({"status": "error", "message": f"Image file not found: {image_path}"}), 400

    try:
        if fmr_id not in gdf.index:
            return jsonify({"status": "error", "message": f"FMR ID {fmr_id} not found"}), 400
        
        fmr_geometry = gdf.loc[fmr_id].geometry
        
        fmr_gdf = gpd.GeoDataFrame({"geometry": [fmr_geometry]}, crs="EPSG:4326")
          
        print(f"Processing FMR {fmr_id} with image {image_path}")
        print(f"FMR geometry CRS: {fmr_gdf.crs}")
        
        # Create image preview
        preview = create_image_preview(image_path, fmr_gdf)
        
        if preview:
            return jsonify({
                "status": "success",
                "image_data": preview["base64"],
                "bounds": preview["bounds"],
                "fmr_id": fmr_id,
                "image_path": image_path
            })
        else:
            return jsonify({"status": "error", "message": "Failed to create image preview"}), 500
            
    except Exception as e:
        print(f"Error in display_image: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/process_fmr', methods=['POST'])
def process_fmr():
    """Process the selected FMR with the chosen workflow"""
    data = request.json
    fmr_id = data.get("fmr_id")
    image_path = data.get("image_path")
    workflow_type = data.get("workflow_type")  # 'track' or 'extract'
    workflow_options = data.get("workflow_options", {})
    fmr_db_file = os.path.join(os.path.dirname(shapefile_path), "fmr_database_aina.csv")

    global selected_features, gdf

    if not all([fmr_id is not None, image_path, workflow_type]):
        return jsonify({"status": "error", "message": "Missing required parameters"}), 400
    
    try:
        # Validate that the FMR_ID is in selected_features
        if fmr_id not in selected_features:
            return jsonify({"status": "error", "message": f"FMR ID {fmr_id} is not selected"}), 400
        
        # Get image path from FMR database if not provided or validate existing path
        if not image_path or not os.path.exists(image_path):
            
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
        
        if workflow_type == 'track':
            mode = workflow_options.get('mode', 'automatic')  # 'automatic' or 'manual'
            results = process_tracking(fmr_id, image_path, mode)

            # Update fmr_database_aina.csv with new results
            if results.get('status') == 'success':
                fmr_name = str(gdf.loc[fmr_id].get("name", f"FMR_{fmr_id}"))
                if os.path.exists(fmr_db_file):
                    fmr_database = pd.read_csv(fmr_db_file)
                    row_idx = fmr_database.index[fmr_database["FMR"] == fmr_name].tolist()
                    if row_idx:
                        idx = row_idx[0]
                        if "Current FMR Length" in fmr_database.columns and results.get("Current FMR Length") is not None:
                            fmr_database.at[idx, "Current FMR Length"] = results.get("Current FMR Length")
                        if "FMR Progress" in fmr_database.columns and results.get("FMR progress") is not None:
                            fmr_database.at[idx, "FMR Progress"] = results.get("FMR progress")
                        fmr_database.to_csv(fmr_db_file, index=False)

        # deal with this extraction workflow later  
        # elif workflow_type == 'extract':
        #     image_type = workflow_options.get('image_type', 'BSG')  # 'BSG' or 'PNEO'
        #     results = process_extraction(fmr_id, image_path, image_type)
            
        else:
            return jsonify({"status": "error", "message": "Invalid workflow type"}), 400
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


def run_flask():
    """Run the Flask app using Waitress."""
    serve(app, host="127.0.0.1", port=5000)

def create_fmr_map(input_gdf=None):
    """Create an interactive FMR map and save as 'fmr_interactive_map.html'."""
    map_gdf = input_gdf if input_gdf is not None else gdf
    if map_gdf.empty:
        print("Shapefile is empty!")
        return ""

    # Load FMR database to get BSG information
    fmr_db_file = os.path.join(os.path.dirname(shapefile_path), "fmr_database_aina.csv")
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

        # Get BSG information from database
        bsg_info = ""
        if fmr_database is not None:
            # Filter database entries for this FMR
            fmr_entries = fmr_database[(fmr_database["FMR"] == fmr_name) & (fmr_database["BSG"].notna()) & (fmr_database["BSG"] != "")]
            if not fmr_entries.empty:
                bsg_info = "<b>Available BSG Images:</b><br>"
                for _, db_row in fmr_entries.iterrows():
                    if pd.notna(db_row.get("BSG")):
                        bsg_file = db_row["BSG"]
                        date_str = db_row.get("Date", "N/A")
                        # time_str = db_row.get("Time", "N/A")
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
        geo_layer_var_lines.append(f"geoLayers['{layer_name}'] = {geojson.get_name()};")

    geo_layer_script = "\n".join(geo_layer_var_lines)
    provinces = sorted(set(p.title() for p in gdf["PROV_NAME"].dropna()))
    province_options = "".join([f"<option value='{p}'>{p}</option>" for p in provinces])

    js_ui = f"""
    <link rel=\"stylesheet\" href=\"https://unpkg.com/leaflet-draw/dist/leaflet.draw.css\" />
    <script src=\"https://unpkg.com/leaflet-draw/dist/leaflet.draw.js\"></script>
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
            max-width: 250px;
        }}
        #selection-panel ul {{ max-height: 100px; overflow-y: auto; padding-left: 20px; }}
        #selection-panel select, #selection-panel button {{ width: 100%; margin-top: 6px; }}
        .clear-btn {{ background-color: #dc3545; color: white; }}
        .clear-btn:hover {{ background-color: #a71d2a; }}
        #processFMRBtn:disabled {{ background-color: #e0e0e0; color: #777777; cursor: not-allowed; }}
        .image-preview {{ max-width: 300px; max-height: 200px; margin-top: 10px; }}
    </style>
    <div id=\"selection-panel\">
        <b>Province Filter:</b>
        <select id=\"provinceSelect\" onchange=\"filterByProvince()\">
            <option value=\"All\">All</option>
            {province_options}
        </select>
        <b>Selected FMR(s):</b>
        <ul id=\"fmr-list\"></ul>
        <button id="displayImagesBtn" onclick="displaySelectedImages()" disabled>Display Images</button>
        <button onclick="downloadSelected()">Export Selected</button>
        <button class=\"clear-btn\" onclick=\"clearSelections()\">🗑 Clear</button>
        <button onclick=\"updateFMRs()\">🔄 Update FMR</button>

        <div id="dynamic-processing-panel" style="margin-top: 20px;"></div>
    </div>
    """

    # Add UI and JavaScript hook to HTML
    fmap.get_root().header.add_child(folium.Element(f"<script>window.onload = function() {{ {geo_layer_script} }}</script>"))
    fmap.get_root().html.add_child(folium.Element(js_ui))

    html_path = r"C:\Users\user-307E4B3400\Desktop\BAFE FMR\fmr_interactive_map.html" # Path changed aina 
    
    fmap.get_root().html.add_child(folium.Element("""
        <script>
            L.Map.addInitHook(function () {
                window._map = this; 
                console.log("Leaflet map initialized and exposed as window._map");
            });
        </script>
        """))

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