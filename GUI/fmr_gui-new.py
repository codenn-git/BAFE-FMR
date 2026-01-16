# please check 8/27

from datetime import datetime  #11/07
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
from shapely.geometry import box, shape
from shapely.ops import transform as shapely_transform
from pyproj import Transformer
from datetime import datetime
from rasterio.transform import xy  # Make sure this is imported at the top

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QHBoxLayout, QLabel, QPushButton
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl, QTimer
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from waitress import serve
from io import BytesIO

from utilv3 import Preprocessing, Filters, Morph, MeasureWidth, measure_line, export
from fmr_incremental_updater import IncrementalUpdater
from fmr_processing import process_automatic, process_manual

import matplotlib
matplotlib.use("Agg")

# =================================================
# Configuration: remember user paths between runs
# =================================================

from pathlib import Path
import json

APP_ROOT    = Path(__file__).resolve().parent
CONFIG_PATH = APP_ROOT / "fmr_config.json"


def _prompt_path(description, must_be_file=False):
    """
    Ask user for a path via CMD until they give a valid one.
    """
    print("\n" + description)
    while True:
        user_input = input("> ").strip().strip('"')
        if not user_input:
            print("Please enter a path.")
            continue

        p = Path(user_input)

        if must_be_file and not p.is_file():
            print("That file does not exist. Please try again.")
            continue
        if not must_be_file and not p.is_dir():
            print("That folder does not exist. Please try again.")
            continue

        return str(p)


def load_or_create_config():
    """
    Load fmr_config.json if it exists.
    If not complete, ask user for missing paths once and save them.
    """
    config = {}

    # Try to read existing config
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception as e:
            print(f"[WARN] Could not read config file: {e}")
            config = {}

    # Ask only for things that are missing
    if not config.get("master_fmr_path"):
        config["master_fmr_path"] = _prompt_path(
            "First-time setup:\nEnter FULL PATH to the master FMR shapefile (.shp):",
            must_be_file=True,
        )

    if not config.get("images_folder"):
        config["images_folder"] = _prompt_path(
            "Enter FULL PATH to the folder containing the satellite images (TIFFs):",
            must_be_file=False,
        )

    # Where to save the interactive map HTML – default: same folder as this script
    if not config.get("map_html_path"):
        default_html = APP_ROOT / "fmr_interactive_map.html"
        config["map_html_path"] = str(default_html)

    # Save config back to disk so next run doesn't ask again
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        print(f"\n[INFO] Saved configuration to {CONFIG_PATH}")
    except Exception as e:
        print(f"[WARN] Could not save config file: {e}")

    return config


CONFIG = load_or_create_config()

# Single source of truth used everywhere below
shapefile_path = CONFIG["master_fmr_path"]
bsg_folder     = CONFIG["images_folder"]
MAP_HTML_PATH  = CONFIG["map_html_path"]

incremental_updater = None  # stores the updater
# ==========================================================
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
# ============================================================================

#11/07: manual branch saves centerline (always) + polygon (only if width given), and updates CSV
@app.route('/process_fmr', methods=['POST'])
def process_fmr():  #11/07; #Need to add Width Margin of Error
    """Process FMR with chosen workflow.
       Manual:
         1) Save drawn centerline to fmr_centerlines.geojson.
         2) If mean_width_m (or width_m) > 0 is provided, build & save polygon to fmr_polygons.geojson.
         3) Update CSV: overwrite last row when status == 'On-going', else append.
       Automatic: unchanged, uses existing processing().
    """  #11/07

    #11/07: shared output targets (match automatic)
    GEOJSON_OUTPUT = os.path.join(os.path.dirname(bsg_folder), "Outputs")  #11/07

    data = request.json  #11/07
    workflow_type = data.get("workflow_type")             # 'manual' or 'automatic'  #11/07
    image_type = data.get("image_type")                   # #11/07, 11/12 will remove this part (in both .py and .js since automatic image detection is implemented)
    fmr_id = data.get("fmr_id")                           # automatic  #11/07
    image_path = data.get("image_path")                   # automatic (or later use)  #11/07
    manual_fmr = data.get("manual_fmr")                   # manual  #11/07

    global selected_features, gdf  #11/07
    if not workflow_type:
        return jsonify({"status": "error", "message": "Missing required parameter: workflow_type"}), 400  #11/07

    try:
        # CSV database path (shared)
        fmr_db_file = os.path.join(os.path.dirname(shapefile_path), "fmr_database.csv")  #11/07

        def _ensure_csv_columns(df):  #11/07
            cols = [
                "FMR", "Current FMR Length", "FMR Progress", "FMR Status",
                "Mean FMR Width", 'FMR Width ME' "Image Type", "Processing Type", "TIMESTAMP", "Output_Paths"
            ]
            for c in cols:
                if c not in df.columns:
                    df[c] = ""
            return df  #11/07

        fmr_name = None  #11/07

        # ----------------------------
        # MANUAL
        # ----------------------------
        if workflow_type == 'manual':
            mf = manual_fmr or {}
            sel_id = mf.get("selected_fmr_id", None)
            geom_json = mf.get("geometry", None)
            image_path = data.get("image_path")

            if sel_id is None or geom_json is None:
                return jsonify({
                    "status": "error",
                    "message": "Manual workflow requires manual_fmr with selected_fmr_id and geometry"
                }), 400

            if not image_path or not os.path.exists(image_path):
                return jsonify({
                    "status": "error",
                    "message": "Manual workflow requires a valid image_path"
                }), 400

            # Get selected FMR geometry (for reference only)
            try:
                sel_row = gdf.loc[sel_id]
                fmr_name = str(sel_row.get("name", f"FMR-{sel_id}"))
                fmr_geom = sel_row.geometry
                selected_fmr_gdf = gpd.GeoDataFrame({'geometry': [fmr_geom]}, crs=gdf.crs)
                if selected_fmr_gdf.crs is None:
                    selected_fmr_gdf = selected_fmr_gdf.set_crs('EPSG:4326')
            except Exception as e:
                return jsonify({
                    "status": "error",
                    "message": f"Selected FMR id {sel_id} not found: {e}"
                }), 400

            # Process using refactored manual processor WITH IMAGE PROCESSING
            processing_result = process_manual(
                manual_centerline_geojson=geom_json,
                selected_fmr_gdf=selected_fmr_gdf,
                raster_path=image_path,  # NEW: Required for image processing
                output_base_dir=os.path.join(os.path.dirname(bsg_folder), "Outputs"),
                geojson_output_dir=GEOJSON_OUTPUT,
                image_type=image_type or "",
                fmr_name=fmr_name  # Optional, will auto-detect
            )
            
            # Update CSV database
            if processing_result.get("status") == "success":
                fmr_db_file = os.path.join(os.path.dirname(shapefile_path), "fmr_database.csv")
                if os.path.exists(fmr_db_file):
                    try:
                        df = pd.read_csv(fmr_db_file)
                        df = _ensure_csv_columns(df)
                        
                        results = processing_result.get("results", {})
                        output_paths = processing_result.get("output_paths", {})
                        
                        mask = (df["FMR"] == fmr_name) & (df["Image Path"] == image_path)
                        #11/12 this mask should update based on image_path as to consider fmrs with multiple images (rows) [11/12 - DONE]
                        
                        csv_updates = {
                            "Current FMR Length": results.get("manual_length_m"),
                            "FMR Progress": results.get("progress_percent"),
                            "FMR Status": results.get("status"),
                            "Mean FMR Width": results.get("mean_width_m", ""),  # From transects
                            "FMR Width ME": results.get("width_MoE"),
                            "Image Type": results.get("image_type", ""),  # Auto-detected
                            "Processing Type": "Manual",
                            "TIMESTAMP": results.get("TIMESTAMP")
                        }
                        
                        # Update or insert
                        status = results.get("status", "")
                        if status == "On-going" and mask.any():
                            idx = df.index[mask][-1]
                            for k, v in csv_updates.items():
                                if k in df.columns and v is not None:
                                    df.at[idx, k] = v
                        else:
                            insert_pos = (df.index[mask][-1] + 1) if mask.any() else len(df)
                            upper = df.iloc[:insert_pos]
                            lower = df.iloc[insert_pos:]
                            new_row = pd.DataFrame([csv_updates], columns=df.columns)
                            
                            #11/21: avoid FutureWarning by excluding empty slices before concat
                            frames = [upper, new_row, lower]
                            frames = [frame for frame in frames if not frame.empty]
                            df = pd.concat(frames, ignore_index=True)
                        
                        if output_paths:
                            output_paths_str = "; ".join([f"{desc}: {path}" for desc, path in output_paths.items()])
                            if mask.any():
                                df.loc[mask.iloc[-1:].index, "Output_Paths"] = output_paths_str
                        
                        df.to_csv(fmr_db_file, index=False)
                        processing_result["database_updated"] = True
                    except Exception as e:
                        processing_result["database_update_error"] = str(e)
            
            return jsonify(processing_result)

        # ----------------------------
        # AUTOMATIC (unchanged)
        # ----------------------------
        elif workflow_type == 'automatic':  #11/07
            if fmr_id is None:
                return jsonify({"status": "error", "message": "Automatic workflow requires fmr_id"}), 400  #11/07
            if fmr_id not in selected_features:
                return jsonify({"status": "error", "message": f"FMR ID {fmr_id} is not selected"}), 400  #11/07

            fmr_row = gdf.loc[fmr_id]
            fmr_name = str(fmr_row["name"]) if ("name" in fmr_row and pd.notna(fmr_row["name"])) else f"FMR-{fmr_id}"

            if not image_path:
                fmr_db_alt = os.path.join(os.path.dirname(shapefile_path), "fmr_database.csv")
                if os.path.exists(fmr_db_alt):
                    fmr_database = pd.read_csv(fmr_db_alt)
                    fmr_entry = fmr_database[fmr_database["FMR"] == fmr_name]
                    if not fmr_entry.empty and pd.notna(fmr_entry.iloc[0].get("Image Path")):
                        image_paths = fmr_entry.iloc[0]["Image Path"].split(", ")
                        if image_paths:
                            image_path = image_paths[0]
                if not image_path or not os.path.exists(image_path):
                    return jsonify({"status": "error", "message": "No valid image path found for this FMR"}), 400
            elif not os.path.exists(image_path):
                return jsonify({"status": "error", "message": "Provided image path does not exist"}), 400

            fmr_geom = gdf.loc[fmr_id].geometry
            fmr_gdf = gpd.GeoDataFrame({'geometry': [fmr_geom]}, crs=gdf.crs)
            if fmr_gdf.crs is None:
                fmr_gdf = fmr_gdf.set_crs('EPSG:4326')

            processing_result = process_automatic(
                                fmr_gdf=fmr_gdf,
                                raster_path=image_path,
                                output_base_dir=os.path.join(os.path.dirname(bsg_folder), "Outputs"),
                                geojson_output_dir=GEOJSON_OUTPUT,
                                fmr_name=fmr_name
                                )

        else:
            return jsonify({"status": "error", "message": "Invalid workflow type. Must be 'manual' or 'automatic'"}), 400  #11/07

        # ----------------------------
        # AUTOMATIC CSV update
        # ----------------------------
        if processing_result.get("status") == "success":
            fmr_db_file2 = os.path.join(os.path.dirname(shapefile_path), "fmr_database.csv")
            if os.path.exists(fmr_db_file2):
                try:
                    df = pd.read_csv(fmr_db_file2)
                    df = _ensure_csv_columns(df)

                    results = processing_result.get("results", {})
                    output_paths = processing_result.get("output_paths", {})

                    csv_updates = {
                        "Current FMR Length": results.get("length_m"),
                        "FMR Progress": results.get("progress_percent"),
                        "FMR Status": results.get("status"),
                        "Mean FMR Width": results.get("mean_width_m"),
                        "FMR Width ME": results.get("width_MoE"),
                        "Image Type": results.get("image_type"),  # Auto-detected
                        "Processing Type": "Automatic",
                        "TIMESTAMP": results.get("TIMESTAMP")
                    }

                    if image_path:
                        mask = (df["FMR"] == fmr_name) & (df["Image Path"] == image_path)
                    else:
                        mask = (df["FMR"] == fmr_name)

                    if not mask.any() and workflow_type == 'automatic':
                        alt_fmr_name = f"FMR_{fmr_id}"
                        mask = (df["FMR"] == alt_fmr_name)

                    if mask.any():
                        for column, value in csv_updates.items():
                            if column in df.columns and value is not None:
                                df.loc[mask, column] = value

                        if output_paths:
                            output_paths_str = "; ".join([f"{desc}: {path}" for desc, path in output_paths.items()])
                            df.loc[mask, "Output_Paths"] = output_paths_str
                        df.to_csv(fmr_db_file2, index=False)
                        processing_result["database_updated"] = True
                        processing_result["updated_rows"] = int(mask.sum())
                        processing_result["output_paths_added"] = len(output_paths) if output_paths else 0
                    else:
                        processing_result["database_update_warning"] = f"No matching rows found for FMR: {fmr_name}"
                except Exception as e:
                    processing_result["database_update_error"] = str(e)

        return jsonify(processing_result)  #11/07

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Processing failed: {str(e)}"}), 500  #11/07

#11/21: fetch the latest manual centerline geometry for a given selected_fmr_id
@app.route('/get_manual_centerline', methods=['POST'])
def get_manual_centerline():
    data = request.json or {}
    sel_id = data.get("selected_fmr_id", None)

    if sel_id is None:
        return jsonify({
            "status": "error",
            "message": "selected_fmr_id is required"
        }), 400

    # Resolve FMR name from shapefile (mirror process_fmr manual logic)
    try:
        # sel_id may come as string or int; try both
        try:
            fmr_row = gdf.loc[sel_id]
        except KeyError:
            try:
                fmr_row = gdf.loc[int(sel_id)]
            except Exception:
                return jsonify({
                    "status": "error",
                    "message": f"FMR with id {sel_id} not found in shapefile"
                }), 404

        if "name" in fmr_row and pd.notna(fmr_row["name"]):
            fmr_name = str(fmr_row["name"])
        else:
            fmr_name = f"FMR-{sel_id}"
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error resolving FMR name for id {sel_id}: {e}"
        }), 500

    # Look up latest manual centerline from consolidated GeoJSON
    try:
        centerlines_path = os.path.join(
            os.path.dirname(bsg_folder), "Outputs", "fmr_centerlines.geojson"
        )
        if not os.path.exists(centerlines_path):
            return jsonify({
                "status": "error",
                "message": "No manual centerlines file found yet."
            }), 404

        manual_gdf = gpd.read_file(centerlines_path)
        if manual_gdf.empty or "FMR_ID" not in manual_gdf.columns:
            return jsonify({
                "status": "error",
                "message": "No manual centerline entries found."
            }), 404

        subset = manual_gdf[manual_gdf["FMR_ID"] == fmr_name]

        # Prefer only manual processing type if column exists
        if "processing_type" in subset.columns:
            subset = subset[subset["processing_type"].astype(str).str.lower() == "manual"]

        if subset.empty:
            return jsonify({
                "status": "error",
                "message": f"No manual centerline found for FMR '{fmr_name}'."
            }), 404

        # If TIMESTAMP exists, pick latest; otherwise last row
        if "TIMESTAMP" in subset.columns:
            subset = subset.copy()
            subset["__dt"] = pd.to_datetime(subset["TIMESTAMP"], errors="coerce")
            subset = subset.sort_values("__dt")
            row = subset.iloc[-1]
        else:
            row = subset.iloc[-1]

        geom = row.geometry
        if geom is None:
            return jsonify({
                "status": "error",
                "message": "Manual centerline has no geometry."
            }), 500

        geom_geojson = geom.__geo_interface__
        return jsonify({
            "status": "success",
            "geometry": geom_geojson,
            "fmr_name": fmr_name
        })
    except Exception as e:
        print("Error in get_manual_centerline:", e)
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# ==========================================================
# Database (CSV)
# ==========================================================

def getDatabase():
    """Efficiently scan FMR and BSG images, log all raster-FMR matches (1 row per match),
    sorted numerically by FMR index and date. Skips entries that are already in the database.
    Uses sampling along the FMR line (instead of polygons) to check for nodata coverage.
    Rejects if any part of the line touches nodata.
    """

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
        run_keys = set()

    # 01/06/2026: === Part 1: Preload raster bounds and reproject to EPSG:32651 ===
    raster_bounds_dict = {}

    for root, _, files in os.walk(bsg_folder_path):
        for tif_file in files:
            n = tif_file.lower()

            # accept .tif/.tiff, skip aux sidecars
            if not n.endswith((".tif", ".tiff")):
                continue
            if n.endswith((".tif.aux", ".tif.aux.xml", ".tiff.aux", ".tiff.aux.xml")):
                continue

            # optional: keep your original intent (only BSG "Tiff" products)
            # comment this out if you want ALL tif/tiff files included
            if "tiff" not in n:
                continue

            tif_path = os.path.join(root, tif_file)

            try:
                with rasterio.open(tif_path) as src:
                    minx, miny, maxx, maxy = src.bounds

                    # more robust than assuming EPSG:4326
                    if src.crs is None:
                        continue
                    raster_to_fmr_crs = Transformer.from_crs(src.crs, fmr_gdf.crs, always_xy=True)

                    minx_t, miny_t = raster_to_fmr_crs.transform(minx, miny)
                    maxx_t, maxy_t = raster_to_fmr_crs.transform(maxx, maxy)
                    reprojected_bounds = box(minx_t, miny_t, maxx_t, maxy_t)

                    # use full path as key to avoid collisions across subfolders
                    raster_bounds_dict[tif_path] = {
                        "file": tif_file,
                        "path": tif_path,
                        "bounds_geom": reprojected_bounds
                    }

            except Exception as e:
                print(f"Error reading {tif_path}: {e}")
                continue

    # === Part 2: For each FMR, log all raster matches ===
    results = []
    for idx, row in fmr_gdf.iterrows():
        fmr_name = str(row.get("name", f"FMR-{idx}"))
        fmr_geom = row.geometry
        planned_length = fmr_geom.length

        matched = False

        for tif_path, data in raster_bounds_dict.items():
            tif_file = data["file"]
            tif_path = data["path"]


            try:
                with rasterio.open(tif_path) as src:
                    # Quick reject: if no bbox intersection
                    if not fmr_geom.intersects(data["bounds_geom"]):
                        continue

                    # 08/27 no data pixel check: reproject FMR into raster CRS
                    geom_proj = gpd.GeoSeries([fmr_geom], crs=fmr_gdf.crs).to_crs(src.crs)
                    fmr_line = geom_proj.iloc[0]

                    # 08/27 no data pixel check: densify line into points
                    N = 10  # meters between sample points
                    num_segments = max(2, int(fmr_line.length / N))
                    sample_points = [
                        fmr_line.interpolate(dist) 
                        for dist in np.linspace(0, fmr_line.length, num_segments)
                    ]
                    coords = [(pt.x, pt.y) for pt in sample_points]

                    # Sample raster at those coordinates
                    values = list(src.sample(coords))

                    # Reject if any point lies on nodata
                    nodata_val = src.nodata if src.nodata is not None else 0
                    if any(val[0] == nodata_val or val[0] == 0 for val in values):
                        continue

            except Exception as e:
                print(f"Error validating {tif_file} with FMR {fmr_name}: {e}")
                continue

            # If we reach here → real image fully covers the FMR
            matched = True
            match = re.search(r"(\d{8})-(\d{6})", tif_file)
            if match:
                raw_date, raw_time = match.groups()
                try:
                    dt = datetime.strptime(raw_date + raw_time, "%Y%m%d%H%M%S")
                    formatted_date = dt.strftime("%Y-%m-%d")
                    formatted_time = dt.strftime("%H:%M:%S")
                except ValueError:
                    formatted_date, formatted_time = "", ""
            else:
                formatted_date, formatted_time = "", ""

            # Skip duplicates before appending
            k = (fmr_name, tif_file, formatted_date)

            if k in existing_keys or k in run_keys:
                continue

            results.append({
                "FMR": fmr_name,
                "BSG": tif_file,
                "Date": formatted_date,
                "Time": formatted_time,
                "Planned FMR Length": planned_length,
                "Current FMR Length": "",
                "FMR Progress": "",
                "FMR Status": "",
                "Mean FMR Width": "",
                "FMR Width ME": "",
                "Processing Type": "",
                "Image Path": tif_path
            })

            run_keys.add(k)

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
                    "FMR Status": "",
                    "Mean FMR Width": "",
                    "FMR Width ME": "",
                    "Processing Type": "",
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
    #----- 11/11 forced adding "Processing Type" column, removed for now
    # if not existing_df.empty:
    #     if "Processing Type" not in existing_df.columns:
    #         existing_df["Processing Type"] = ""
    #     final_df = pd.concat([existing_df, results_df], ignore_index=True)
    # else:
    #     final_df = results_df
    #----- end

    # 01/06/2026: Try ko lang ito HAHAHAHAHA
    if not existing_df.empty:
        for col in existing_df.columns:
            if col not in results_df.columns:
                results_df[col] = ""
        for col in results_df.columns:
            if col not in existing_df.columns:
                existing_df[col] = ""

        final_df = pd.concat([existing_df, results_df], ignore_index=True)
    else:
        final_df = results_df.copy()

    final_df["FMR_INDEX"] = final_df["FMR"].str.extract(r"(\d+)", expand=False).astype(int)
    final_df["Date"] = pd.to_datetime(final_df["Date"], errors="coerce")
    final_df = final_df.sort_values(by=["FMR_INDEX", "Date"])
    final_df = final_df.drop(columns=["FMR_INDEX"])

    final_df.to_csv(fmr_db_file, index=False)
    print(f"Done! FMR database saved to:\n{fmr_db_file}")

    
## ================= 08/29: UPDATING DATABASE FUNCTIONS =============== ##
    
def updateFMRs(master_path):
    """Merge other FMR shapefiles into the master FMR shapefile with data type cleaning."""
    master_dir = os.path.dirname(master_path)
    master_name = os.path.splitext(os.path.basename(master_path))[0]

    master_gdf = gpd.read_file(master_path)
    master_crs = master_gdf.crs
    gdfs_to_merge = []
    
    # Clean the master GDF first
    master_gdf = clean_gdf_for_shapefile(master_gdf)
    
    for file in os.listdir(master_dir):
        if file.endswith('.shp'):
            base_name = os.path.splitext(file)[0]
            if base_name != master_name:
                file_path = os.path.join(master_dir, file)
                try:
                    gdf = gpd.read_file(file_path)
                    if gdf.crs != master_crs:
                        gdf = gdf.to_crs(master_crs)
                    
                    # Clean the GDF before merging
                    gdf = clean_gdf_for_shapefile(gdf)
                    gdfs_to_merge.append(gdf)
                except Exception as e:
                    print(f"Could not read {file_path}: {e}")

    if gdfs_to_merge:
        merged_gdf = pd.concat([master_gdf] + gdfs_to_merge, ignore_index=True)
        # Clean the merged GDF as well
        merged_gdf = clean_gdf_for_shapefile(merged_gdf)
    else:
        merged_gdf = master_gdf

    try:
        merged_gdf.to_file(master_path)
        print(f"Successfully updated master shapefile: {master_path}")
    except Exception as e:
        print(f"Error writing shapefile: {e}")
        # Try with a more restrictive schema
        try:
            write_shapefile_with_schema(merged_gdf, master_path)
        except Exception as e2:
            print(f"Failed to write with custom schema: {e2}")
            raise e2

    # Move processed files to merged folder
    merged_folder = os.path.join(master_dir, "merged")
    os.makedirs(merged_folder, exist_ok=True)

    for file in os.listdir(master_dir):
        base_name, ext = os.path.splitext(file)
        if base_name != master_name and ext.lower() in ['.shp', '.shx', '.dbf', '.prj', '.cpg', '.qix']:
            full_file = os.path.join(master_dir, file)
            if os.path.isfile(full_file):
                try:
                    shutil.move(full_file, os.path.join(merged_folder, file))
                except Exception as e:
                    print(f"Warning: Could not move {file}: {e}")

def clean_gdf_for_shapefile(gdf):
    """Clean GeoDataFrame to ensure compatibility with shapefile format."""
    # Create a copy to avoid modifying the original
    cleaned_gdf = gdf.copy()
    
    for column in cleaned_gdf.columns:
        if column == 'geometry':
            continue
            
        col_data = cleaned_gdf[column]
        
        # Check if column contains bytes objects
        if col_data.dtype == object:
            # Check if any values are bytes
            has_bytes = any(isinstance(val, bytes) for val in col_data.dropna())
            if has_bytes:
                print(f"Converting bytes column '{column}' to string")
                # Convert bytes to string, handle NaN values
                cleaned_gdf[column] = col_data.apply(
                    lambda x: x.decode('utf-8', errors='ignore') if isinstance(x, bytes) 
                    else str(x) if pd.notna(x) else None
                )
        
        # Handle other problematic data types
        elif col_data.dtype.name.startswith('datetime'):
            # Convert datetime to string for shapefile compatibility
            print(f"Converting datetime column '{column}' to string")
            cleaned_gdf[column] = col_data.dt.strftime('%Y-%m-%d %H:%M:%S')
            
        elif col_data.dtype.name in ['complex64', 'complex128']:
            # Convert complex numbers to string
            print(f"Converting complex column '{column}' to string")
            cleaned_gdf[column] = col_data.astype(str)
            
        # Ensure string columns don't exceed shapefile field width limits
        if cleaned_gdf[column].dtype == object:
            # Check for overly long strings and truncate if necessary
            max_length = 254  # DBF field limit
            if cleaned_gdf[column].astype(str).str.len().max() > max_length:
                print(f"Truncating long strings in column '{column}' to {max_length} characters")
                cleaned_gdf[column] = cleaned_gdf[column].astype(str).str.slice(0, max_length)
    
    return cleaned_gdf

def write_shapefile_with_schema(gdf, output_path):
    """Write shapefile with explicitly defined schema to avoid data type issues."""
    import fiona
    from fiona.crs import from_epsg
    
    # Define schema with safe data types
    schema = {
        'geometry': 'LineString',  # Assuming FMRs are line features
        'properties': {}
    }
    
    # Examine each column and assign appropriate schema type
    for column in gdf.columns:
        if column == 'geometry':
            continue
            
        col_data = gdf[column].dropna()
        if len(col_data) == 0:
            schema['properties'][column] = 'str:254'
            continue
            
        # Sample a few values to determine type
        sample_val = col_data.iloc[0] if len(col_data) > 0 else None
        
        if pd.api.types.is_numeric_dtype(gdf[column]):
            if pd.api.types.is_integer_dtype(gdf[column]):
                schema['properties'][column] = 'int:10'
            else:
                schema['properties'][column] = 'float:19.11'
        else:
            # Default to string for everything else
            schema['properties'][column] = 'str:254'
    
    # Get CRS
    crs = gdf.crs
    if crs is None:
        crs = from_epsg(4326)  # Default to WGS84
    
    # Write the shapefile
    with fiona.open(
        output_path,
        'w',
        driver='ESRI Shapefile',
        crs=crs,
        schema=schema
    ) as output:
        for idx, row in gdf.iterrows():
            # Prepare properties dict with safe values
            properties = {}
            for column in gdf.columns:
                if column == 'geometry':
                    continue
                
                value = row[column]
                if pd.isna(value):
                    properties[column] = None
                elif isinstance(value, bytes):
                    properties[column] = value.decode('utf-8', errors='ignore')
                else:
                    properties[column] = value
            
            # Create feature
            feature = {
                'geometry': row['geometry'].__geo_interface__,
                'properties': properties
            }
            
            output.write(feature)
    
    print(f"Successfully wrote shapefile with custom schema: {output_path}")


## ================= DISPLAY FUNCTIONS =============== ##

def create_image_preview(image_path, fmr_gdf): 
    try:
        preprocessor = Preprocessing()
        _,_, rep_crs, _ = preprocessor.reproject(image_path)
        clipped_data, clipped_transform = preprocessor.clipraster(vector_data=fmr_gdf, buffer_dist=25, bbox=True)
        
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
        return jsonify({"status": "error", "message": str(e)}), 500
    
## ==========================================================

@app.route('/get_matching_images', methods=['POST'])
def get_matching_images():
    data = request.json
    fmr_id = data.get("fmr_id")
    fmr_name = str(gdf.loc[fmr_id].get("name", f"FMR-{fmr_id}"))
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
                images.append({
                    "filename": os.path.basename(p),
                    "path": p,
                    "date": row.get("Date", "")  # 08/27: added date so JS can display it
                })

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
    fmr_ids = df["FMR"].str.extract(r"FMR-(\d+)", expand=False).dropna().astype(int).unique().tolist()

    return jsonify({"status": "success", "fmr_ids": fmr_ids})

@app.route('/')
def serve_map():
    if not os.path.exists(MAP_HTML_PATH):
        create_fmr_map()
    return send_file(os.path.abspath(MAP_HTML_PATH))


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


@app.route('/filter_by_province', methods=['POST'])
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
            base_name = f"FMR-{fmr_id}"
        else:
            if "FMR_ID" in selected.columns:
                id_list = [str(row["FMR_ID"]) for _, row in selected.iterrows()]
            else:
                id_list = [str(i) for i in ids]
            base_name = f"multiFMR-{'-'.join(id_list)}"

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
            # Only          header if not streaming (send_file disables custom headers for streamed files)
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
    image_name = data.get("BSG")

    if not os.path.exists(image_path):
        return jsonify({"status": "error", "message": "Image file not found."}), 404

    try:
        if fmr_id not in gdf.index:
            return jsonify({"status": "error", "message": f"FMR ID {fmr_id} not found"}), 400
        
        fmr_geometry = gdf.loc[fmr_id].geometry
        fmr_gdf = gpd.GeoDataFrame({"geometry": [fmr_geometry]}, crs="EPSG:4326")
        
        # Create image preview clipped to this FMR
        preview = create_image_preview(image_path, fmr_gdf)
        
        if preview:
            # 08/27: Generate unique overlay key (FMR + image)
            overlay_key = f"{fmr_id}_{os.path.basename(image_path)}"

            return jsonify({
                "status": "success",
                "image_data": preview["base64"],
                "bounds": preview["bounds"],
                "fmr_id": fmr_id,
                "image_name": image_name,
                "image_path": image_path,
                "overlay_key": overlay_key  # 08/27: send back to frontend
            })
        else:
            return jsonify({"status": "error", "message": "Failed to create image preview"}), 500
            
    except Exception as e:
        print(f"Error in display_image: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

def run_flask():
    """Run the Flask app using Waitress."""
    serve(app, host="127.0.0.1", port=5000)

#11/21: add manual centerline overlay (GeoJSON) and legend to the generated map
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

    # --- Province filter: auto-pan/zoom to current (possibly filtered) dataset ---
    try:
        minx, miny, maxx, maxy = map_gdf.total_bounds
        if np.isfinite([minx, miny, maxx, maxy]).all() and (minx != maxx) and (miny != maxy):
            fmap.fit_bounds([[miny, minx], [maxy, maxx]])
    except Exception as e:
        print(f"[create_fmr_map] fit_bounds skipped: {e}")

    geo_layer_var_lines = []
    processing_info_js_lines = []  #11/23: per-FMR processing type/status for front-end filters
    
    for idx, row in map_gdf.iterrows():
        layer_name = f"geoLayer_{idx}"
        brgy = row.get("BRGY_NAME", "N/A")
        mun = row.get("MUN_NAME", "N/A")
        prov = row.get("PROV_NAME", "N/A")
        fmr_name = str(row.get("name", f"FMR-{idx}"))

        bsg_info = ""
        if fmr_database is not None:
            fmr_entries = fmr_database[(fmr_database["FMR"] == fmr_name) & (fmr_database["BSG"].notna()) & (fmr_database["BSG"] != "")]

            # 08/22: Should filter out "manual" Processing types to avoid displaying duplicate image names in GUI
            if "Processing Type" in fmr_database.columns:
                fmr_entries = fmr_entries[~fmr_entries["Processing Type"].astype(str).str.lower().eq("manual")]
            
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

        #11/23: compute latest processing type + status for this FMR (for filters)
        proc_type_js = ""
        status_js = ""

        if fmr_database is not None and "FMR" in fmr_database.columns:
            try:
                # Match rows in DB for this FMR name
                entries = fmr_database[fmr_database["FMR"].astype(str) == fmr_name]
                if not entries.empty:
                    latest = entries

                    # Prefer most recent TIMESTAMP if available
                    if "TIMESTAMP" in entries.columns:
                        try:
                            #11/26: parse TIMESTAMP safely for mixed formats
                            # - legacy:  dd/mm/yyyy HH:MM        (dayfirst=True)
                            # - new:     yyyy-mm-dd HH:MM:SS      (explicit format)
                            ts_raw = entries["TIMESTAMP"].astype(str)

                            # Rows that look like YYYY-MM-DD...
                            iso_mask = ts_raw.str.match(r"\d{4}-\d{2}-\d{2}")

                            # Parse ISO-style timestamps with explicit format (no dayfirst)
                            ts_iso = pd.to_datetime(
                                ts_raw.where(iso_mask),
                                errors="coerce",
                                format="%Y-%m-%d %H:%M:%S",
                            )

                            # Parse legacy ones with dayfirst=True (dd/mm/yyyy HH:MM)
                            ts_legacy = pd.to_datetime(
                                ts_raw.where(~iso_mask),
                                errors="coerce",
                                dayfirst=True,
                            )

                            # Combine: prefer ISO parse, fall back to legacy
                            ts = ts_iso.fillna(ts_legacy)

                            if ts.notna().any():
                                latest = entries.loc[[ts.idxmax()]]
                            else:
                                latest = entries.iloc[[-1]]
                        except Exception:
                            latest = entries.iloc[[-1]]
                    else:
                        latest = entries.iloc[[-1]]

                    latest_row = latest.iloc[0]

                    raw_type = str(latest_row.get("Processing Type", "")).strip().lower()
                    raw_status = str(latest_row.get("FMR Status", "")).strip().lower()

                    if "manual" in raw_type:
                        proc_type_js = "manual"
                    elif "auto" in raw_type:
                        proc_type_js = "automatic"

                    if "complete" in raw_status:
                        status_js = "completed"
                    elif "on-going" in raw_status or "ongoing" in raw_status:
                        status_js = "on-going"
            except Exception as e:
                print(f"Warning building processing info for {fmr_name}: {e}")

        processing_info_js_lines.append(
            f"fmrProcessingInfo[{idx}] = {{ processingType: '{proc_type_js}', status: '{status_js}' }};"
        )

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
        geo_layer_var_lines.append(f"geoLayers['geoLayer_{idx}'] = {geojson_js_var};")


    # 11/26: compute JS registration for base FMR layers (moved outside loop to avoid duplication/lag)
    geo_layer_script = "\n".join(geo_layer_var_lines)

    # 11/26: build script for centerline overlay layers (manual + automatic) from consolidated GeoJSON
    # (moved outside the per-FMR loop so we only load and draw them once)
    manual_layer_var_lines = []
    centerline_meta_js_lines = []  # per-layer meta: FMR ID + processing_type for filters

    try:
        manual_centerlines_path = os.path.join(
            os.path.dirname(bsg_folder), "Outputs", "fmr_centerlines.geojson"
        )
        if os.path.exists(manual_centerlines_path):
            manual_gdf = gpd.read_file(manual_centerlines_path)

            for m_idx, m_row in manual_gdf.iterrows():
                m_layer_name = f"manualLayer_{m_idx}"

                # Normalise processing_type
                proc_type = str(m_row.get("processing_type", "")).strip().lower()
                if proc_type not in ("manual", "automatic"):
                    # skip weird/empty rows – treated as unprocessed for now
                    continue

                # Try to extract numeric FMR ID (e.g., FMR-123, FMR_123, "123")
                raw_fmr = str(m_row.get("FMR_ID", "")).strip()
                fmr_num = None
                try:
                    m = re.search(r"(\d+)", raw_fmr)
                    if m:
                        fmr_num = int(m.group(1))
                except Exception:
                    pass

                # Color by STATUS to match the status legend (Completed=green, On-going=red)
                raw_status = str(m_row.get("status", "")).strip().lower()

                status_js = "unknown"
                if "complete" in raw_status:
                    status_js = "completed"
                elif "on-going" in raw_status or "ongoing" in raw_status:
                    status_js = "on-going"

                if status_js == "completed":
                    color = "#2ecc71"   # green
                elif status_js == "on-going":
                    color = "#e74c3c"   # red
                else:
                    color = "#7f8c8d"   # gray for unknown

                # Line style shows processing method: solid = automatic, dashed = manual
                dash_array = "8,6" if proc_type == "manual" else None

                def _centerline_style(_feature, color=color, dash_array=dash_array):
                    style = {"color": color, "weight": 3.0, "opacity": 0.95}
                    if dash_array:
                        style["dashArray"] = dash_array
                    return style

                mj = folium.GeoJson(
                    m_row.geometry,
                    name=m_layer_name,
                    tooltip=f"Centerline ({proc_type.title()}, {status_js.replace('-', ' ').title()}): {raw_fmr or 'N/A'}",
                    style_function=_centerline_style,
                )

                mj.add_to(fmap)
                mj_js_var = mj.get_name()

                # Register Leaflet layer in JS
                manual_layer_var_lines.append(
                    f"manualCenterlineLayers['{m_layer_name}'] = {mj_js_var};"
                )

                # Also register per-layer meta so filters can use processing_type + FMR ID
                if fmr_num is not None:
                    centerline_meta_js_lines.append(
                        "manualCenterlineMeta['{name}'] = "
                        "{{ fmrId: {fid}, processingType: '{ptype}' }};".format(
                            name=m_layer_name,
                            fid=fmr_num,
                            ptype=proc_type,
                        )
                    )

    except Exception as e:
        print(f"Error loading manual/automatic centerlines: {e}")

    manual_layer_script = "\n".join(manual_layer_var_lines + centerline_meta_js_lines)

    # Provinces/options only need to be computed once as well
    provinces = sorted(set(p.title() for p in gdf["PROV_NAME"].dropna()))
    province_options = "".join([f"<option value='{p}'>{p}</option>" for p in provinces])

    js_ui = f"""
        <link rel="stylesheet" href="https://unpkg.com/leaflet-draw/dist/leaflet.draw.css" />
        <script src="https://unpkg.com/leaflet-draw/dist/leaflet.draw.js"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" 
            crossorigin="anonymous" referrerpolicy="no-referrer" />
        
        <!-- Include the updated JavaScript -->
        <script src="/static/fmr_ui_script.js"></script>
        
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
                width: 300px;
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
            #runBtn:disabled {{
                background-color: #e0e0e0 !important;
                color: #777777 !important;
                cursor: not-allowed !important;
            }}
            #clearBtn:disabled {{
                background-color: #e0e0e0 !important;
                color: #777777 !important;
                cursor: not-allowed !important;
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
            
            /* Processing Modal styles */
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
            
            /* Image Toggle Button */
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
            
            .draw-fmr-btn, .delete-fmr-btn {{
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 6px 8px;
                font-size: 14px;
                border-radius: 4px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            .draw-fmr-btn i, .delete-fmr-btn i {{
                pointer-events: none;
            }}
            .delete-fmr-btn {{
                background-color: #dc3545;
            }}
            .draw-fmr-btn:hover {{
                background-color: #45a049;
            }}
            .delete-fmr-btn:hover {{
                background-color: #a71d2a;
            }}
            
            #selected-fmrs-panel {{
                position: fixed;
                bottom: 20px;
                right: 5px;
                background: rgba(255,255,255,0.95);
                padding: 10px;
                border-radius: 8px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.3);
                z-index: 9999;
                max-width: 300px;
                max-height: 50vh;
                overflow-y: auto;
            }}
            
            /* Update notification badge */
            .update-badge {{
                background-color: #ff9800;
                color: white;
                padding: 2px 6px;
                border-radius: 10px;
                font-size: 11px;
                margin-left: 5px;
                animation: pulse 2s infinite;
            }}
        </style>

        <!------------ Selection Panel ------------>
        <div id="selection-panel">
            <b>Province Filter:</b>
            <select id="provinceSelect" onchange="filter_by_province()">
                <option value="All">All</option>
                {province_options}
            </select>
            <button onclick="downloadSelected()">Export Selected</button>
            
            <!-- Update status will be added dynamically by JavaScript -->
            <div id="dynamic-processing-panel" style="margin-top: 20px;"></div>
        </div>

        <!-- Selected FMRs Panel -->
        <div id="selected-fmrs-panel">
            <b>Selected FMR(s):</b>
            <ul id="fmr-list"></ul>
            
            <button id="runBtn" onclick="showProcessingModal()" disabled 
                    style="width: 100%; margin-top: 10px; background-color: #28a745; color: white; border: none; padding: 6px; border-radius: 4px; cursor: pointer;">
                Run
            </button>
            <button id="clearBtn" onclick="clearSelections()" disabled
                    style="width: 100%; margin-top: 6px; background-color: #dc3545; color: white; border: none; padding: 6px; border-radius: 4px; cursor: pointer;">
                Clear
            </button>
        </div>

        <!-- Processing Modal -->
        <div id="processing-modal">
            <div id="processing-modal-content">
                <h3>Processing Options</h3>

                <div class="option-group">
                    <strong>Process:</strong>
                    <label><input type="radio" name="process-type" value="selected" checked> Selected images only</label>
                    <label><input type="radio" name="process-type" value="all"> All images</label>
                </div>

                <div class="option-group">
                    <strong>Workflow Type:</strong>
                    <label><input type="radio" name="workflow-type" value="manual" onchange="toggleManualSection()"> Manual</label>
                    <label><input type="radio" name="workflow-type" value="automatic" onchange="toggleManualSection()" checked> Automatic</label>
                </div>
                
                <!-- Manual Drawing UI -->
                <div id="manual-fmr-section" style="display: none; margin-top: 10px;">
                    <strong>Draw FMR Centerlines:</strong>
                    <div id="manual-fmr-container" style="margin-bottom: 10px;"></div>
                    <button type="button" onclick="addManualFMRRow()">+ Add FMR</button>
                </div>

                <div class="option-group">
                    <strong>Image Type:</strong>
                    <select id="image-type">
                        <option value="BSG">BSG</option>
                        <option value="PNEO">PNEO</option>
                        <option value="SkySat">SkySat</option>
                    </select>
                </div>

                <div class="modal-buttons">
                    <button id="cancel-processing" onclick="hideProcessingModal()">Close</button>
                    <button id="run-processing" onclick="runProcessing()">Run</button>
                </div>
            </div>
        </div>

        <!--11/23: FMR Filters Button + Panel (top-right) -->
        <div class="leaflet-top leaflet-right">
            <!-- Toggle button -->
            <div id="fmr-filter-toggle"
                 class="leaflet-control leaflet-bar leaflet-control-custom"
                 title="Show/hide FMR filters"
                 onclick="toggleFMRFilterPanel()"
                 style="background-color: #ffffff; border-radius: 4px; cursor: pointer;
                        display: flex; align-items: center; justify-content: center;">
                <i class="fas fa-filter"></i>
            </div>

            <!-- Collapsible filter panel -->
            <div id="fmr-filter-panel"
                 class="leaflet-control"
                 style="display: none; margin-top: 4px;
                        background: rgba(255,255,255,0.96); padding: 8px 10px;
                        border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.35);
                        font-size: 11px; min-width: 230px;">
                <div style="font-weight:bold; margin-bottom: 6px;">FMR Filters</div>

                <div style="margin-bottom:8px;">
                    <div style="font-weight:bold; margin-bottom:4px; font-size:13px;">Images</div>
                    <label style="display:flex; align-items:center; gap:6px; margin-bottom:2px;">
                        <input type="radio" name="image-filter-mode" value="all" checked
                               onchange="setImageFilterMode('all')">
                        <span>All FMRs</span>
                    </label>
                    <label style="display:flex; align-items:center; gap:6px; margin-bottom:2px;">
                        <input type="radio" name="image-filter-mode" value="with"
                               onchange="setImageFilterMode('with')">
                        <span>Only FMRs with images</span>
                    </label>
                    <label style="display:flex; align-items:center; gap:6px;">
                        <input type="radio" name="image-filter-mode" value="without"
                               onchange="setImageFilterMode('without')">
                        <span>Only FMRs without images</span>
                    </label>
                </div>

                <div style="margin-top:8px;">
                    <div style="font-weight:bold; margin-bottom:4px; font-size:13px;">Processing Status</div>
                    <label style="display:flex; align-items:center; gap:6px; margin-bottom:2px;">
                        <input type="checkbox" id="filter-show-unprocessed" checked
                               onchange="updateStatusFilterUnprocessed(this.checked)">
                        <span>Unprocessed</span>
                    </label>
                    <label style="display:flex; align-items:center; gap:6px; margin-bottom:2px;">
                        <input type="checkbox" id="filter-show-automatic" checked
                               onchange="updateStatusFilterAutomatic(this.checked)">
                        <span>Automatic</span>
                    </label>
                    <label style="display:flex; align-items:center; gap:6px;">
                        <input type="checkbox" id="filter-show-manual" checked
                               onchange="updateStatusFilterManual(this.checked)">
                        <span>Manual</span>
                    </label>
                </div>
            </div>
        </div>
       
        <!-- Collapsible main controls button -->
        <div id="toggle-main-controls" 
            style="position: fixed; bottom: 5px; left: 5px; 
                    background: #fff; 
                    border-radius: 6px; 
                    padding: 6px 8px; 
                    box-shadow: 0 2px 6px rgba(0,0,0,0.4); 
                    z-index: 10000; 
                    cursor: pointer;">
            <i class="fas fa-sliders-h"></i>
        </div>
    """

    # --- Province filter JS (fixes "filter_by_province is not defined") ---
    js_ui += """
    <script>
      (function () {
        // restore last selection after reload (optional, but helps UX)
        document.addEventListener('DOMContentLoaded', function () {
          const sel = document.getElementById('provinceSelect');
          const last = localStorage.getItem('selectedProvince');
          if (sel && last) sel.value = last;
        });

        // must be global because HTML calls it directly
        window.filter_by_province = function () {
          const sel = document.getElementById('provinceSelect');
          if (!sel) {
            console.warn('[province filter] #provinceSelect not found');
            return;
          }

          const province = sel.value || 'All';
          localStorage.setItem('selectedProvince', province);

          fetch('/filter_by_province', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ province: province })
          })
          .then(r => r.json())
          .then(data => {
            if (data && data.status === 'filtered') {
              // IMPORTANT: backend regenerates the map HTML, so reload to show it
              window.location.href = '/';
            } else {
              console.error('[province filter] unexpected response:', data);
            }
          })
          .catch(err => console.error('[province filter] failed:', err));
        };
      })();
    </script>
    """

    fmap.get_root().html.add_child(folium.Element(js_ui))

    # 11/23: expose the Leaflet map instance as window._map so filters & overlays can work
    fmap.get_root().html.add_child(folium.Element("""
        <script>
            L.Map.addInitHook(function () {
                window._map = this;
            });
        </script>
    """))

    #11/23: expose per-FMR processing info for front-end filters
    if processing_info_js_lines:
        processing_info_script = "\n".join(processing_info_js_lines)
        fmap.get_root().html.add_child(folium.Element(f"""
            <script>
                window.fmrProcessingInfo = window.fmrProcessingInfo || {{}};
                L.Map.addInitHook(function () {{
                    {processing_info_script}
                }});
            </script>
        """))

    fmap.get_root().html.add_child(folium.Element(f"""
        <script>
            L.Map.addInitHook(function () {{
                setTimeout(function () {{
                    {geo_layer_script}
                    {manual_layer_script}  //11/21: register manual centerline layers in JS
                }}, 0);
            }});
        </script>
    """))

    # 01/12/2026: NEW - pan to selected FMR when Manual Processing dropdown changes
    fmap.get_root().html.add_child(folium.Element(r"""
    <script>
    (function () {
    function getGeoLayers() {
        // geoLayers is used by your injected geo_layer_script; this keeps compatibility either way
        if (typeof geoLayers !== "undefined") return geoLayers;
        if (window.geoLayers) return window.geoLayers;
        return null;
    }

    function normalizeFmrId(val) {
        if (val === null || val === undefined) return null;
        var s = String(val).trim();
        if (!s) return null;

        // supports values like "12", "FMR-12", "FMR_12", etc.
        var m = s.match(/(\d+)/);
        return m ? m[1] : null;
    }

    function panToFmrId(rawVal) {
        var fid = normalizeFmrId(rawVal);
        if (!fid) return;

        var tries = 0;
        function attempt() {
        var map = window._map;
        var layers = getGeoLayers();
        var layer = layers && layers["geoLayer_" + fid];

        if (map && layer && typeof layer.getBounds === "function") {
            var b = layer.getBounds();
            if (b && typeof b.isValid === "function" && b.isValid()) {
            map.fitBounds(b, { padding: [25, 25] });
            } else {
            map.fitBounds(b);
            }
            return;
        }

        tries += 1;
        if (tries <= 25) setTimeout(attempt, 100);
        }
        attempt();
    }

    // optional: expose for debugging
    window.panToManualFMR = panToFmrId;

    // 1) Pan when user changes the dropdown
    document.addEventListener("change", function (e) {
        var el = e.target;
        if (!el) return;

        if (!el.closest || !el.closest("#manual-fmr-container")) return;
        if ((el.tagName || "").toLowerCase() !== "select") return;

        panToFmrId(el.value);
    }, true);

    // 2) Also pan immediately when a new Manual row is added (so user sees it even if they don't change the default)
    function attachObserver() {
        var container = document.getElementById("manual-fmr-container");
        if (!container) return;

        var obs = new MutationObserver(function (mutations) {
        mutations.forEach(function (mu) {
            (mu.addedNodes || []).forEach(function (node) {
            if (!node) return;

            // node might be the row wrapper; find the select inside it
            var sel = null;
            if (node.tagName && String(node.tagName).toLowerCase() === "select") sel = node;
            else if (node.querySelector) sel = node.querySelector("select");

            if (sel && sel.value) panToFmrId(sel.value);
            });
        });
        });

        obs.observe(container, { childList: true, subtree: true });
    }

    // run after page is ready
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", attachObserver);
    } else {
        attachObserver();
    }
    })();
    </script>
    """))

    #11/23: Status legend panel under Database Status (Completed / On-going)
    fmap.get_root().html.add_child(folium.Element("""
        <script>
            L.Map.addInitHook(function () {
                if (document.getElementById('status-legend')) return;

                var lg = document.createElement('div');
                lg.id = 'status-legend';
                lg.style.position = 'fixed';
                lg.style.background = 'rgba(255,255,255,0.96)';
                lg.style.padding = '8px 10px';
                lg.style.borderRadius = '8px';
                lg.style.boxShadow = '0 2px 6px rgba(0,0,0,0.35)';
                lg.style.font = '12px/1.4 sans-serif';
                lg.style.zIndex = 9999;
                lg.style.minWidth = '150px';

                lg.innerHTML =
                    '<div style="font-weight:bold;margin-bottom:6px;">Legend</div>' +

                    '<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">' +
                        '<span style="display:inline-block;width:10px;height:10px;border-radius:50%;' +
                            'background:#2ecc71;margin-right:4px;"></span>' +
                        '<span>Completed</span>' +
                    '</div>' +

                    '<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">' +
                        '<span style="display:inline-block;width:10px;height:10px;border-radius:50%;' +
                            'background:#e74c3c;margin-right:4px;"></span>' +
                        '<span>On-going</span>' +
                    '</div>' +

                    '<hr style="margin:6px 0;border:0;border-top:1px solid #ddd;">' +
                    '<div style="font-weight:bold;margin-bottom:6px;">Processing method</div>' +

                    '<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">' +
                        '<span style="display:inline-block;width:22px;height:0;border-top:3px solid #444;"></span>' +
                        '<span>Automatic (solid line)</span>' +
                    '</div>' +

                    '<div style="display:flex;align-items:center;gap:6px;">' +
                        '<span style="display:inline-block;width:22px;height:0;border-top:3px dashed #444;"></span>' +
                        '<span>Manual (dashed line)</span>' +
                    '</div>';

                document.body.appendChild(lg);

                // Position legend just below the Database Status panel
                var positionLegend = function () {
                    var db = document.getElementById('database-stats');
                    if (db) {
                        var rect = db.getBoundingClientRect();
                        var gapY = 10;
                        lg.style.top = (rect.bottom + gapY) + 'px';
                        lg.style.left = rect.left + 'px';
                    } else {
                        lg.style.top = '120px';
                        lg.style.left = '10px';
                    }
                };

                setTimeout(positionLegend, 0);
                setTimeout(positionLegend, 150);
                window.addEventListener('resize', positionLegend);
            });
        </script>
    """))

    html_path = MAP_HTML_PATH
    fmap.save(html_path)
    print(f"Interactive FMR map created: {html_path}")
    return os.path.abspath(html_path)

@app.route('/check_updates', methods=['GET'])
def check_updates():
    """Check if there are new files to process without actually processing them"""
    global incremental_updater
    
    if not incremental_updater:
        incremental_updater = IncrementalUpdater(shapefile_path, bsg_folder)
    
    status = incremental_updater.check_for_updates()
    
    return jsonify({
        'has_updates': status['has_updates'],
        'new_shapefiles': len(status['new_shapefiles']),
        'new_rasters': len(status['new_rasters']),
        'details': {
            'shapefiles': status['new_shapefiles'][:5],  # Show first 5
            'rasters': status['new_rasters'][:5]
        }
    })

@app.route('/manual_refresh', methods=['POST'])
def manual_refresh():
    """Perform incremental update of database and shapefiles"""
    global incremental_updater, gdf, filtered_gdf
    
    try:
        if not incremental_updater:
            incremental_updater = IncrementalUpdater(shapefile_path, bsg_folder)
        
        # Perform incremental update
        results = incremental_updater.perform_incremental_update()
        
        # Reload GeoDataFrame if shapefiles were updated
        if results.get('shapefiles_updated', False):
            gdf = gpd.read_file(shapefile_path).to_crs(epsg=4326)
            filtered_gdf = gdf.copy()
            create_fmr_map()  # Recreate map only if shapefiles changed
            
        return jsonify({
            'status': 'success',
            'shapefiles_updated': results.get('shapefiles_updated', False),
            'new_database_entries': results.get('new_database_entries', 0),
            'message': f"Added {results.get('new_database_entries', 0)} new entries"
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/full_rebuild', methods=['POST'])
def full_rebuild():
    """Force a complete database rebuild (for troubleshooting)"""
    global gdf, filtered_gdf, incremental_updater
    
    try:
        # Clear the cache to force full reprocessing
        if incremental_updater:
            incremental_updater.clear_cache()
        
        # Run original full update functions
        updateFMRs(shapefile_path)
        getDatabase()
        
        # Reload data
        gdf = gpd.read_file(shapefile_path).to_crs(epsg=4326)
        filtered_gdf = gdf.copy()
        create_fmr_map()
        
        return jsonify({
            'status': 'success',
            'message': 'Full database rebuild completed'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/get_update_stats', methods=['GET'])
def get_update_stats():
    """Get statistics about the database"""
    fmr_db_file = os.path.join(os.path.dirname(shapefile_path), "fmr_database.csv")
    
    stats = {
        'total_fmrs': len(gdf) if 'gdf' in globals() else 0,
        'database_entries': 0,
        'last_update': None
    }
    
    if os.path.exists(fmr_db_file):
        df = pd.read_csv(fmr_db_file)
        stats['database_entries'] = len(df)
        stats['last_update'] = datetime.fromtimestamp(
            os.path.getmtime(fmr_db_file)
        ).strftime('%Y-%m-%d %H:%M:%S')
    
    return jsonify(stats)

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
        """Initialize the user interface"""
        self.setWindowTitle("FMR Monitoring Application")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Add a refresh status bar
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.refresh_button = QPushButton("Check for Updates")
        self.refresh_button.clicked.connect(self.check_for_updates)
        
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        status_layout.addWidget(self.refresh_button)
        layout.addLayout(status_layout)
        
        # Create web view
        self.web_view = QWebEngineView()
        layout.addWidget(self.web_view)
        
        # Load the map
        self.load_map()
        
        # Check for updates after UI is loaded
        QTimer.singleShot(1000, self.check_for_updates)
    
    def check_for_updates(self):
        """Check if there are new files available"""
        try:
            response = requests.get("http://127.0.0.1:5000/check_updates")
            if response.ok:
                data = response.json()
                if data['has_updates']:
                    self.status_label.setText(
                        f"Updates available: {data['new_shapefiles']} shapefiles, "
                        f"{data['new_rasters']} rasters"
                    )
                    self.status_label.setStyleSheet("color: orange;")
                    self.refresh_button.setText("Apply Updates")
                    self.refresh_button.setStyleSheet("background-color: #ff9800;")
                else:
                    self.status_label.setText("No updates available")
                    self.status_label.setStyleSheet("color: green;")
                    self.refresh_button.setText("Check for Updates")
                    self.refresh_button.setStyleSheet("")
        except:
            # Server might not be ready yet
            pass
    
    def start_flask_server(self):
        """Start the Flask server in a separate thread"""
        if self.flask_thread is None:
            self.flask_thread = threading.Thread(target=run_flask, daemon=True)
            self.flask_thread.start()
            print("Flask server started on http://127.0.0.1:5000")
    
    def load_map(self):
        """Load the FMR map in the web view"""
        map_url = QUrl("http://127.0.0.1:5000/")
        self.web_view.load(map_url)
    
    def closeEvent(self, event):
        """Handle application close event"""
        print("Closing FMR GUI application...")
        event.accept()

# 11/11 removed function migrate_database_add_processing_type, since it was not used.
# should add or replace it with a column adder

def main():
    """Optimized main function with lazy loading"""
    global gdf, filtered_gdf, incremental_updater
    
    print("Starting FMR Processing GUI (Optimized)...")
    print("=" * 50)
    
    # Quick load of existing data without processing
    try:
        print("Loading existing FMR shapefile...")
        gdf = gpd.read_file(shapefile_path).to_crs(epsg=4326)
        filtered_gdf = gdf.copy()
        print(f"✓ Loaded {len(gdf)} FMR features")
    except Exception as e:
        print(f"✗ Error loading shapefile: {e}")
        print("Please ensure the master FMR shapefile exists.")
        return
    
    # Check if database exists
    fmr_db_file = os.path.join(os.path.dirname(shapefile_path), "fmr_database.csv")
    if os.path.exists(fmr_db_file):
        try:
            df = pd.read_csv(fmr_db_file)
            print(f"✓ Found existing database with {len(df)} entries")
        except:
            print("✗ Database exists but couldn't be read")
    else:
        print("! No database found - will be created on first refresh")
    
    # Initialize incremental updater
    print("Initializing incremental update system...")
    incremental_updater = IncrementalUpdater(shapefile_path, bsg_folder)
    
    # Quick check for updates without processing
    status = incremental_updater.check_for_updates()
    if status['has_updates']:
        print(f"! Found {len(status['new_shapefiles'])} new shapefiles and {len(status['new_rasters'])} new rasters")
        print("  Use the Refresh button in the GUI to process them")
    else:
        print("✓ No new files detected")
    
    # Create initial map
    print("Creating interactive map...")
    try:
        create_fmr_map()
        print("✓ Map created successfully")
    except Exception as e:
        print(f"✗ Error creating map: {e}")
        return
    
    print("=" * 50)
    print("Starting GUI application...")
    
    # Create and run the GUI application
    app = QApplication(sys.argv)
    app.setApplicationName("FMR Processing GUI (Optimized)")
    
    # Create main window WITHOUT auto-updater
    main_window = FMRMainWindow()  # Pass None for auto_updater
    main_window.show()
    
    print("✓ FMR GUI ready!")
    print("Access the web interface at: http://127.0.0.1:5000")
    print("\nTips:")
    print("• Click 'Refresh Database' to check for new files")
    print("• Processing is now incremental (only new files)")
    print("• Use 'Full Rebuild' if you encounter issues")
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
