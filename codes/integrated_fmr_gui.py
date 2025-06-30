import sys
import os
import re
import shutil
import threading
import tempfile
import pandas as pd
import geopandas as gpd
import folium
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import cv2
from shapely.geometry import box
from shapely.ops import transform as shapely_transform
from pyproj import Transformer
from datetime import datetime
import base64
from io import BytesIO

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from waitress import serve

# Import your processing utilities
# NOTE: You'll need to make sure these imports work with your util.py
from utilv1 import Preprocessing, Filters, Morph, Interaction, MeasureWidth, measure_line, export

# ==========================================================
# Paths
# ==========================================================
shapefile_path = r"C:\Users\user-307E123400\OneDrive - Philippine Space Agency\SDMAD_SHARED\PROJECTS\SAKA\FMR\GUI\Master FMR\NE_master_fmr.shp"
bsg_folder = r"C:\Users\user-307E123400\OneDrive - Philippine Space Agency\SDMAD_SHARED\PROJECTS\SAKA\FMR\GUI\Raster images"

# ==========================================================
# Flask Setup
# ==========================================================
app = Flask(__name__)
CORS(app)
selected_features = []
current_fmr_data = None
current_image_data = None
lock = threading.Lock()
gdf = gpd.read_file(shapefile_path).to_crs(epsg=4326)
filtered_gdf = gdf.copy()

# ==========================================================
# Processing Functions (adapted from your main-v1.1.py)
# ==========================================================

def stretch_band(band, lower_percent=2, upper_percent=98):
    """Stretch the bands of the image for display purposes"""
    lower = np.percentile(band, lower_percent)
    upper = np.percentile(band, upper_percent)
    stretched = np.clip((band - lower) / (upper - lower), 0, 1)
    return stretched

def find_matching_images(fmr_geometry, bsg_folder_path):
    """Find BSG images that intersect with the FMR geometry"""
    tif_files = [f for f in os.listdir(bsg_folder_path) if f.endswith("-Tiff.tif")]
    matching_images = []
    
    # Transformer for reprojection
    pre = Preprocessing()
    
    for tif_file in tif_files:
        tif_path = os.path.join(bsg_folder_path, tif_file)
        try:
            _, _, raster_bounds, _ = pre.reproject(tif_path, fmr_geometry)
            
            if isinstance(raster_bounds, (list, tuple)) and len(raster_bounds) == 4:
                bounds_geom = box(*raster_bounds)
                if fmr_geometry.intersects(bounds_geom):
                    matching_images.append({
                        'filename': tif_file,
                        'path': tif_path,
                        'bounds': bounds_geom
                    })
        except Exception as e:
            print(f"Error processing {tif_file}: {e}")
            continue
    
    return matching_images

def create_image_preview(image_path, fmr_geometry):
    """Create a preview of the clipped and reprojected image"""
    try:
        # This would use your Preprocessing class
        preprocessor = Preprocessing()
        preprocessor.reproject(image_path, fmr_geometry)
        clipped_data, clipped_transform = preprocessor.clipraster(buffer_dist=25, bbox=True)
        
        preprocessor.display()
                
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return img_base64
        
    except Exception as e:
        print(f"Error creating image preview: {e}")
        return None

def process_tracking(image_path, fmr_geometry, mode='automatic'):
    """Process FMR tracking (adapted from your tracking workflow)"""
    try:
        # This would use your processing pipeline
        preprocessor = Preprocessing()
        preprocessor.reproject(image_path, fmr_geometry)
        clipped_data, clipped_transform = preprocessor.clipraster(buffer_dist=25, bbox=True)
        
        results = {
            'status': 'success',
            'mode': mode,
            'message': f'Tracking completed in {mode} mode'
        }
        
        if mode == 'automatic':
            # Apply your automatic processing pipeline
            filter = Filters()
            warm_raster = filter.enhance_image_warmth(clipped_data)
            stretch_raster = filter.enhance_linear_stretch(clipped_data)

            morph = Morph()
            morph_warm = morph.process(warm_raster)
            morph_stretch = morph.process(stretch_raster)                                                                  

            #merges the applied morphed warmth and stretch function
            final_binary_raster = np.logical_or(morph_warm, morph_stretch)

            final_binary_raster = cv2.morphologyEx(final_binary_raster.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=3)

            final_clipped_data, final_clipped_transform = preprocessor.clipraster(
                                    raster_data = final_binary_raster.astype(np.uint8),
                                    transform = clipped_transform,
                                    buffer_dist = 1)
            
            final_line = measure_line(final_clipped_data, final_clipped_transform, spacing=3)
            # ... rest of your automatic processing
            
            final_line_length = final_line.length.values[0]
            vector_length = fmr_geometry.length.sum()

            results['length'] = final_line.length.values[0]
            results['progress'] = (final_line_length / vector_length) * 100 #'Progress percentage would go here'
            
        elif mode == 'manual':
            # Set up for manual interaction
            interaction = Interaction(clipped_data, fmr_geometry)
            results['message'] = 'Manual tracking mode - interaction interface would be displayed'
        
        return results
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Error in tracking: {str(e)}'
        }

def process_extraction(image_path, fmr_geometry, image_type='BSG'):
    """Process road width extraction (adapted from your extraction workflow)"""
    try:
        # This would use your processing pipeline
        preprocessor = Preprocessing()
        preprocessor.reproject(image_path, fmr_geometry)
        
        results = {
            'status': 'success',
            'image_type': image_type,
            'message': f'Width extraction completed for {image_type} image'
        }
        
        if image_type == 'PNEO':
            # Apply PNEO-specific processing
            # int, tol, res = 3, 0.15, 0.3
            # clipped_data, clipped_transform = preprocessor.clipraster(bbox=True)
            # filter = Filters(clipped_data)
            # cielab = filter.cielab()
            # ... rest of PNEO processing
            
            results['mean_width'] = 'PNEO width calculation would go here'
            
        elif image_type == 'BSG':
            # Apply BSG-specific processing
            int, tol, res = 3, 0.4, 0.3
            raster_data, _ = preprocessor.clipraster(bbox=True)
            clipped_data, clipped_transform = preprocessor.clipraster(buffer_dist=25)

            #Filters: warmth and linear stretch
            filter = Filters() #NEW: filter = Filters()
            warm_raster = filter.enhance_image_warmth(clipped_data) #NEW: warm_raster  = filter.enhance_image_warmth(clipped_data)
            stretch_raster = filter.enhance_linear_stretch(clipped_data)  #NEW: warm_raster  = filter.enhance_linear_stretch(clipped_data)

            #Apply Morphological Operations
            morph = Morph()
            morph_warm = morph.process(warm_raster)
            morph_stretch = morph.process(stretch_raster)

            #merges the applied morphed warmth and stretch function
            merged_or = np.logical_or(morph_warm, morph_stretch)
            initial_binary_raster = merged_or

            final_binary_transform = clipped_transform
 
            # plt.imshow(final_clipped_data, cmap="gray")
            final_binary_raster = morph.remove_small_islands(initial_binary_raster, min_size=1000)
            final_binary_raster = cv2.morphologyEx(final_binary_raster.astype(np.uint8), 
                                                   cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=3)

            measure = MeasureWidth(final_binary_raster, final_binary_transform, fmr_geometry)
            measure.process(int=int, tol=tol, res=res)

            results['mean_width'] = measure.clipped_transects['width'].mean()
        
        return results
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Error in extraction: {str(e)}'
        }

# ==========================================================
# Original Flask Routes
# ==========================================================

def getDatabase():
    """Scan FMR and BSG images, extracting match information and save results to 'fmr_database.csv'."""
    master_fmr = shapefile_path
    bsg_folder_path = bsg_folder
    fmr_db_file = os.path.join(os.path.dirname(master_fmr), "fmr_database.csv")

    # Read and reproject FMR
    fmr_gdf = gpd.read_file(master_fmr).to_crs("EPSG:32651")  

    # List .tif files
    tif_files = [f for f in os.listdir(bsg_folder_path) if f.lower().endswith(".tif")]

    # Transformer
    fmr_transformer = Transformer.from_crs("EPSG:32651", "EPSG:4326", always_xy=True)

    results = []
    for idx, row in fmr_gdf.iterrows():
        fmr_name = str(row.get("name", f"FMR_{idx}"))
        fmr_geom = row.geometry
        planned_length = fmr_geom.length
        matched_images = []
        for tif_file in tif_files:
            tif_path = os.path.join(bsg_folder_path, tif_file)

            with rasterio.open(tif_path) as src:
                raster_bounds = box(*src.bounds)

            # Reproject FMR geometry to raster CRS
            fmr_geom_in_raster_crs = shapely_transform(fmr_transformer.transform, fmr_geom)

            if fmr_geom_in_raster_crs.intersects(raster_bounds):
                matched_images.append(tif_file)

        if matched_images:
            for tif_file in matched_images:
                match = re.search(r"(\d{8})-(\d{6})", tif_file)
                if match:
                    raw_date, raw_time = match.groups()
                    try:
                        dt = datetime.strptime(raw_date + raw_time, "%Y%m%d%H%M%S")
                        formatted_date = dt.strftime("%Y-%m-%d")
                        formatted_time = dt.strftime("%H:%M:%S")
                    except ValueError:
                        formatted_date = None
                        formatted_time = None
                else:
                    formatted_date = None
                    formatted_time = None

                results.append({
                    "FMR": fmr_name,
                    "BSG": tif_file,
                    "Date": formatted_date,
                    "Time": formatted_time,
                    "Planned FMR Length": planned_length,
                    "Current FMR Length": "",
                    "FMR Progress": ""
                })
        else:
            results.append({
                "FMR": fmr_name,
                "BSG": None,
                "Date": None,
                "Time": None,
                "Planned FMR Length": planned_length,
                "Current FMR Length": "",
                "FMR Progress": ""
            })

    results_df = pd.DataFrame(results)

    if os.path.exists(fmr_db_file):
        existing_df = pd.read_csv(fmr_db_file)
        final_df = pd.concat([existing_df, results_df], ignore_index=True).drop_duplicates(subset=["FMR", "BSG"])
    else:
        final_df = results_df

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


@app.route('/')
def serve_map():
    return send_file(r"fmr_interactive_map.html")  # Path changed aina


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
    
    if fmr_id in selected_fmrs:
        selected_fmrs.remove(fmr_id)
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
        gdf = gpd.read_file(shapefile_path).to_crs(epsg=4326)
        filtered_gdf = gdf.copy()
        create_fmr_map(gdf)
        return jsonify({"status": "success", "message": "FMR updated successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/export', methods=['GET'])
def export_selected():
    fmt = request.args.get("format", "geojson")
    with lock:
        if not selected_features:
            return jsonify({"status": "error", "message": "No FMRs selected"}), 400
        selected_gdf = gdf.loc[selected_features]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tempdir = tempfile.gettempdir()
        if fmt == "shp":
            export_path = os.path.join(tempdir, f"selected_fmrs_{timestamp}.shp")
            selected_gdf.to_file(export_path)
        elif fmt == "csv":
            export_path = os.path.join(tempdir, f"selected_fmrs_{timestamp}.csv")
            selected_gdf.drop(columns="geometry").to_csv(export_path, index=False)
        else:
            export_path = os.path.join(tempdir, f"selected_fmrs_{timestamp}.geojson")
            selected_gdf.to_file(export_path, driver="GeoJSON")
        return send_file(export_path, as_attachment=True)

# ==========================================================
# New Processing Routes
# ==========================================================

@app.route('/get_matching_images', methods=['POST'])
def get_matching_images():
    """Get list of images that match the selected FMR"""
    fmr_id = request.json.get("fmr_id")
    if fmr_id is None:
        return jsonify({"status": "error", "message": "No FMR ID provided"}), 400
    
    try:
        fmr_geometry = gdf.loc[fmr_id].geometry
        matching_images = find_matching_images(fmr_geometry, bsg_folder)
        
        return jsonify({
            "status": "success",
            "fmr_id": fmr_id,
            "matching_images": matching_images
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/display_image', methods=['POST'])
def display_image():
    """Display the selected image clipped to FMR extent"""
    data = request.json
    fmr_id = data.get("fmr_id")
    image_path = data.get("image_path")
    
    if not fmr_id or not image_path:
        return jsonify({"status": "error", "message": "Missing FMR ID or image path"}), 400
    
    try:
        fmr_geometry = gdf.loc[fmr_id].geometry
        image_preview = create_image_preview(image_path, fmr_geometry)
        
        if image_preview:
            return jsonify({
                "status": "success",
                "image_data": image_preview,
                "fmr_id": fmr_id,
                "image_path": image_path
            })
        else:
            return jsonify({"status": "error", "message": "Failed to create image preview"}), 500
            
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/process_fmr', methods=['POST'])
def process_fmr():
    """Process the selected FMR with the chosen workflow"""
    data = request.json
    fmr_id = data.get("fmr_id")
    image_path = data.get("image_path")
    workflow_type = data.get("workflow_type")  # 'track' or 'extract'
    workflow_options = data.get("workflow_options", {})
    
    if not all([fmr_id, image_path, workflow_type]):
        return jsonify({"status": "error", "message": "Missing required parameters"}), 400
    
    try:
        fmr_geometry = gdf.loc[fmr_id].geometry
        
        if workflow_type == 'track':
            mode = workflow_options.get('mode', 'automatic')  # 'automatic' or 'manual'
            results = process_tracking(image_path, fmr_geometry, mode)
            
        elif workflow_type == 'extract':
            image_type = workflow_options.get('image_type', 'BSG')  # 'BSG' or 'PNEO'
            results = process_extraction(image_path, fmr_geometry, image_type)
            
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

    center = map_gdf.union_all().centroid
    fmap = folium.Map(location=[center.y, center.x], zoom_start=10, tiles="Esri.WorldImagery")

    geo_layer_var_lines = []
    for idx, row in map_gdf.iterrows():
        layer_name = f"geoLayer_{idx}"
        brgy = row.get("BRGY_NAME", "N/A")
        mun = row.get("MUN_NAME", "N/A")
        prov = row.get("PROV_NAME", "N/A")

        popup_html = f"""
        <div style='word-wrap: break-word; max-width: 350px;'>
            <b>FMR ID:</b> {idx}<br>
            <b>Barangay:</b> {brgy}<br>
            <b>Municipality:</b> {mun}<br>
            <b>Province:</b> {prov}<br><br>
            <button onclick=\"selectFMR({idx})\">Select FMR</button><br><br>
            <button onclick="deselectFMR({idx})">Deselect FMR</button>
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
        <button id=\"processFMRBtn\" disabled>Process FMR</button>
        <select id=\"exportFormat\">
            <option value=\"geojson\">GeoJSON</option>
            <option value=\"shp\">Shapefile</option>
            <option value=\"csv\">CSV (attributes only)</option>
        </select>
        <button onclick=\"downloadSelected()\">⬇ Download Selected</button>
        <button class=\"clear-btn\" onclick=\"clearSelections()\">🗑 Clear</button>
        <button onclick=\"updateFMRs()\">🔄 Update FMR</button>

        <div id="dynamic-processing-panel" style="margin-top: 20px;"></div>
    </div>
    """

    # Add UI and JavaScript hook to HTML
    fmap.get_root().header.add_child(folium.Element(f"<script>window.onload = function() {{ {geo_layer_script} }}</script>"))
    fmap.get_root().html.add_child(folium.Element(js_ui))

    html_path = r"fmr_interactive_map.html" # Path changed aina 
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