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
from shapely.geometry import box
from shapely.ops import transform as shapely_transform
from pyproj import Transformer
from datetime import datetime

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from waitress import serve

# ==========================================================
# Paths
# ==========================================================
shapefile_path = r"C:\Users\user-307E123400\OneDrive - Philippine Space Agency\FMR\.for GUI\Master FMR\master-fmr.shp"
bsg_folder = r"C:\Users\user-307E123400\OneDrive - Philippine Space Agency\SDMAD_SHARED\PROJECTS\SAKA\FMR\GUI\BSG images"

# ==========================================================
# Flask Setup
# ==========================================================
app = Flask(__name__)
CORS(app)
selected_features = []
lock = threading.Lock()
gdf = gpd.read_file(shapefile_path).to_crs(epsg=4326)
filtered_gdf = gdf.copy()


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

    print(f"✅ Done! FMR database saved to:\n{fmr_db_file}")


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
    return send_file("fmr_interactive_map.html")


@app.route('/select', methods=['POST'])
def select_fmr():
    fmr_id = request.json.get("fmr_id")
    with lock:
        if fmr_id is not None and fmr_id not in selected_features:
            selected_features.append(fmr_id)
            return jsonify({"status": "selected", "selected": selected_features})
    return jsonify({"status": "error"}), 400


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
            <button onclick="selectFMR({idx})">Select this FMR</button>
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
    <link rel="stylesheet" href="https://unpkg.com/leaflet-draw/dist/leaflet.draw.css" />
    <script src="https://unpkg.com/leaflet-draw/dist/leaflet.draw.js"></script>
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
        #selection-panel select, button {{ width: 100%; margin-top: 6px; }}
        .clear-btn {{ background-color: #dc3545; color: white; }}
        .clear-btn:hover {{ background-color: #a71d2a; }}
    </style>
    <div id="selection-panel">
        <b>Province Filter:</b>
        <select id="provinceSelect" onchange="filterByProvince()">
            <option value="All">All</option>
            {province_options}
        </select>
        <b>Selected FMRs:</b>
        <ul id="fmr-list"></ul>
        <select id="exportFormat">
            <option value="geojson">GeoJSON</option>
            <option value="shp">Shapefile</option>
            <option value="csv">CSV (attributes only)</option>
        </select>
        <button onclick="downloadSelected()">⬇ Download Selected</button>
        <button class="clear-btn"onclick="clearSelections()">🗑 Clear</button>
        <button onclick="updateFMRs()">🔄 Update FMR</button>
    </div>
    <script>
        const selectedIds = new Set();
        const geoLayers = {{}};

        window.onload = function() {{ {geo_layer_script} }};

        function updateFMRList() {{
            const ul = document.getElementById("fmr-list");
            ul.innerHTML = ""; 
            selectedIds.forEach(id => {{
                const li = document.createElement("li");
                li.textContent = "FMR-" + id;
                ul.appendChild(li);
                const layer = geoLayers["geoLayer_" + id];
                if (layer) layer.setStyle({{color: "red", weight: 3.5}});
            }});
        }}

        function selectFMR(fmr_id) {{
            fetch("http://localhost:5000/select", {{
                method: "POST",
                headers: {{ "Content-Type": "application/json" }},
                body: JSON.stringify({{fmr_id: fmr_id}})
            }}).then(res => res.json())
            .then(data => {{
                if (data.status === "selected") {{
                    selectedIds.add(fmr_id);
                    updateFMRList();
                }} else alert("Already selected.");
            }});
        }}

        function clearSelections() {{
            fetch("http://localhost:5000/clear", {{ method: "POST" }})
            .then(res => res.json())
            .then(data => {{
                if (data.status === "cleared") {{
                    selectedIds.clear();
                    updateFMRList();
                    for (const key in geoLayers) geoLayers[key].setStyle({{color: "yellow", weight: 3.5}});
                }}
            }});
        }}

        function filterByProvince() {{
            const prov = document.getElementById("provinceSelect").value;
            fetch("http://localhost:5000/filter", {{
                method: "POST",
                headers: {{ "Content-Type": "application/json" }},
                body: JSON.stringify({{province: prov}})
            }}).then(res => res.json())
            .then(data => {{
                alert('Filtered: ' + data.count + ' FMRs');
                location.reload();
            }});
        }}

        function updateFMRs() {{
            if (!confirm("Update current FMR list?")) return;
            fetch("http://localhost:5000/update_fmr", {{ method: "POST" }})
            .then(res => res.json())
            .then(data => {{
                if (data.status === "success") {{
                    alert(data.message);
                    location.reload();
                }} else {{
                    alert("Update failed: " + data.message);
                }}
            }});
        }}

        function downloadSelected() {{
            if (selectedIds.size === 0) return alert("No FMRs selected.");
            const format = document.getElementById("exportFormat").value;
            const link = document.createElement("a");
            link.href = `http://localhost:5000/export?format=${{format}}`;
            link.download = `selected_fmrs.${{format}}`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }}
    </script>
    """
    fmap.get_root().html.add_child(folium.Element(js_ui))
    html_path = "fmr_interactive_map.html"
    fmap.save(html_path)
    return os.path.abspath(html_path)


class FMRMapWindow(QMainWindow):
    """Main PyQT Window for FMR Interactive Map."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FMR Interactive Map")
        self.setGeometry(100, 100, 1200, 800)

        layout = QVBoxLayout()
        self.webview = QWebEngineView()
        create_fmr_map()
        self.webview.load(QUrl("http://localhost:5000"))
        layout.addWidget(self.webview)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)


# ==========================================================
# Main Execution
# ==========================================================
if __name__ == "__main__":
    # ✅ Initial FMR database generation
    getDatabase()

    # ✅ Start Flask in a separate thread
    threading.Thread(target=run_flask, daemon=True).start()

    # ✅ Launch PyQT Application
    app_qt = QApplication(sys.argv)
    window = FMRMapWindow()
    window.show()
    sys.exit(app_qt.exec_())
