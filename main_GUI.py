import sys
import os
import geopandas as gpd
import folium
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from waitress import serve
import threading
from datetime import datetime
import tempfile
import shutil
import pandas as pd

# === Master shapefile path ===
shapefile_path = r"C:\Users\user-307E123400\OneDrive - Philippine Space Agency\FMR\.for GUI\Master FMR\master-fmr.shp"

# === Flask setup ===
app = Flask(__name__)
CORS(app)
selected_features = []
lock = threading.Lock()
gdf = gpd.read_file(shapefile_path).to_crs(epsg=4326)
filtered_gdf = gdf.copy()


def updateFMRs(master_path):
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
                    gdf_temp = gpd.read_file(file_path, encoding='latin1')
                    if gdf_temp.crs != master_crs:
                        gdf_temp = gdf_temp.to_crs(master_crs)
                    gdfs_to_merge.append(gdf_temp)
                except Exception as e:
                    print(f"Could not read {file_path}: {e}")

    if not gdfs_to_merge:
        print("No additional FMR shapefiles found to merge.")
        return False  # Nothing merged

    merged_gdf = pd.concat([master_gdf] + gdfs_to_merge, ignore_index=True)
    merged_gdf.to_file(master_path)

    merged_folder = os.path.join(master_dir, "merged")
    os.makedirs(merged_folder, exist_ok=True)

    for file in os.listdir(master_dir):
        base_name, ext = os.path.splitext(file)
        if base_name != master_name and ext.lower() in ['.shp', '.shx', '.dbf', '.prj', '.cpg', '.qix']:
            full_file = os.path.join(master_dir, file)
            if os.path.isfile(full_file):
                shutil.move(full_file, os.path.join(merged_folder, file))

    return True


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
        create_fmr_map(filtered_gdf, pan_to=filtered_gdf)
        return jsonify({"status": "filtered", "count": len(filtered_gdf)})


@app.route('/merge', methods=['POST'])
def merge_fmrs():
    try:
        result = updateFMRs(shapefile_path)
        if not result:
            return jsonify({"status": "info", "message": "No additional FMRs found to merge."})
        global gdf, filtered_gdf
        gdf = gpd.read_file(shapefile_path).to_crs(epsg=4326)
        filtered_gdf = gdf.copy()
        create_fmr_map(gdf)
        return jsonify({"status": "success", "message": "FMR list updated successfully"})
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
    serve(app, host="127.0.0.1", port=5000)


def create_fmr_map(input_gdf=None, pan_to=None):
    map_gdf = input_gdf if input_gdf is not None else gdf
    if map_gdf.empty:
        print("Shapefile is empty!")
        return ""

    center = map_gdf.union_all().centroid
    fmap = folium.Map(location=[center.y, center.x], zoom_start=10, tiles="Esri.WorldImagery")

    provinces = sorted(set(p.title() for p in gdf['PROV_NAME'].dropna()))
    province_options = "".join([f"<option value='{p}'>{p}</option>" for p in provinces])

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

    js_ui = f"""
    <div id="selection-panel" style="
        position: fixed; bottom: 20px; left: 20px; background: rgba(255,255,255,0.95);
        padding: 10px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.3); z-index: 9999; max-width: 300px;">
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
        <button onclick="clearSelections()" style="background-color:#dc3545;color:white;">🗑 Clear</button>
        <button onclick="mergeFMRs()">🔄 Update FMR</button>
    </div>
    <script>
        const selectedIds = new Set();
        function updateFMRList() {{
            const ul = document.getElementById("fmr-list");
            ul.innerHTML = "";
            selectedIds.forEach(id => {{
                const li = document.createElement("li");
                li.textContent = "FMR-" + id;
                ul.appendChild(li);
            }});
        }}

        function selectFMR(fmr_id) {{
            fetch("http://localhost:5000/select", {{
                method: "POST",
                headers: {{"Content-Type": "application/json"}},
                body: JSON.stringify({{fmr_id}})
            }}).then(res => res.json()).then(data => {{
                if (data.status === "selected") {{
                    selectedIds.add(fmr_id);
                    updateFMRList();
                }} else {{
                    alert("Already selected.");
                }}
            }});
        }}

        function clearSelections() {{
            fetch("http://localhost:5000/clear", {{ method: "POST" }})
            .then(res => res.json()).then(data => {{
                if (data.status === "cleared") {{
                    selectedIds.clear();
                    updateFMRList();
                }}
            }});
        }}

        function filterByProvince() {{
            const prov = document.getElementById("provinceSelect").value;
            fetch("http://localhost:5000/filter", {{
                method: "POST",
                headers: {{"Content-Type": "application/json"}},
                body: JSON.stringify({{province: prov}})
            }}).then(res => res.json()).then(data => {{
                alert('Filtered: ' + data.count + ' FMRs');
                location.reload();
            }});
        }}

        function mergeFMRs() {{
            if (!confirm("Update current FMR list?")) return;
            fetch("http://localhost:5000/merge", {{ method: "POST" }})
            .then(res => res.json()).then(data => {{
                if (data.status === "success" || data.status === "info") {{
                    alert(data.message);
                    location.reload();
                }} else {{
                    alert("Merge failed: " + data.message);
                }}
            }});
        }}

        function downloadSelected() {{
            if (selectedIds.size === 0) {{
                alert("No FMRs selected.");
                return;
            }}
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


if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    app_qt = QApplication(sys.argv)
    window = FMRMapWindow()
    window.show()
    sys.exit(app_qt.exec_())
