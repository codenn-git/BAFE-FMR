// JavaScript logic for FMR GUI
// Optimized: Collapsible FMRs, scrollable panel, expand/collapse all, and lazy overlay
// Optional: Add buttons to expand/collapse all in #selection-panel manually
// <button onclick="toggleAllFMRs(true)">Expand All</button>
// <button onclick="toggleAllFMRs(false)">Collapse All</button>

const selectedIds = new Set();
const geoLayers = {};
let currentMatchingImages = {};
const style = document.createElement('style');
const imageCache = {};  // key: image path, value: { base64, bounds }
const overlayLayers = {};  // key: image path, value: leaflet layer

function overlayImage({ image_base64, image_bounds }, imagePath) {
    if (!window._map) return;
    const layerKey = `overlay_${btoa(imagePath)}`;
    if (overlayLayers[layerKey]) return;

    const img = L.imageOverlay(`data:image/png;base64,${image_base64}`, image_bounds).addTo(window._map);
    overlayLayers[layerKey] = img;
}

function removeOverlay(imagePath) {
    if (!window._map || !imagePath) return;
    const layerKey = `overlay_${btoa(imagePath)}`;
    const layer = overlayLayers[layerKey];
    if (layer) {
        window._map.removeLayer(layer);
        delete overlayLayers[layerKey];
    }
}

document.addEventListener('change', function (e) {
    if (e.target && e.target.classList.contains('image-checkbox')) {
        const checkbox = e.target;
        const imagePath = checkbox.dataset.imagePath;
        const fmrId = checkbox.dataset.fmrId;

        if (checkbox.checked) {
            if (imageCache[imagePath]) {
                overlayImage(imageCache[imagePath], imagePath);
            } else {
                fetch('/display_selected_image', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image_path: imagePath, fmr_id: parseInt(fmrId) })
                })
                .then(res => res.json())
                .then(data => {
                    if (data.status === "success") {
                        imageCache[imagePath] = {
                            image_base64: data.image_base64,
                            image_bounds: data.image_bounds
                        };
                        overlayImage(imageCache[imagePath], imagePath);
                    }
                });
            }
        } else {
            removeOverlay(imagePath);
        }
    }
});

function selectedFMRList() {
    return Array.from(selectedIds);
}

style.textContent = `
  #fmr-list li input[type="checkbox"] {
    margin-right: 6px;
  }
  #fmr-list li label {
    font-size: 0.95em;
  }
  #selection-panel {
    max-height: 90vh;
    overflow-y: auto;
  }
  .image-list {
    display: none;
    padding-top: 8px;
  }
  .image-list.show {
    display: block;
  }
  .fmr-header {
    display: flex;
    justify-content: space-between;
    cursor: pointer;
    font-weight: bold;
    font-size: 1rem;
  }
  .toggle-icon {
    font-size: 0.9em;
    color: #444;
  }
`;
document.head.appendChild(style);

function updateFMRList() {
    const ul = document.getElementById("fmr-list");
    ul.innerHTML = "";

    selectedIds.forEach(id => {
        const li = document.createElement("li");
        li.classList.add("fmr-item");

        const header = document.createElement("div");
        header.classList.add("fmr-header");
        header.innerHTML = `<span><b>FMR-${id}</b></span><span class="toggle-icon">▶</span>`;
        li.appendChild(header);

        const images = currentMatchingImages[id] || [];
        const imageList = document.createElement("ul");
        imageList.classList.add("image-list");

        header.addEventListener("click", () => {
            imageList.classList.toggle("show");
            const icon = header.querySelector(".toggle-icon");
            icon.textContent = imageList.classList.contains("show") ? "▼" : "▶";
        });

        if (images.length > 0) {
            images.forEach((img, idx) => {
                const item = document.createElement("li");
                item.classList.add("image-option");

                const checkbox = document.createElement("input");
                checkbox.type = "checkbox";
                checkbox.classList.add("image-checkbox");
                checkbox.dataset.imagePath = img.path;
                checkbox.dataset.fmrId = id;
                checkbox.id = `img-${id}-${idx}`;
                checkbox.disabled = true;

                const label = document.createElement("label");
                label.htmlFor = checkbox.id;
                label.textContent = " " + img.path.split(/[\\/]/).pop().split("-").slice(0, 4).join("-");

                item.appendChild(checkbox);
                item.appendChild(label);
                imageList.appendChild(item);

                setTimeout(() => {
                    checkbox.disabled = false;
                }, 300);
            });
        } else {
            const note = document.createElement("div");
            note.style.fontSize = "0.85em";
            note.style.color = "#888";
            note.textContent = "(No matching images)";
            imageList.appendChild(note);
        }

        li.appendChild(imageList);
        ul.appendChild(li);

        const layer = geoLayers["geoLayer_" + id];
        if (layer) layer.setStyle({color: "red", weight: 3.5});
    });
}

function toggleAllFMRs(expand) {
    const lists = document.querySelectorAll(".image-list");
    const icons = document.querySelectorAll(".toggle-icon");
    lists.forEach((list, idx) => {
        if (expand) list.classList.add("show");
        else list.classList.remove("show");

        if (icons[idx]) {
            icons[idx].textContent = expand ? "▼" : "▶";
        }
    });
}

function selectFMR(fmr_id) {
    fetch("http://localhost:5000/select", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fmr_id: fmr_id })
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === "selected") {
            selectedIds.add(fmr_id);
            return fetch("http://localhost:5000/get_matching_images", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ fmr_id: fmr_id })
            });
        } else {
            alert("Already selected.");
            throw new Error("Already selected");
        }
    })
    .then(res => res.json())
    .then(imageData => {
        currentMatchingImages[fmr_id] = imageData.status === "success" ? imageData.images || [] : [];
        updateFMRList();
    })
    .catch(err => console.error("Error in selectFMR:", err));
}

function deselectFMR(fmr_id) {
    fetch("http://localhost:5000/deselect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({fmr_id: fmr_id})
    }).then(res => res.json())
    .then(data => {
        if (data.status === "deselected") {
            selectedIds.delete(fmr_id);
            delete currentMatchingImages[fmr_id];
            updateFMRList();
            const layer = geoLayers["geoLayer_" + fmr_id];
            if (layer) layer.setStyle({color: "yellow", weight: 3.5});
        } else {
            alert("FMR not selected or error occurred.");
        }
    });
}

function clearSelections() {
    fetch("http://localhost:5000/clear", {method: "POST"})
    .then(res => res.json())
    .then(data => {
        if (data.status === "cleared") {
            selectedIds.clear();
            Object.keys(currentMatchingImages).forEach(key => delete currentMatchingImages[key]);
            Object.values(overlayLayers).forEach(layer => window._map.removeLayer(layer));
            Object.keys(overlayLayers).forEach(key => delete overlayLayers[key]);
            updateFMRList();
            Object.values(geoLayers).forEach(layer => layer.setStyle({color: "yellow", weight: 3.5}));
            document.querySelectorAll('[id^="processing-buttons-"]').forEach(el => el.style.display = 'none');
            document.querySelectorAll('[id^="image-display-"]').forEach(el => el.style.display = 'none');
            document.querySelectorAll('[id^="results-"]').forEach(el => el.style.display = 'none');
        }
    });
}

