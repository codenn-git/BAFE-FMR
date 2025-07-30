// JavaScript logic for FMR GUI
// Fixed: Multiple BSG images can now be displayed simultaneously
// Optimized: Collapsible FMRs, scrollable panel, expand/collapse all, and lazy loading

const selectedIds = new Set();
const geoLayers = {};
let currentMatchingImages = {};
const style = document.createElement('style');
const imageCache = {};  // key: image path, value: { base64, bounds }
const overlayLayers = {};  // key: image path, value: leaflet layer

function overlayImage({ image_base64, image_bounds }, imagePath) {
    if (!window._map) {
        console.error('Map not available');
        return;
    }
    
    const layerKey = `overlay_${btoa(imagePath).replace(/[^a-zA-Z0-9]/g, '')}`;
    
    // Remove existing overlay if it exists
    if (overlayLayers[layerKey]) {
        window._map.removeLayer(overlayLayers[layerKey]);
        delete overlayLayers[layerKey];
    }

    try {
        // Create image overlay with proper bounds
        const img = L.imageOverlay(
            `data:image/png;base64,${image_base64}`, 
            image_bounds,
            {
                opacity: 1.0,
                interactive: false,
                zIndex: 100
            }
        );
        
        // Add to map and store reference
        img.addTo(window._map);
        overlayLayers[layerKey] = img;
        
        console.log(`Added overlay for: ${imagePath}`);
        
    } catch (error) {
        console.error('Error creating image overlay:', error);
    }
}

function removeOverlay(imagePath) {
    if (!window._map || !imagePath) return;
    
    const layerKey = `overlay_${btoa(imagePath).replace(/[^a-zA-Z0-9]/g, '')}`;
    const layer = overlayLayers[layerKey];
    
    if (layer) {
        try {
            window._map.removeLayer(layer);
            delete overlayLayers[layerKey];
            console.log(`Removed overlay for: ${imagePath}`);
        } catch (error) {
            console.error('Error removing overlay:', error);
        }
    }
}

// Enhanced event listener for image checkboxes
document.addEventListener('change', function (e) {
    if (e.target && e.target.classList.contains('image-checkbox')) {
        const checkbox = e.target;
        const imagePath = checkbox.dataset.imagePath;
        const fmrId = parseInt(checkbox.dataset.fmrId);

        if (checkbox.checked) {
            // Show loading indicator
            const label = checkbox.nextElementSibling;
            const originalText = label.textContent;
            label.textContent = originalText + ' (Loading...)';
            
            if (imageCache[imagePath]) {
                // Use cached image
                overlayImage(imageCache[imagePath], imagePath);
                label.textContent = originalText;
            } else {
                // Fetch and display image
                fetch('/display_selected_image', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        image_path: imagePath, 
                        fmr_id: fmrId 
                    })
                })
                .then(res => {
                    if (!res.ok) {
                        throw new Error(`HTTP error! status: ${res.status}`);
                    }
                    return res.json();
                })
                .then(data => {
                    if (data.status === "success") {
                        // Cache the image data
                        imageCache[imagePath] = {
                            image_base64: data.image_data,
                            image_bounds: data.bounds
                        };
                        // Display the image
                        overlayImage(imageCache[imagePath], imagePath);
                        label.textContent = originalText;
                    } else {
                        throw new Error(data.message || 'Failed to load image');
                    }
                })
                .catch(error => {
                    console.error('Error loading image:', error);
                    label.textContent = originalText + ' (Error)';
                    checkbox.checked = false;
                    alert(`Failed to load image: ${error.message}`);
                });
            }
        } else {
            // Remove overlay when unchecked
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
    font-size: 1.4rem;
  }
  .toggle-icon {
    font-size: 0.9em;
    color: #444;
  }
  .image-option {
    margin: 2px 0;
    padding: 2px 0;
    font-size: 1.2rem;
  }
  .image-checkbox:disabled + label {
    color: #999;
    cursor: not-allowed;
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

                // Enable checkbox after short delay
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

        // Update FMR layer style
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
            updateRunButtonState();  // for updating the run button (disabling/enabling)
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
            updateRunButtonState();
            
            // Remove all overlays for this FMR
            const images = currentMatchingImages[fmr_id] || [];
            images.forEach(img => removeOverlay(img.path));
            
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
            updateRunButtonState();
            
            // Remove all overlays
            Object.keys(overlayLayers).forEach(key => {
                if (overlayLayers[key]) {
                    window._map.removeLayer(overlayLayers[key]);
                    delete overlayLayers[key];
                }
            });
            
            // Clear data structures
            Object.keys(currentMatchingImages).forEach(key => delete currentMatchingImages[key]);
            Object.keys(imageCache).forEach(key => delete imageCache[key]);
            
            updateFMRList();
            
            // Reset all FMR layer styles
            Object.values(geoLayers).forEach(layer => layer.setStyle({color: "yellow", weight: 3.5}));
            
            // Hide processing panels
            document.querySelectorAll('[id^="processing-buttons-"]').forEach(el => el.style.display = 'none');
            document.querySelectorAll('[id^="image-display-"]').forEach(el => el.style.display = 'none');
            document.querySelectorAll('[id^="results-"]').forEach(el => el.style.display = 'none');
        }
    });
}

// Additional utility functions for better overlay management
function removeAllOverlays() {
    Object.keys(overlayLayers).forEach(key => {
        if (overlayLayers[key] && window._map) {
            window._map.removeLayer(overlayLayers[key]);
            delete overlayLayers[key];
        }
    });
}

function getActiveOverlays() {
    return Object.keys(overlayLayers).map(key => ({
        key: key,
        layer: overlayLayers[key]
    }));
}

// Function to check if image is already displayed
function isImageDisplayed(imagePath) {
    const layerKey = `overlay_${btoa(imagePath).replace(/[^a-zA-Z0-9]/g, '')}`;
    return !!overlayLayers[layerKey];
}

// processing modal functions //
function showProcessingModal() {
    const modal = document.getElementById('processing-modal');
    if (modal) modal.style.display = 'flex';
}

function hideProcessingModal() {
    const modal = document.getElementById('processing-modal');
    if (modal) modal.style.display = 'none';
}

function runProcessing() {
    const processType = document.querySelector('input[name="process-type"]:checked').value;
    const workflowType = document.querySelector('input[name="workflow-type"]:checked').value;
    const imageType = document.getElementById('image-type').value;
    
    // Get selected FMR IDs
    const fmrIds = Array.from(selectedIds);
    
    // Get selected images if "selected" is chosen
    const imagesToProcess = [];
    
    if (processType === 'selected') {
        // Find all checked image checkboxes
        document.querySelectorAll('.image-checkbox:checked').forEach(checkbox => {
            imagesToProcess.push({
                fmr_id: parseInt(checkbox.dataset.fmrId),
                image_path: checkbox.dataset.imagePath
            });
        });
        
        if (imagesToProcess.length === 0) {
            alert('Please select at least one image to process');
            return;
        }
    } else {
        // Process all images for selected FMRs
        fmrIds.forEach(fmrId => {
            const images = currentMatchingImages[fmrId] || [];
            images.forEach(img => {
                imagesToProcess.push({
                    fmr_id: fmrId,
                    image_path: img.path
                });
            });
        });
    }
    
    if (imagesToProcess.length === 0) {
        alert('No images found to process');
        return;
    }
    
    // Process each image
    imagesToProcess.forEach(item => {
        processFMR(item.fmr_id, item.image_path, workflowType, imageType);
    });
    
    hideProcessingModal();
}

function processFMR(fmr_id, image_path, workflow_type, image_type) {
    fetch('/process_fmr', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
            fmr_id: fmr_id,
            image_path: image_path,
            workflow_type: workflow_type,
            image_type: image_type
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'error') {
            console.error('Processing error:', data.message);
            alert(`Error processing FMR-${fmr_id}: ${data.message}`);
        } else {
            console.log('Processing results:', data);
            // You can display results here or update UI as needed
            alert(`Processing completed for FMR-${fmr_id}`);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert(`Error processing FMR-${fmr_id}: ${error.message}`);
    });
}

// Update the run button state based on selections
function updateRunButtonState() {
    const runBtn = document.getElementById('runBtn');
    runBtn.disabled = selectedIds.size === 0;
}

// Export functions for external use
window.FMRUtils = {
    selectedFMRList,
    removeAllOverlays,
    getActiveOverlays,
    isImageDisplayed
};
let showingOnlyWithImages = false;

function toggleImageVisibility(button) {
    showingOnlyWithImages = !showingOnlyWithImages;
    button.classList.toggle('active');

    if (showingOnlyWithImages) {
        fetch('/get_fmrs_with_images')
            .then(res => res.json())
            .then(data => {
                if (data.status === "success") {
                    const visibleIds = new Set(data.fmr_ids.map(id => parseInt(id)));

                    Object.entries(geoLayers).forEach(([key, layer]) => {
                        const id = parseInt(key.replace("geoLayer_", ""));
                        if (visibleIds.has(id)) {
                            window._map.addLayer(layer);
                        } else {
                            window._map.removeLayer(layer);
                        }
                    });

                    console.log("Now showing only FMRs with satellite images.");
                } else {
                    alert("Failed to filter FMRs: " + data.message);
                }
            })
            .catch(err => {
                console.error("Error fetching FMRs with images:", err);
                alert("Could not fetch FMRs with images.");
            });
    } else {
        // Show all layers again
        Object.values(geoLayers).forEach(layer => {
            window._map.addLayer(layer);
        });
        console.log("Restored all FMRs.");
    }
}

// EVERYTHING MANUAL 

let manualFMRs = []; // [{ id, geometry: GeoJSON, name }]
let manualFMRCounter = 0;

function addManualFMRRow() {
    const container = document.getElementById('manual-fmr-container');
    const index = manualFMRCounter++;

    const row = document.createElement('div');
    row.classList.add('manual-fmr-row');
    row.id = `manual-fmr-${index}`;
    row.style.display = 'flex';
    row.style.alignItems = 'center';
    row.style.gap = '10px';
    row.style.marginBottom = '10px';

    const drawBtn = document.createElement('button');
    drawBtn.textContent = '🖊️ Draw FMR';
    drawBtn.onclick = () => drawManualLine(index);

    const input = document.createElement('input');
    input.type = 'text';
    input.placeholder = 'Enter FMR name';
    input.oninput = (e) => {
        const fmr = manualFMRs.find(f => f.id === index);
        if (fmr) fmr.name = e.target.value;
    };

    const deleteBtn = document.createElement('button');
    deleteBtn.textContent = '❌';
    deleteBtn.onclick = () => removeManualFMRRow(index);

    row.appendChild(drawBtn);
    row.appendChild(input);
    row.appendChild(deleteBtn);
    container.appendChild(row);

    manualFMRs.push({ id: index, name: '', geometry: null });

    // Optional: grow modal dynamically
    document.getElementById('processing-modal').style.height = 'auto';
}

function removeManualFMRRow(index) {
    document.getElementById(`manual-fmr-${index}`).remove();
    manualFMRs = manualFMRs.filter(f => f.id !== index);
}

function drawManualLine(index) {
    if (!window._map) return;

    toggleDrawingUI(true); // 🔍 Hide everything except map

    if (!window._drawControl) {
        window._drawControl = new L.Control.Draw({
            draw: { polygon: false, marker: false, circle: false, rectangle: false, circlemarker: false, polyline: true },
            edit: false
        });
        window._map.addControl(window._drawControl);
    }

    const drawHandler = new L.Draw.Polyline(window._map);
    drawHandler.enable();

    window._map.once(L.Draw.Event.CREATED, function (e) {
        const layer = e.layer;
        const geojson = layer.toGeoJSON();
        const fmr = manualFMRs.find(f => f.id === index);
        if (fmr) fmr.geometry = geojson.geometry;

        layer.addTo(window._map);

        // 🔙 Show UI back
        toggleDrawingUI(false);
    });
}


// Patch runProcessing to send manualFMRs
function runProcessing() {
    const workflowType = document.querySelector('input[name="workflow-type"]:checked').value;
    const imageType = document.getElementById('image-type').value;
    const processType = document.querySelector('input[name="process-type"]:checked').value;

    if (workflowType === 'manual') {
        const valid = manualFMRs.every(f => f.geometry && f.name);
        if (!valid) {
            alert("Please complete all drawn FMRs and name them.");
            return;
        }

        fetch('/process_fmr', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                workflow_type: 'manual',
                image_type: imageType,
                manual_fmrs: manualFMRs.map(f => ({
                    name: f.name,
                    geometry: f.geometry
                }))
            })
        })
        .then(res => res.json())
        .then(data => {
            if (data.status === 'success') {
                alert("Manual processing complete. Check results console.");
                console.log(data.results);
            } else {
                alert("Error: " + data.message);
            }
        })
        .catch(err => alert("Request error: " + err.message));

        return;
    }
}

function toggleManualSection() {
    const isManual = document.querySelector('input[name="workflow-type"]:checked').value === 'manual';
    document.getElementById('manual-fmr-section').style.display = isManual ? 'block' : 'none';
}

function toggleDrawingUI(showMapOnly) {
    // Hide or show the processing modal
    const modal = document.getElementById('processing-modal');
    if (modal) modal.style.display = showMapOnly ? 'none' : 'flex';

    // Hide or show the selection panel
    const panel = document.getElementById('selection-panel');
    if (panel) panel.style.display = showMapOnly ? 'none' : 'block';
}
