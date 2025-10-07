// PLease check 8/27
// JavaScript logic for FMR GUI

const selectedIds = new Set();
const geoLayers = {};
let currentMatchingImages = {};
const style = document.createElement('style');
const imageCache = {};  // key: image path, value: { base64, bounds }
const overlayLayers = {};  // key: image path, value: leaflet layer

let manualFMRs = [];          // { id, selectedFmrId, geometry, layer }
let manualFMRCounter = 0;

// 08/27: Updated to use overlayKey (FMR+image) while keeping window._map safe
function overlayImage({ image_base64, image_bounds }, overlayKey) {
    if (!window._map) {
        console.error('Map not available');
        return;
    }
    
    // Use overlayKey directly (already unique per FMR+image)
    const layerKey = `overlay_${btoa(overlayKey).replace(/[^a-zA-Z0-9]/g, '')}`;
    
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
        
        console.log(`Added overlay for: ${overlayKey}`);
        
    } catch (error) {
        console.error('Error creating image overlay:', error);
    }
}

// 08/27: Updated to remove overlays by overlayKey
function removeOverlay(overlayKey) {
    if (!window._map || !overlayKey) return;
    
    const layerKey = `overlay_${btoa(overlayKey).replace(/[^a-zA-Z0-9]/g, '')}`;
    const layer = overlayLayers[layerKey];
    
    if (layer) {
        try {
            window._map.removeLayer(layer);
            delete overlayLayers[layerKey];
            console.log(`Removed overlay for: ${overlayKey}`);
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

        // 08/27: Unique key per (FMR + image) to allow multiple cropped overlays
        const overlayKey = `${fmrId}_${imagePath}`;

        if (checkbox.checked) {
            // Show loading indicator
            const label = checkbox.nextElementSibling;
            const originalText = label.textContent;
            label.textContent = originalText + ' (Loading...)';
            
            if (imageCache[overlayKey]) {
                // 08/27: Use cached overlay specific to this FMR+image
                overlayImage(imageCache[overlayKey], overlayKey);
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
                        // 08/27: Cache per unique key (not just imagePath)
                        imageCache[overlayKey] = {
                            image_base64: data.image_data,
                            image_bounds: data.bounds
                        };
                        // Display the image
                        overlayImage(imageCache[overlayKey], overlayKey); // 08/27: pass overlayKey
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
            // 08/27: Remove overlay using unique key (FMR+image)
            removeOverlay(overlayKey);
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

    // 08/27: Preserve checked state before rebuilding
    const previouslyChecked = new Set();
    document.querySelectorAll(".image-checkbox:checked").forEach(cb => {
        previouslyChecked.add(`${cb.dataset.fmrId}_${cb.dataset.imagePath}`);
    });

    // 08/27: Preserve expanded/collapsed state before rebuilding
    const expandedState = {};
    document.querySelectorAll(".fmr-item").forEach(item => {
        const header = item.querySelector(".fmr-header");
        const fmrLabel = header?.querySelector("span")?.textContent || "";
        const fmrMatch = fmrLabel.match(/FMR-(\d+)/);
        if (fmrMatch) {
            const fmrId = fmrMatch[1];
            const isExpanded = item.querySelector(".image-list")?.classList.contains("show");
            expandedState[fmrId] = isExpanded;
        }
    });

    ul.innerHTML = "";

    selectedIds.forEach(id => {
        const li = document.createElement("li");
        li.classList.add("fmr-item");

        const header = document.createElement("div");
        header.classList.add("fmr-header");
        header.innerHTML = `<span><b>FMR-${id}</b></span><span class="toggle-icon">▶</span>`;
        li.appendChild(header);

        const images = currentMatchingImages[id] || [];

        // 08/27: Remove duplicates by unique path
        const seenPaths = new Set();
        const uniqueImages = images.filter(img => {
            if (seenPaths.has(img.path)) return false;
            seenPaths.add(img.path);
            return true;
        });

        const imageList = document.createElement("ul");
        imageList.classList.add("image-list");

        header.addEventListener("click", () => {
            imageList.classList.toggle("show");
            const icon = header.querySelector(".toggle-icon");
            icon.textContent = imageList.classList.contains("show") ? "▼" : "▶";
        });

        if (uniqueImages.length > 0) {
            uniqueImages.forEach((img, idx) => {
                const item = document.createElement("li");
                item.classList.add("image-option");

                const checkbox = document.createElement("input");
                checkbox.type = "checkbox";
                checkbox.classList.add("image-checkbox");
                checkbox.dataset.imagePath = img.path;
                checkbox.dataset.fmrId = id;
                checkbox.id = `img-${id}-${idx}`;
                checkbox.disabled = true;

                // 08/27: Restore checked state if previously selected
                if (previouslyChecked.has(`${id}_${img.path}`)) {
                    checkbox.checked = true;
                }

                const label = document.createElement("label");
                label.htmlFor = checkbox.id;
                const filename = img.filename || img.path.split(/[\\/]/).pop();
                const date = img.date ? ` (${img.date})` : "";
                label.textContent = " " + filename + date;

                item.appendChild(checkbox);
                item.appendChild(label);
                imageList.appendChild(item);

                setTimeout(() => {
                    checkbox.disabled = false;
                }, 300);

                checkbox.addEventListener("change", updateRunButtonState);
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

        // 08/27: Restore expanded/collapsed state
        if (expandedState[id]) {
            imageList.classList.add("show");
            const icon = header.querySelector(".toggle-icon");
            icon.textContent = "▼";
        }

        const layer = geoLayers["geoLayer_" + id];
        if (layer) layer.setStyle({color: "red", weight: 3.5});
    });
}

// 08/27: Enable/disable Run button dynamically
function updateRunButtonState() {
  const runBtn = document.getElementById("runBtn");
  if (!runBtn) return;

  // count selected checkboxes
  const anySelected = document.querySelectorAll(".image-checkbox:checked").length > 0;
  runBtn.disabled = !anySelected;
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
            updateClearButtonState();
            
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

document.addEventListener('DOMContentLoaded', function() {
});

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

// 08/13: also had modifications here
function runProcessing() {
  const workflowType = document.querySelector('input[name="workflow-type"]:checked').value;
  const imageType = document.getElementById('image-type').value; // still passed to keep API stable
  const processType = document.querySelector('input[name="process-type"]:checked').value; // unused in manual here

  // MANUAL PROCESSING
  if (workflowType === 'manual') {
    // validate rows
    const valid = manualFMRs.every(m => m.geometry && Number.isInteger(m.selectedFmrId));
    if (!valid) {
      alert('Please pick an FMR for each row and finish drawing the line.');
      return;
    }
    // send one request per manual row
    manualFMRs.forEach(m => {
      processFMR(null, null, 'manual', imageType, {
        selectedFmrId: m.selectedFmrId,
        geometry: m.geometry
      });
    });
    return;
  }

  
  const fmrIds = Array.from(selectedIds);
  const imagesToProcess = [];
  if (processType === 'selected') {
    document.querySelectorAll('.image-checkbox:checked').forEach(cb => {
      imagesToProcess.push({ fmr_id: parseInt(cb.dataset.fmrId), image_path: cb.dataset.imagePath });
    });
    if (imagesToProcess.length === 0) {
      alert('Please select at least one image to process');
      return;
    }
  } else {
    fmrIds.forEach(fid => {
      const imgs = currentMatchingImages[fid] || [];
      imgs.forEach(img => imagesToProcess.push({ fmr_id: fid, image_path: img.path }));
    });
  }
  if (imagesToProcess.length === 0) {
    alert('No images found to process');
    return;
  }
  imagesToProcess.forEach(item => processFMR(item.fmr_id, item.image_path, 'automatic', imageType));
}

// 8:13: Function to process FMRs
function processFMR(fmr_id, image_path, workflow_type, image_type, manualFMR = null) {
  const body = { workflow_type, image_type };
  if (workflow_type === 'manual' && manualFMR) {
        body.manual_fmr = {
        selected_fmr_id: manualFMR.selectedFmrId,
        geometry: manualFMR.geometry
        };
    } else {
        body.fmr_id = fmr_id;
        body.image_path = image_path;
  }

  fetch('/process_fmr', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    })
    .then(r => r.json())
    .then(data => {
        if (data.status === 'error') {
        console.error('Processing error:', data.message);
        alert(`Error: ${data.message}`);
        } else {
        console.log('Processing results:', data);
        alert('Processing complete.');
        }
    })
    .catch(err => {
        console.error(err);
        alert(`Error: ${err.message}`);
    });
}

// Update the run button state based on selections
function updateRunButtonState() {
    const runBtn = document.getElementById('runBtn');
    runBtn.disabled = selectedIds.size === 0;
}

// 08/27: Enable/disable Clear button depending on selected FMRs
function updateClearButtonState() {
    const clearBtn = document.getElementById("clearBtn");
    if (selectedIds.size > 0) {
        clearBtn.disabled = false;
    } else {
        clearBtn.disabled = true;
    }
}

// 08/27: Collapsible main controls (hide/show instead of shrinking)
document.addEventListener("DOMContentLoaded", () => {
  const toggleBtn = document.getElementById("toggle-main-controls");
  const mainControls = document.getElementById("selection-panel"); // target whole panel

  if (toggleBtn && mainControls) {
    toggleBtn.addEventListener("click", () => {
      const isHidden = mainControls.style.display === "none";
      mainControls.style.display = isHidden ? "block" : "none";
      toggleBtn.title = isHidden ? "Hide Controls" : "Show Controls";
    });

    // start collapsed by default
    mainControls.style.display = "none";
  }
});

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

                    console.log("Showing only FMRs with satellite images.");
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
 // 8/13: modified manual FMR drawing logic
function addManualFMRRow() {
    const index = manualFMRCounter++;

    const row = document.createElement('div');
    row.classList.add('manual-fmr-row');
    row.style.display = 'flex';
    row.style.alignItems = 'center';
    row.style.gap = '8px';
    row.style.marginBottom = '10px';
    row.id = `manual-fmr-${index}`;
 

    // Draw button - pencil icon only
    const drawBtn = document.createElement('button');
    drawBtn.classList.add('draw-fmr-btn');
    drawBtn.innerHTML = '<i class="fas fa-pencil-alt"></i>';
    drawBtn.title = 'Draw FMR';
    drawBtn.onclick = () => {
        const fmrId = parseInt(select.value);
        drawManualLine(index, fmrId);
    };

    // Dropdown of selected FMRs
    const select = document.createElement('select');
    selectedIds.forEach(id => {
        const opt = document.createElement('option');
        opt.value = id;
        opt.textContent = `FMR-${id}`;
        select.appendChild(opt);
    });

    // Delete button - X icon only
    const deleteBtn = document.createElement('button');
    deleteBtn.classList.add('delete-fmr-btn');
    deleteBtn.innerHTML = '<i class="fas fa-times"></i>';
    deleteBtn.title = 'Remove row';
    deleteBtn.onclick = () => removeManualFMRRow(index);

    row.appendChild(drawBtn);
    row.appendChild(select);
    row.appendChild(deleteBtn);
    document.getElementById('manual-fmr-container').appendChild(row);

    manualFMRs.push({ id: index, selectedFmrId: null, geometry: null });
}

function removeManualFMRRow(index) {
    document.getElementById(`manual-fmr-${index}`).remove();
    manualFMRs = manualFMRs.filter(f => f.id !== index);
}

// 8/13: Changes
function drawManualLine(index, selectedFmrId) {
    manualFMRs[index].selectedFmrId = selectedFmrId;

    // Hide UI, keep map only
    toggleDrawingUI(true);

    // 🔹 Hide all FMR lines while drawing
    Object.values(geoLayers).forEach(layer => {
        window._map.removeLayer(layer);
    });

    // Initialize draw control if not yet done
    if (!window._drawControl) {
        window._drawControl = new L.Control.Draw({
            draw: {
                polygon: false,
                marker: false,
                circle: false,
                rectangle: false,
                circlemarker: false,
                polyline: { shapeOptions: { color: '#222', weight: 4, opacity: 1.0 } }
            },
            edit: false
        });
        window._map.addControl(window._drawControl);
    }

    // Create polyline draw handler with custom style
    const drawHandler = new L.Draw.Polyline(window._map, {
        shapeOptions: { color: '#222', weight: 4, opacity: 1.0 }
    });
    drawHandler.enable();

    // When drawing is finished
    window._map.once(L.Draw.Event.CREATED, function (e) {
        const layer = e.layer;
        const geojson = layer.toGeoJSON();
        manualFMRs[index].geometry = geojson.geometry;

        // Add the drawn line to the map
        layer.addTo(window._map);

        // 🔹 Restore all FMR lines
        Object.values(geoLayers).forEach(layer => {
            window._map.addLayer(layer);
        });

        // Show UI back
        toggleDrawingUI(false);
    });
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

// ============= UPDATE NOTIFICATION SYSTEM =============

let updateCheckInterval = null;
let hasUpdatesAvailable = false;

// Create update notification element
function createUpdateNotification() {
    const notification = document.createElement('div');
    notification.id = 'update-notification';
    notification.style.cssText = `
        position: fixed;
        top: 10px;
        right: 10px;
        background: #ff9800;
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        z-index: 10001;
        display: none;
        font-size: 14px;
        max-width: 300px;
    `;
    notification.innerHTML = `
        <div style="display: flex; align-items: center; gap: 10px;">
            <i class="fas fa-exclamation-circle"></i>
            <div>
                <div style="font-weight: bold;">Updates Available</div>
                <div id="update-details" style="font-size: 12px; margin-top: 4px;"></div>
                <button onclick="applyUpdates()" style="
                    margin-top: 8px;
                    background: white;
                    color: #ff9800;
                    border: none;
                    padding: 4px 12px;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 12px;
                ">Apply Updates</button>
            </div>
        </div>
    `;
    document.body.appendChild(notification);
}

// Check for updates periodically
function startUpdateChecker() {
    // Check immediately
    checkForUpdates();
    
    // Then check every 30 seconds
    updateCheckInterval = setInterval(checkForUpdates, 30000);
}

function stopUpdateChecker() {
    if (updateCheckInterval) {
        clearInterval(updateCheckInterval);
        updateCheckInterval = null;
    }
}

function checkForUpdates() {
    fetch('/check_updates')
        .then(res => res.json())
        .then(data => {
            const notification = document.getElementById('update-notification');
            const updateBtn = document.getElementById('refresh-btn');
            
            if (data.has_updates) {
                hasUpdatesAvailable = true;
                
                // Show notification
                if (notification) {
                    const details = document.getElementById('update-details');
                    details.textContent = `${data.new_shapefiles} new shapefiles, ${data.new_rasters} new images`;
                    notification.style.display = 'block';
                }
                
                // Update refresh button
                if (updateBtn) {
                    updateBtn.innerHTML = '<i class="fas fa-download"></i> Apply Updates';
                    updateBtn.style.backgroundColor = '#ff9800';
                    updateBtn.classList.add('pulse-animation');
                }
            } else {
                hasUpdatesAvailable = false;
                
                // Hide notification
                if (notification) {
                    notification.style.display = 'none';
                }
                
                // Reset refresh button
                if (updateBtn) {
                    updateBtn.innerHTML = '<i class="fas fa-sync-alt"></i> Refresh';
                    updateBtn.style.backgroundColor = '#4CAF50';
                    updateBtn.classList.remove('pulse-animation');
                }
            }
        })
        .catch(err => console.error('Error checking for updates:', err));
}

// Apply updates (incremental)
function applyUpdates() {
    manualRefresh();
}

// Manual refresh function
function manualRefresh() {
    // Show loading state
    const refreshBtn = document.getElementById('refresh-btn');
    const originalContent = refreshBtn ? refreshBtn.innerHTML : '';
    if (refreshBtn) {
        refreshBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Updating...';
        refreshBtn.disabled = true;
    }
    
    // Hide update notification
    const notification = document.getElementById('update-notification');
    if (notification) {
        notification.style.display = 'none';
    }
    
    fetch('/manual_refresh', { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            if (data.status === 'success') {
                // Show success message
                showToast(`✓ Update complete! ${data.message}`, 'success');
                
                // Reload page if shapefiles were updated
                if (data.shapefiles_updated) {
                    setTimeout(() => {
                        location.reload();
                    }, 2000);
                } else if (data.new_database_entries > 0) {
                    // Just refresh the FMR list if only database was updated
                    updateFMRList();
                }
                
                // Reset button
                if (refreshBtn) {
                    refreshBtn.innerHTML = '<i class="fas fa-sync-alt"></i> Refresh';
                    refreshBtn.disabled = false;
                    refreshBtn.style.backgroundColor = '#4CAF50';
                }
            } else {
                showToast('Update failed: ' + (data.message || 'Unknown error'), 'error');
                if (refreshBtn) {
                    refreshBtn.innerHTML = originalContent;
                    refreshBtn.disabled = false;
                }
            }
        })
        .catch(err => {
            console.error('Error during refresh:', err);
            showToast('Update failed: ' + err.message, 'error');
            if (refreshBtn) {
                refreshBtn.innerHTML = originalContent;
                refreshBtn.disabled = false;
            }
        });
}

// Full rebuild (for troubleshooting)
function fullRebuild() {
    if (!confirm('This will completely rebuild the database. This may take several minutes. Continue?')) {
        return;
    }
    
    showToast('Starting full database rebuild...', 'info');
    
    fetch('/full_rebuild', { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            if (data.status === 'success') {
                showToast('✓ Full rebuild complete!', 'success');
                setTimeout(() => location.reload(), 2000);
            } else {
                showToast('Rebuild failed: ' + data.message, 'error');
            }
        })
        .catch(err => {
            console.error('Error during rebuild:', err);
            showToast('Rebuild failed: ' + err.message, 'error');
        });
}

// Toast notification system
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = 'toast-notification';
    
    const colors = {
        'success': '#4CAF50',
        'error': '#f44336',
        'warning': '#ff9800',
        'info': '#2196F3'
    };
    
    toast.style.cssText = `
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        background: ${colors[type]};
        color: white;
        padding: 12px 24px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        z-index: 10002;
        font-size: 14px;
        animation: slideUp 0.3s ease-out;
    `;
    
    toast.textContent = message;
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'fadeOut 0.3s ease-out';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// Add CSS animations
const animationStyles = document.createElement('style');
animationStyles.textContent = `
    @keyframes slideUp {
        from {
            transform: translate(-50%, 100%);
            opacity: 0;
        }
        to {
            transform: translate(-50%, 0);
            opacity: 1;
        }
    }
    
    @keyframes fadeOut {
        from {
            opacity: 1;
        }
        to {
            opacity: 0;
        }
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.6; }
        100% { opacity: 1; }
    }
    
    .pulse-animation {
        animation: pulse 2s infinite;
    }
    
    #database-stats {
        position: fixed;
        top: 10px;
        left: 10px;
        background: rgba(255, 255, 255, 0.95);
        padding: 10px 15px;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        z-index: 1000;
        font-size: 12px;
    }
    
    #database-stats h4 {
        margin: 0 0 8px 0;
        font-size: 14px;
    }
    
    #database-stats div {
        margin: 4px 0;
    }
    
    .refresh-controls {
        display: flex;
        gap: 8px;
        align-items: center;
        margin-top: 10px;
    }
    
    .refresh-btn {
        background: #4CAF50;
        color: white;
        border: none;
        padding: 6px 12px;
        border-radius: 4px;
        cursor: pointer;
        font-size: 12px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    
    .refresh-btn:hover {
        opacity: 0.9;
    }
    
    .refresh-btn:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }
    
    .rebuild-btn {
        background: #666;
        color: white;
        border: none;
        padding: 4px 8px;
        border-radius: 4px;
        cursor: pointer;
        font-size: 11px;
    }
`;
document.head.appendChild(animationStyles);

// Initialize update checker on page load
document.addEventListener('DOMContentLoaded', function() {
    // Create update notification
    createUpdateNotification();
    
    // Add database stats panel
    createDatabaseStats();
    
    // Start checking for updates
    startUpdateChecker();
    
    // Load initial stats
    loadDatabaseStats();
});

// Create database stats panel
function createDatabaseStats() {
    const statsPanel = document.createElement('div');
    statsPanel.id = 'database-stats';
    statsPanel.innerHTML = `
        <h4>📊 Database Status</h4>
        <div id="stats-content">
            <div>FMRs: <span id="stat-fmrs">-</span></div>
            <div>Entries: <span id="stat-entries">-</span></div>
            <div>Last Update: <span id="stat-update">-</span></div>
        </div>
        <div class="refresh-controls">
            <button id="refresh-btn" class="refresh-btn" onclick="manualRefresh()">
                <i class="fas fa-sync-alt"></i> Refresh
            </button>
            <button class="rebuild-btn" onclick="fullRebuild()" title="Full Rebuild">
                <i class="fas fa-hammer"></i>
            </button>
        </div>
    `;
    document.body.appendChild(statsPanel);
}

// Load database statistics
function loadDatabaseStats() {
    fetch('/get_update_stats')
        .then(res => res.json())
        .then(data => {
            document.getElementById('stat-fmrs').textContent = data.total_fmrs || '0';
            document.getElementById('stat-entries').textContent = data.database_entries || '0';
            document.getElementById('stat-update').textContent = data.last_update || 'Never';
        })
        .catch(err => console.error('Error loading stats:', err));
}

// Update stats after refresh
function updateStatsAfterRefresh() {
    loadDatabaseStats();
}

// Override the existing clearSelections to stop update checker during processing
const originalClearSelections = clearSelections;
clearSelections = function() {
    // Stop checking for updates while processing
    stopUpdateChecker();
    
    // Call original function
    originalClearSelections();
    
    // Restart checker after a delay
    setTimeout(() => startUpdateChecker(), 5000);
};