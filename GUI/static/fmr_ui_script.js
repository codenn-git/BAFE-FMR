// PLease check 8/27
// JavaScript logic for FMR GUI

const selectedIds = new Set();
const geoLayers = {};
const manualCenterlineLayers = {};  //11/21
const manualCenterlineMeta = {};    // 11/24
let currentMatchingImages = {};
const style = document.createElement('style');
const imageCache = {};  // key: image path, value: { base64, bounds }
const overlayLayers = {};  // key: image path, value: leaflet layer

//11/23: filters for images + processing status
let imageFilterMode = "all";           // "all" | "with" | "without"
let showUnprocessed = true;
let showAutomatic = true;
let showManual = true;
let fmrHasImageIds = null;             // Set of FMR IDs that have at least one image

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
        if (layer) layer.setStyle({color: "blue", weight: 3.5});
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

// 11/11 updated runProcessing to accommodate the new ManualRoadProcessor workflow in fmr_processing.py
function runProcessing() {
  const workflowType = document.querySelector('input[name="workflow-type"]:checked').value;
  const imageType = document.getElementById('image-type').value; // still passed to keep API stable
  const processType = document.querySelector('input[name="process-type"]:checked').value; // unused in manual here

  // MANUAL PROCESSING
  if (workflowType === 'manual') {
    // Validate that manual FMRs have been drawn
    if (manualFMRs.length === 0) {
      alert('Please add at least one FMR row for manual processing');
      return;
    }
    
    // Check that all have geometry drawn
    const incompleteRows = manualFMRs.filter(m => !m.geometry);
    if (incompleteRows.length > 0) {
      alert('Please finish drawing centerlines for all FMR rows');
      return;
    }
    
    let hasErrors = false;
    manualFMRs.forEach(m => {
        // Get image_path for each manual FMR
        const fmrId = m.selectedFmrId;
        
        // Try to get a checked image for this FMR
        let imagePath = getSelectedImagePath(fmrId);
        
        // If no checked image, try to get the first available image for this FMR
        if (!imagePath) {
            const availableImages = currentMatchingImages[fmrId] || [];
            if (availableImages.length > 0) {
                imagePath = availableImages[0].path;
                console.log(`Using first available image for FMR-${fmrId}: ${imagePath}`);
            }
        }
        
        if (!imagePath) {
            alert(`No image available for FMR-${fmrId}. Please select an FMR with available images.`);
            hasErrors = true;
            return;
        }
        
        // Process this manual FMR with the image path
        processFMR(null, imagePath, 'manual', imageType, {
            selectedFmrId: m.selectedFmrId,
            geometry: m.geometry
        });
    });
    
    if (!hasErrors) {
        hideProcessingModal();
    }
    return;
  }
  
  // AUTOMATIC PROCESSING 
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
  hideProcessingModal();
}

// 11/11 updated to accommodate the new ManualRoadProcessor workflow in fmr_processing.py
function processFMR(fmr_id, image_path, workflow_type, image_type, manualFMR = null) {
  const body = { 
    workflow_type, 
    image_type,
    image_path
  };
  
  if (workflow_type === 'manual' && manualFMR) {
    body.manual_fmr = {
      selected_fmr_id: manualFMR.selectedFmrId,
      geometry: manualFMR.geometry
    };
  } else {
    // Automatic workflow
    body.fmr_id = fmr_id;
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
      alert(`Processing complete for ${workflow_type === 'manual' ? 'Manual FMR' : `FMR-${fmr_id}`}`);
    }
  })
  .catch(err => {
    console.error(err);
    alert(`Error: ${err.message}`);
  });
}

// 11/11: function to get selected FMR path
function getSelectedImagePath(fmrId) {
    // Get the checked image checkbox for this FMR
    const checkedBox = document.querySelector(
        `.image-checkbox:checked[data-fmr-id="${fmrId}"]`
    );
    return checkedBox ? checkedBox.dataset.imagePath : null;
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

//11/23: image filter logic (by radio group)
function passesImageFilter(id) {
    if (imageFilterMode === "all") return true;

    // If we don't know which FMR this is, don't block it based on images.
    if (id == null || Number.isNaN(id)) return true;

    const hasImage = fmrHasImageIds && fmrHasImageIds.has(id);
    if (imageFilterMode === "with")    return !!hasImage;
    if (imageFilterMode === "without") return !hasImage;

    return true;
}

//11/23: processing-status filter logic (by Processing Type) for base FMR lines
function passesStatusFilter(id) {
    const info = (window.fmrProcessingInfo || {})[id] || {};
    const pt = (info.processingType || "").toLowerCase();

    if (pt === "manual")    return showManual;
    if (pt === "automatic") return showAutomatic;

    // Everything else (including empty) is treated as "unprocessed"
    return showUnprocessed;
}

//11/24: recompute which FMR layers are visible (base FMR + centerlines)
function recomputeFMRLayerVisibility() {
    if (!window._map) return;

    // 1) Base FMR polylines from the master shapefile
    Object.entries(geoLayers).forEach(([key, layer]) => {
        const id = parseInt(key.replace("geoLayer_", ""), 10);
        if (Number.isNaN(id)) return;

        const visible = passesImageFilter(id) && passesStatusFilter(id);

        if (visible) {
            if (!window._map.hasLayer(layer)) {
                window._map.addLayer(layer);
            }
        } else {
            if (window._map.hasLayer(layer)) {
                window._map.removeLayer(layer);
            }
        }
    });

    // 2) Centerline overlays (manual + automatic) from fmr_centerlines_migo.geojson
    Object.entries(manualCenterlineLayers).forEach(([key, layer]) => {
        const meta = manualCenterlineMeta[key] || {};
        const fmrId = meta.fmrId;
        const procType = (meta.processingType || "").toLowerCase();

        // Status filter for this overlay
        let visibleByStatus = true;
        if (procType === "manual")        visibleByStatus = showManual;
        else if (procType === "automatic") visibleByStatus = showAutomatic;
        else                               visibleByStatus = showUnprocessed;

        // Image filter: only applies if we have a numeric FMR ID
        const visibleByImage =
            (fmrId != null && !Number.isNaN(fmrId)) ? passesImageFilter(fmrId) : true;

        const visible = visibleByStatus && visibleByImage;

        if (visible) {
            if (!window._map.hasLayer(layer)) {
                window._map.addLayer(layer);
            }
        } else {
            if (window._map.hasLayer(layer)) {
                window._map.removeLayer(layer);
            }
        }
    });
}

//11/23: set the Images filter mode ("all" | "with" | "without")
function setImageFilterMode(mode) {
    imageFilterMode = mode;

    if (mode === "all") {
        recomputeFMRLayerVisibility();
        return;
    }

    // For "with" or "without", we need to know which FMRs HAVE images
    if (fmrHasImageIds) {
        recomputeFMRLayerVisibility();
        return;
    }

    fetch('/get_fmrs_with_images')
        .then(res => res.json())
        .then(data => {
            if (data.status === "success" && Array.isArray(data.fmr_ids)) {
                fmrHasImageIds = new Set(
                    data.fmr_ids
                        .map(id => parseInt(id, 10))
                        .filter(id => !Number.isNaN(id))
                );
                recomputeFMRLayerVisibility();
            } else {
                alert("Failed to get FMRs with images: " + (data.message || "Unknown error"));
            }
        })
        .catch(err => {
            console.error("Error fetching FMRs with images:", err);
            alert("Could not fetch FMRs with images.");
        });
}

//11/23: update processing-status filters
function updateStatusFilterUnprocessed(checked) {
    showUnprocessed = checked;
    recomputeFMRLayerVisibility();
}

function updateStatusFilterAutomatic(checked) {
    showAutomatic = checked;
    recomputeFMRLayerVisibility();
}

function updateStatusFilterManual(checked) {
    showManual = checked;
    recomputeFMRLayerVisibility();
}

//11/23: open/close the top-right filter panel
function toggleFMRFilterPanel() {
    const panel = document.getElementById('fmr-filter-panel');
    if (!panel) return;
    const isHidden = panel.style.display === '' || panel.style.display === 'none';
    panel.style.display = isHidden ? 'block' : 'none';
}

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

//11/21: manual FMR row with comfortable spacing between buttons and dropdown
function addManualFMRRow() {
    const index = manualFMRCounter++;

    const row = document.createElement('div');
    row.id = `manual-fmr-${index}`;
    row.classList.add('manual-fmr-row');

    // Force a nice horizontal layout with spacing
    row.style.display = 'flex';
    row.style.alignItems = 'center';
    row.style.gap = '8px';          // space between each control
    row.style.marginBottom = '10px';

    // Draw button (pencil)
    const drawBtn = document.createElement('button');
    drawBtn.classList.add('draw-fmr-btn');
    drawBtn.innerHTML = '<i class="fas fa-pencil-alt"></i>';
    drawBtn.title = 'Draw / edit centerline';
    drawBtn.style.padding = '4px 6px';
    drawBtn.style.marginRight = '2px';

    drawBtn.onclick = () => {
        const selectEl = document.getElementById(`manual-fmr-select-${index}`);
        if (!selectEl || !selectEl.value) {
            alert('Please select an FMR before drawing.');
            return;
        }
        const fmrId = parseInt(selectEl.value, 10);
        if (Number.isNaN(fmrId)) {
            alert('Invalid FMR selection.');
            return;
        }
        drawManualLine(index, fmrId);
    };

    // Load previous manual line button (history icon)
    const loadBtn = document.createElement('button');
    loadBtn.classList.add('draw-fmr-btn');
    loadBtn.innerHTML = '<i class="fas fa-history"></i>';
    loadBtn.title = 'Load previous manual line for this FMR';
    loadBtn.style.padding = '4px 6px';
    loadBtn.style.marginRight = '2px';

    loadBtn.onclick = () => loadPreviousManualLine(index);

    // Dropdown of selected FMRs
    const select = document.createElement('select');
    select.id = `manual-fmr-select-${index}`;
    select.style.minWidth = '150px';
    select.style.padding = '2px 4px';
    select.style.marginRight = '2px';

    select.innerHTML = '<option value="">-- Select FMR --</option>';
    selectedIds.forEach(id => {
        const opt = document.createElement('option');
        opt.value = id;
        opt.textContent = `FMR-${id}`;
        select.appendChild(opt);
    });

    select.onchange = () => {
        const val = select.value;
        const rowInfo = manualFMRs[index];
        if (!rowInfo) return;
        if (!val) {
            rowInfo.selectedFmrId = null;
            return;
        }
        const fmrId = parseInt(val, 10);
        rowInfo.selectedFmrId = Number.isNaN(fmrId) ? null : fmrId;
    };

    // Delete button (red X)
    const deleteBtn = document.createElement('button');
    deleteBtn.classList.add('delete-fmr-btn');
    deleteBtn.innerHTML = '<i class="fas fa-times"></i>';
    deleteBtn.title = 'Remove row';
    deleteBtn.style.padding = '4px 6px';

    deleteBtn.onclick = () => removeManualFMRRow(index);

    row.appendChild(drawBtn);
    row.appendChild(loadBtn);
    row.appendChild(select);
    row.appendChild(deleteBtn);

    document.getElementById('manual-fmr-container').appendChild(row);

    // track this row
    manualFMRs[index] = { id: index, selectedFmrId: null, geometry: null };
}

function removeManualFMRRow(index) {
  const rowEl = document.getElementById(`manual-fmr-${index}`);
  if (rowEl) {
    rowEl.remove();
  }
  manualFMRs = manualFMRs.filter(f => f.id !== index);
}

//11/21: load latest manual centerline for the selected FMR from backend and enter edit mode
function loadPreviousManualLine(index) {
  const rowInfo = manualFMRs.find(f => f.id === index);
  if (!rowInfo) {
    alert('Manual row not found.');
    return;
  }

  // Resolve selected FMR ID
  let selectedFmrId = rowInfo.selectedFmrId;
  if (selectedFmrId == null) {
    const selectEl = document.getElementById(`manual-fmr-select-${index}`);
    if (selectEl && selectEl.value) {
      const parsed = parseInt(selectEl.value, 10);
      if (!Number.isNaN(parsed)) {
        selectedFmrId = parsed;
        rowInfo.selectedFmrId = parsed;
      }
    }
  }

  if (selectedFmrId == null || Number.isNaN(selectedFmrId)) {
    alert('Please select an FMR first before loading a previous line.');
    return;
  }

  fetch('/get_manual_centerline', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ selected_fmr_id: selectedFmrId })
  })
    .then(res => res.json())
    .then(data => {
      if (data.status !== 'success') {
        alert(data.message || 'No saved manual line found for this FMR.');
        return;
      }

      const geom = data.geometry;
      if (!geom || geom.type !== 'LineString' || !Array.isArray(geom.coordinates)) {
        alert('Invalid geometry returned for this FMR.');
        return;
      }

      // Store geometry into this row and immediately enter draw/edit mode
      rowInfo.geometry = geom;
      drawManualLine(index, selectedFmrId);
    })
    .catch(err => {
      console.error('Error loading manual centerline:', err);
      alert('Error loading previous manual line. See console for details.');
    });
}

// ============================================================================
// Manual Drawing — Shared Helpers
// ============================================================================

//11/03: lightweight Chaikin smoothing helper (shared)
function smoothLineChaikin(latlngs, iterations = 1) { //11/03: new
  let pts = latlngs.map(ll => [ll.lat, ll.lng]); //11/03: new
  for (let it = 0; it < iterations; it++) { //11/03: new
    const out = []; //11/03: new
    for (let i = 0; i < pts.length - 1; i++) { //11/03: new
      const [x0, y0] = pts[i], [x1, y1] = pts[i+1]; //11/03: new
      out.push([0.75*x0 + 0.25*x1, 0.75*y0 + 0.25*y1]); //11/03: new
      out.push([0.25*x0 + 0.75*x1, 0.25*y0 + 0.75*y1]); //11/03: new
    } //11/03: new
    pts = [pts[0], ...out, pts[pts.length - 1]]; //11/03: new
  } //11/03: new
  return pts.map(p => L.latLng(p[0], p[1])); //11/03: new
}

//11/03: measure polyline length in meters (shared)
function measurePolylineMeters(latlngs) { //11/03: new
  if (!window._map || latlngs.length < 2) return 0; //11/03: new
  let m = 0; //11/03: new
  for (let i = 0; i < latlngs.length - 1; i++) { //11/03: new
    m += window._map.distance(latlngs[i], latlngs[i+1]); //11/03: new
  } //11/03: new
  return m; //11/03: new
}

//11/03: light client-side densify for nicer preview (shared)
function densifyLatLngs(latlngs, stepMeters = 2) { //11/03: new
  const out = []; //11/03: new
  for (let i = 0; i < latlngs.length - 1; i++) { //11/03: new
    const a = latlngs[i], b = latlngs[i+1]; //11/03: new
    out.push(a); //11/03: new
    const seg = window._map.distance(a, b); //11/03: new
    const n = Math.max(0, Math.floor(seg / stepMeters) - 1); //11/03: new
    for (let k = 1; k <= n; k++) { //11/03: new
      const t = k / (n + 1); //11/03: new
      out.push(L.latLng(a.lat + t*(b.lat - a.lat), a.lng + t*(b.lng - a.lng))); //11/03: new
    } //11/03: new
  } //11/03: new
  out.push(latlngs[latlngs.length - 1]); //11/03: new
  return out; //11/03: new
}

// ============================================================================
// Manual Drawing — HUD (QoL: Cursor-Following Tooltip)
// ============================================================================

let _lengthTip = null; //11/03: new

//11/03: create/ensure a floating tooltip near the cursor
function ensureMeasureTooltip() { //11/03: new
  if (_lengthTip) return _lengthTip; //11/03: new
  _lengthTip = document.createElement('div'); //11/03: new
  _lengthTip.id = 'draw-length-tip'; //11/03: new
  _lengthTip.style.position = 'fixed'; //11/03: new
  _lengthTip.style.zIndex = 9999; _lengthTip.style.pointerEvents = 'none'; //11/03: new
  _lengthTip.style.left = '0px'; _lengthTip.style.top = '0px'; //11/03: new
  _lengthTip.style.padding = '6px 10px'; //11/03: new
  _lengthTip.style.background = 'rgba(0,0,0,0.65)'; _lengthTip.style.color = '#fff'; //11/03: new
  _lengthTip.style.borderRadius = '6px'; _lengthTip.style.font = '12px/1.2 sans-serif'; //11/03: new
  _lengthTip.style.borderTop = '2px solid #4ade80'; //11/03: new
  _lengthTip.style.borderBottom = '2px solid #4ade80'; //11/03: new
  _lengthTip.textContent = 'Length: 0 m'; //11/03: new
  document.body.appendChild(_lengthTip); //11/03: new
  return _lengthTip; //11/03: new
}

//11/03: update text in tooltip
function updateMeasureTooltipText(meters) { //11/03: new
  const tip = ensureMeasureTooltip(); //11/03: new
  tip.textContent = `Length: ${meters.toFixed(1)} m`; //11/03: new
}

//11/03: move tooltip near cursor
function moveMeasureTooltip(clientX, clientY) { //11/03: new
  const tip = ensureMeasureTooltip(); //11/03: new
  const offset = 16; //11/03: new
  tip.style.left = `${clientX + offset}px`; //11/03: new
  tip.style.top  = `${clientY + offset}px`; //11/03: new
}

//11/03: remove tooltip when done
function removeMeasureTooltip() { //11/03: new
  if (_lengthTip && _lengthTip.parentNode) _lengthTip.parentNode.removeChild(_lengthTip); //11/03: new
  _lengthTip = null; //11/03: new
}

// ============================================================================
// Manual Drawing — Hotkeys (ESC/Enter/Backspace/Arrows)
// ============================================================================

//11/21: respect ctx.mode so edit-mode only uses ESC/Enter, not Backspace/Arrows
function attachManualDrawHotkeys(ctx) {
  const onKeyDown = (e) => {
    if (!ctx.active) return;
    const key = e.key || '';
    const isEsc = key === 'Escape' || key === 'Esc' || e.keyCode === 27;
    const isBackspace = key === 'Backspace' || e.keyCode === 8;
    const isEnter = key === 'Enter' || e.keyCode === 13;
    const isArrow = ['ArrowUp','ArrowDown','ArrowLeft','ArrowRight'].includes(key) ||
                    [37,38,39,40].includes(e.keyCode);

    const inEditMode = ctx.mode === 'edit';

    // Prevent other handlers (or the browser) from swallowing the event
    if (isEsc || isBackspace || isEnter || isArrow) {
      e.preventDefault(); e.stopPropagation();
    }

    if (isEsc) {
      // Cancel drawing: remove temp polyline, HUD, listeners, and restore styles
      try { if (ctx.layer) window._map.removeLayer(ctx.layer); } catch(_) {}
      try { window._map.off('click', ctx._onClick); window._map.off('mousemove', ctx._onMove); } catch(_) {}
      try { if (ctx._restore) ctx._restore.forEach(fn => fn()); } catch(_) {}
      try { removeMeasureTooltip(); } catch(_) {}
      ctx.active = false;
      toggleDrawingUI(false);
      hideManualBanner();
      if (ctx._detachHotkeys) ctx._detachHotkeys();
      setManualCursor(false);
      return;
    }

    if (isEnter) {
      if (ctx.layer && ctx.latlngs.length >= 2) {
        hideManualBanner();
        setManualCursor(false);
        if (ctx._detachHotkeys) ctx._detachHotkeys();
        ctx.finish();
      }
      return;
    }

    // In edit mode, ignore Backspace & Arrow logic
    if (inEditMode) return;

    if (isBackspace) {
      if (ctx.latlngs.length > 1) {
        ctx.latlngs.pop();
        ctx.layer.setLatLngs(ctx.latlngs);
        updateMeasureTooltipText(measurePolylineMeters(ctx.latlngs));
      }
      return;
    }

    if (isArrow) {
      if (!ctx.latlngs.length) return;
      const step = 0.000003; // ~0.3 m-ish
      const kc = e.keyCode;
      let last = ctx.latlngs[ctx.latlngs.length - 1];
      if (key === 'ArrowUp' || kc === 38)    last = L.latLng(last.lat + step, last.lng);
      if (key === 'ArrowDown' || kc === 40)  last = L.latLng(last.lat - step, last.lng);
      if (key === 'ArrowLeft' || kc === 37)  last = L.latLng(last.lat, last.lng - step);
      if (key === 'ArrowRight' || kc === 39) last = L.latLng(last.lat, last.lng + step);
      ctx.latlngs[ctx.latlngs.length - 1] = last;
      ctx.layer.setLatLngs(ctx.latlngs);
      updateMeasureTooltipText(measurePolylineMeters(ctx.latlngs));
    }
  };

  // Listen on document (captures more cases); use capture phase to beat other handlers
  document.addEventListener('keydown', onKeyDown, true);

  // Provide a cleanup hook so callers can detach reliably
  ctx._detachHotkeys = () => {
    document.removeEventListener('keydown', onKeyDown, true);
    ctx._detachHotkeys = null;
  };
}

// ============================================================================
// Manual Drawing — Banner (instructions ribbon)
// ============================================================================

// 11/03: create/destroy a slim banner with instructions
function showManualBanner() { // 11/03: new
  if (document.getElementById('manual-banner')) return; // 11/03: new
  const b = document.createElement('div'); // 11/03: new
  b.id = 'manual-banner'; // 11/03: new
  b.style.position = 'fixed'; b.style.left = 0; b.style.right = 0; b.style.top = 0; // 11/03: new
  b.style.zIndex = 9998; b.style.textAlign = 'center'; // 11/03: new
  b.style.background = 'rgba(17,17,17,0.9)'; b.style.color = '#fff'; // 11/03: new
  b.style.borderTop = '2px solid #4ade80'; b.style.borderBottom = '2px solid #4ade80'; // 11/03: new
  b.style.padding = '6px 10px'; b.style.font = '12px/1.2 sans-serif'; // 11/03: new
  b.textContent = ' · Manual mode: Click to add points · Enter = finish · ESC = cancel · Backspace = undo · '; // 11/03: new
  document.body.appendChild(b); // 11/03: new
}

function hideManualBanner() { // 11/03: new
  const b = document.getElementById('manual-banner'); // 11/03: new
  if (b && b.parentNode) b.parentNode.removeChild(b); // 11/03: new
}

//===========================================================
// Manual Drawing — Core (drawManualLine)
// ============================================================================

//11/21: unified manual draw/edit (point-append) with bounded vertices to avoid slowdown
function drawManualLine(index, selectedFmrId) {
  const rowInfo = manualFMRs[index];
  if (!rowInfo) return;

  rowInfo.selectedFmrId = selectedFmrId;
  toggleDrawingUI(true);
  showManualBanner();
  setManualCursor(true);

  const restoreFns = [];
  Object.values(geoLayers).forEach(layer => {
    if (layer.setStyle) {
      const prev = { ...layer.options };
      restoreFns.push(() => { try { layer.setStyle(prev); } catch (e) {} });
      try {
        layer.setStyle({ color: '#999', weight: 1, opacity: 0.5 });
      } catch (e) {}
    }
  });

  const latlngs = [];
  let poly = null;
  const MAX_EDIT_VERTICES = 300; // hard cap so geometries don't explode

  // Detect if this row already has a saved geometry
  const hasExisting =
    rowInfo.geometry &&
    rowInfo.geometry.type === 'LineString' &&
    Array.isArray(rowInfo.geometry.coordinates);

  // Start from existing geometry (but downsample if it's too dense)
  if (hasExisting) {
    const coords = rowInfo.geometry.coordinates; // [lng, lat]
    const baseLatLngs = coords.map(([lng, lat]) => L.latLng(lat, lng));

    if (baseLatLngs.length > MAX_EDIT_VERTICES) {
      const sampled = [];
      const step = (baseLatLngs.length - 1) / (MAX_EDIT_VERTICES - 1);
      for (let i = 0; i < MAX_EDIT_VERTICES; i++) {
        const idx = Math.round(i * step);
        sampled.push(baseLatLngs[idx]);
      }
      sampled.forEach(ll => latlngs.push(ll));
    } else {
      baseLatLngs.forEach(ll => latlngs.push(ll));
    }
  }

  // Create the working polyline (blank for new FMR; prefilled for edit)
  poly = L.polyline(latlngs, { color: '#00d', weight: 3, opacity: 0.95 }).addTo(window._map);
  poly.bringToFront();
  updateMeasureTooltipText(measurePolylineMeters(latlngs));

  const ctx = {
    layer: poly,
    latlngs,
    active: true,
    _onClick: null,
    _onMove: null,
    _restore: restoreFns,
    mode: 'draw',          // treat edit as draw so Backspace/Arrows still work
    finish: null,
    _detachHotkeys: null
  };

  // Click: append a new point to the end (same behavior for new + edit)
  function onClick(e) {
    latlngs.push(e.latlng);
    poly.setLatLngs(latlngs);
    poly.bringToFront();
    updateMeasureTooltipText(measurePolylineMeters(latlngs));
    if (e.originalEvent) {
      moveMeasureTooltip(e.originalEvent.clientX, e.originalEvent.clientY);
    }
  }

  // Mouse move: ghost segment from last point to cursor
  function onMove(e) {
    if (!latlngs.length) {
      if (e.originalEvent) {
        moveMeasureTooltip(e.originalEvent.clientX, e.originalEvent.clientY);
      }
      return;
    }
    const tmp = [...latlngs, e.latlng];
    poly.setLatLngs(tmp);
    poly.bringToFront();
    updateMeasureTooltipText(measurePolylineMeters(tmp));
    if (e.originalEvent) {
      moveMeasureTooltip(e.originalEvent.clientX, e.originalEvent.clientY);
    }
  }

  ctx._onClick = onClick;
  ctx._onMove = onMove;

  window._map.on('click', onClick);
  window._map.on('mousemove', onMove);

  ctx.finish = () => {
    if (latlngs.length < 2) return;

    const isNew = !hasExisting;
    let finalLL;

    if (isNew) {
      // First-time draw: nice smooth curve with some densify
      const applySmooth = true;
      const densified = densifyLatLngs(latlngs, 4);  // ~2 m spacing
      const smoothIterations = 8;
      finalLL = applySmooth ? smoothLineChaikin(densified, smoothIterations) : densified;
    } else {
      // Edit: do NOT densify again → avoid vertex explosion
      // Light smoothing only, then cap vertices if still too high
      let working = latlngs.slice();
      const applySmooth = true;
      const smoothIterations = 1; // gentle smoothing
      if (applySmooth) {
        working = smoothLineChaikin(working, smoothIterations);
      }

      if (working.length > MAX_EDIT_VERTICES) {
        const sampled = [];
        const step = (working.length - 1) / (MAX_EDIT_VERTICES - 1);
        for (let i = 0; i < MAX_EDIT_VERTICES; i++) {
          const idx = Math.round(i * step);
          sampled.push(working[idx]);
        }
        working = sampled;
      }

      finalLL = working;
    }

    poly.setLatLngs(finalLL);
    poly.bringToFront();

    rowInfo.geometry = {
      type: 'LineString',
      coordinates: finalLL.map(ll => [ll.lng, ll.lat])
    };

    updateMeasureTooltipText(measurePolylineMeters(finalLL));

    window._map.off('click', onClick);
    window._map.off('mousemove', onMove);
    ctx.active = false;
    restoreFns.forEach(fn => fn());
    toggleDrawingUI(false);
    removeMeasureTooltip();
    hideManualBanner();
    setManualCursor(false);
    if (ctx._detachHotkeys) ctx._detachHotkeys();
  };

  attachManualDrawHotkeys(ctx);
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

// ============================================================================
// Manual Drawing — Cursor helpers
// ============================================================================

//11/05: toggle crosshair cursor on the Leaflet map container
function setManualCursor(on = true){
    const map = window._map;
    const el = map && map.getContainer ? map.getContainer() : null;
    if (!el) return; 
    if (on){
        if (!el.dataset.prevCursor) el.dataset.prevCursor = el.style.cursor || '';  
        el.style.cursor = 'crosshair';
    } else {
        el.style.cursor = el.dataset.prevCursor || '';
    }
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
