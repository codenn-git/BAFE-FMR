// JavaScript logic for FMR GUI
// July 9: tick box; localhost; image selection

const selectedIds = new Set();
const selectedImagePaths = {};
const geoLayers = {};
let currentMatchingImages = {};

function updateFMRList() {
    const ul = document.getElementById("fmr-list");
    ul.innerHTML = "";
    selectedIds.forEach(id => {
        const li = document.createElement("li");
        li.textContent = "FMR-" + id;
        ul.appendChild(li);
        const layer = geoLayers["geoLayer_" + id];
        if (layer) layer.setStyle({color: "red", weight: 3.5});
    });
    
    updateDisplayButtonState();
}

function selectFMR(fmr_id) {
    fetch("http://localhost:5000/select", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fmr_id })
    })
    .then(res => res.json())
    .then(data => {
        if (data.status !== "selected") {
            alert("Already selected.");
            throw new Error("Already selected");
        }

        selectedIds.add(fmr_id);
        updateFMRList();

        return fetch("http://localhost:5000/get_fmr_metadata", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ fmr_id })
        });
    })
    .then(res => res.json())
    .then(meta => {
        if (meta.status !== "success") {
            throw new Error("Metadata fetch failed");
        }

        return fetch("http://localhost:5000/get_matching_images", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ fmr_id })
        }).then(res => res.json()).then(imageData => {
            currentMatchingImages[fmr_id] = imageData.images || [];

            const layer = geoLayers["geoLayer_" + fmr_id];
            if (!layer) return;

            // Track selected image paths for this FMR
            if (!selectedImagePaths[fmr_id]) {
                selectedImagePaths[fmr_id] = [];
            }

            let popupHtml = `
            <div style="word-wrap: break-word; max-width: 350px;">
                <b>FMR ID:</b> ${fmr_id}<br>
                <b>FMR Name:</b> ${meta.name}<br>
                <b>Barangay:</b> ${meta.barangay}<br>
                <b>Municipality:</b> ${meta.municipality}<br>
                <b>Province:</b> ${meta.province}<br><br>
            `;

            if (imageData.status === "success" && imageData.images.length > 0) {
                popupHtml += `<b>Select BSG Image(s):</b><br><div id="checkbox-container-${fmr_id}">`;

                imageData.images.forEach((img, i) => {
                    let dateText = "";
                    const match = img.filename.match(/BSG-\d{3}-(\d{8})-(\d{6})/);
                    if (match) {
                        const rawDate = match[1];
                        const formattedDate = `${rawDate.slice(0, 4)}-${rawDate.slice(4, 6)}-${rawDate.slice(6)}`;
                        dateText = `<span style='font-size: 0.8em; color: #555;'>Date: ${formattedDate}</span>`;
                    }

                    const imagePath = img.path.replace(/\\/g, "/");
                    const isChecked = selectedImagePaths[fmr_id]?.some(
                        p => p.replace(/\\/g, "/") === imagePath
                    ) ? "checked" : "";

                    popupHtml += `
                        <div class="image-option" style="margin-bottom: 6px;">
                            <input type="checkbox" id="checkbox-${fmr_id}-${i}"
                                class="image-checkbox"
                                data-fmr-id="${fmr_id}"
                                data-image-path="${imagePath}"
                                ${isChecked}>
                            <label for="checkbox-${fmr_id}-${i}">${img.filename}</label><br>
                            ${dateText}
                        </div>
                    `;
                });

                popupHtml += `</div><br>`;
            } else {
                popupHtml += `<b>BSG Images:</b> No matching images found.<br><br>`;
            }

            popupHtml += `
                <div style="display: flex; gap: 8px;">
                    <button onclick="selectFMR(${fmr_id})">Select FMR</button>
                    <button onclick="deselectFMR(${fmr_id})">Deselect FMR</button>
                </div>
            </div>`;

            layer.bindPopup(popupHtml).openPopup();

            setTimeout(() => {
                document.querySelectorAll(`#checkbox-container-${fmr_id} input[type='checkbox']`)
                    .forEach(cb => cb.disabled = false);
            }, 100);

            updateDisplayButtonState();
        });
    })
    .catch(err => {
        console.error("Error in selectFMR:", err);
        updateDisplayButtonState();
    });
}

function deselectFMR(fmr_id) {
    fetch("http://localhost:5000/deselect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fmr_id: fmr_id })
    }).then(res => res.json())
    .then(data => {
        if (data.status === "deselected") {
            selectedIds.delete(fmr_id);
            delete currentMatchingImages[fmr_id];
            updateDisplayButtonState();
            updateFMRList();

            // Reset layer color
            const layer = geoLayers["geoLayer_" + fmr_id];
            if (layer) {
                layer.setStyle({ color: "yellow", weight: 3.5 });

                // Disable checkboxes inside this layer's popup
                const popup = layer.getPopup();
                if (popup) {
                    const content = popup.getContent();
                    const parser = new DOMParser();
                    const doc = parser.parseFromString(content, "text/html");

                    doc.querySelectorAll(`input[type='checkbox']`).forEach(cb => {
                        cb.disabled = true;
                    });

                    popup.setContent(doc.body.innerHTML);
                }
            }
        } else {
            alert("FMR not selected or error occurred.");
        }
    });
}

function displaySelectedImages() {
    if (selectedIds.size === 0) {
        alert("No FMRs selected.");
        return;
    }

    const btn = document.getElementById("displayImagesBtn");
    btn.disabled = true;
    btn.textContent = "Loading...";

    document.querySelectorAll(".leaflet-image-layer").forEach(el => el.remove());

    let pending = 0;

    selectedIds.forEach(fmr_id => {
        const checkboxes = document.querySelectorAll(`#checkbox-container-${fmr_id} input[type='checkbox']:checked`);
        if (!checkboxes.length) return;

        checkboxes.forEach(cb => {
            const imagePath = cb.dataset.imagePath;
            pending++;

            fetch("http://localhost:5000/display_image", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ fmr_id, image_path: imagePath })
            })
            .then(res => res.json())
            .then(data => {
                if (data.status === "success") {
                    const overlay = L.imageOverlay(
                        `data:image/png;base64,${data.image_data}`,
                        data.bounds,
                        { opacity: 1.0 }
                    ).addTo(window._map);
                } else {
                    console.error(`Image error: ${data.message}`);
                }
            })
            .catch(err => {
                console.error("Display error:", err);
            })
            .finally(() => {
                pending--;
                if (pending === 0) {
                    btn.disabled = false;
                    btn.textContent = "Display Images";
                }
            });
        });
    });
}

function runProcessing(fmr_id) {
    const images = currentMatchingImages[fmr_id];
    if (!images || images.length === 0) {
        alert("No matching images found for this FMR");
        return;
    }
    const trackRadio = document.getElementById(`track-${fmr_id}`);
    const extractRadio = document.getElementById(`extract-${fmr_id}`);

    let workflowType, workflowOptions = {};
    if (trackRadio.checked) {
        workflowType = 'track';
        workflowOptions.mode = document.getElementById(`auto-${fmr_id}`).checked ? 'automatic' : 'manual';
    } else if (extractRadio.checked) {
        workflowType = 'extract';
        workflowOptions.image_type = document.getElementById(`bsg-${fmr_id}`).checked ? 'BSG' : 'PNEO';
    } else {
        alert("Please select a processing workflow");
        return;
    }

    const imagePath = images[0].path;
    const resultsDiv = document.getElementById(`results-${fmr_id}`);
    resultsDiv.innerHTML = '<div>Processing... Please wait.</div>';
    resultsDiv.style.display = 'block';

    fetch("http://localhost:5000/process_fmr", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            fmr_id: fmr_id,
            image_path: imagePath,
            workflow_type: workflowType,
            workflow_options: workflowOptions
        })
    }).then(res => res.json())
    .then(data => {
        if (data.status === "success") {
            let resultHtml = `<div><b>Processing Results:</b><br>`;
            resultHtml += `<b>Status:</b> ${data.message}<br>`;
            if (data.length) resultHtml += `<b>Length:</b> ${data.length}<br>`;
            if (data.progress) resultHtml += `<b>Progress:</b> ${data.progress}<br>`;
            if (data.mean_width) resultHtml += `<b>Mean Width:</b> ${data.mean_width}<br>`;
            resultHtml += `</div>`;
            resultsDiv.innerHTML = resultHtml;
        } else {
            resultsDiv.innerHTML = `<div style="color: red;"><b>Error:</b> ${data.message}</div>`;
        }
    }).catch(err => {
        resultsDiv.innerHTML = `<div style="color: red;"><b>Error:</b> ${err.message}</div>`;
    });
}

function clearSelections() {
    fetch("http://localhost:5000/clear", {method: "POST"})
    .then(res => res.json())
    .then(data => {
        if (data.status === "cleared") {
            selectedIds.clear();
            
            Object.keys(currentMatchingImages).forEach(key => delete currentMatchingImages[key]);
            updateDisplayButtonState();

            updateFMRList();
            Object.values(geoLayers).forEach(layer => {
                layer.setStyle({color: "yellow", weight: 3.5});
            });
            document.querySelectorAll('[id^="processing-buttons-"]').forEach(el => el.style.display = 'none');
            document.querySelectorAll('[id^="image-display-"]').forEach(el => el.style.display = 'none');
            document.querySelectorAll('[id^="results-"]').forEach(el => el.style.display = 'none');
        }
    });
}

function filterByProvince() {
    const province = document.getElementById("provinceSelect").value;
    fetch("http://localhost:5000/filter", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({province: province})
    }).then(res => res.json())
    .then(data => {
        if (data.status === "filtered") {
            alert(`Filtered to ${data.count} FMRs`);
            location.reload();
        }
    });
}

function downloadSelected() {
    if (selectedIds.size === 0) {
        alert("No FMRs selected");
        return;
    }
    fetch("http://localhost:5000/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            selected_ids: Array.from(selectedIds)
        })
    })
    .then(async response => {
        if (!response.ok) {
            let msg = "Download failed";
            try {
                const data = await response.json();
                if (data && data.message) msg = data.message;
            } catch (e) {}
            throw new Error(msg);
        }
        return response.blob();
    })
    .then(blob => {
        // Try to get filename from Content-Disposition header if possible
        let filename = "exported_fmr.zip";
        // The fetch API does not expose headers in .then(blob), so fallback to default name
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
    })
    .catch(err => {
        alert(err.message);
    });
}

function updateFMRs() {
    if (confirm("Update FMR data? This may take a while.")) {
        fetch("http://localhost:5000/update_fmr", {method: "POST"})
        .then(res => res.json())
        .then(data => {
            if (data.status === "success") {
                alert("FMR data updated successfully!");
                location.reload();
            } else {
                alert(`Error updating FMR: ${data.message}`);
            }
        });
    }
}

function displaySelectedImages() {
    selectedIds.forEach(fmrId => {
        const checkboxes = document.querySelectorAll(`#checkbox-container-${fmrId} input[type='checkbox']:checked`);
        if (checkboxes.length === 0) return;

        checkboxes.forEach(cb => {
            const imagePath = cb.dataset.imagePath;

            fetch("http://127.0.0.1:5000/display_selected_image", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ fmr_id: fmrId, image_path: imagePath })
            })
            .then(res => res.json())
            .then(data => {
                if (data.status !== "success") {
                    console.error("Failed to load image overlay:", data.message);
                    return;
                }

                const overlay = L.imageOverlay(
                    `data:image/png;base64,${data.image_base64}`,
                    data.image_bounds,
                    { opacity: 1.0 }
                ).addTo(window._map);

                // Optional: store overlay reference if you want to clear later
                if (!window._overlays) window._overlays = {};
                window._overlays[`${fmrId}_${imagePath}`] = overlay;
            })
            .catch(err => {
                console.error("Error fetching image preview:", err);
            });
        });
    });
}


function updateDisplayButtonState() {
    const btn = document.getElementById("displayImagesBtn");
    
    if (!btn) {
        console.error("Display button not found!");
        return;
    }

    const hasImages = Array.from(selectedIds).some(fmr_id =>
        currentMatchingImages[fmr_id] && currentMatchingImages[fmr_id].length > 0
    );

    console.log("Updating display button state:", {
        selectedIds: Array.from(selectedIds),
        hasImages: hasImages,
        currentMatchingImages: currentMatchingImages
    });

    btn.disabled = !hasImages;
    
    // Update button text to provide feedback
    if (selectedIds.size === 0) {
        btn.textContent = "Display Images";
    } else if (hasImages) {
        btn.textContent = "Display Images";
    } else {
        btn.textContent = "No Images Available";
    }
}