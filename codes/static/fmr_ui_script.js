// JavaScript logic for FMR GUI

const selectedIds = new Set();
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
    const processButton = document.getElementById("processFMRBtn");
    processButton.disabled = selectedIds.size === 0;
}

function selectFMR(fmr_id) {
    fetch("http://localhost:5000/select", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({fmr_id: fmr_id})
    }).then(res => res.json())
    .then(data => {
        if (data.status === "selected") {
            selectedIds.add(fmr_id);
            updateFMRList();
            injectProcessingPanel(fmr_id);

            fetch("http://localhost:5000/get_matching_images", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({fmr_id: fmr_id})
            }).then(res => res.json())
            .then(imageData => {
                if (imageData.status === "success") {
                    currentMatchingImages[fmr_id] = imageData.matching_images;
                }
            });
        } else alert("Already selected.");
    });
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
            updateFMRList();
            
            // Hide processing elements for this FMR
            const processingButtons = document.getElementById(`processing-buttons-${fmr_id}`);
            const imageDisplay = document.getElementById(`image-display-${fmr_id}`);
            const results = document.getElementById(`results-${fmr_id}`);
            
            if (processingButtons) processingButtons.style.display = 'none';
            if (imageDisplay) imageDisplay.style.display = 'none';
            if (results) results.style.display = 'none';
            
            // Reset layer color to yellow
            const layer = geoLayers["geoLayer_" + fmr_id];
            if (layer) layer.setStyle({color: "yellow", weight: 3.5});
        } else {
            alert("FMR not selected or error occurred.");
        }
    });
}

function displayImage(fmr_id) {
    const images = currentMatchingImages[fmr_id];
    if (!images || images.length === 0) {
        alert("No matching images found for this FMR");
        return;
    }
    const imagePath = images[0].path;
    fetch("http://localhost:5000/display_image", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({fmr_id: fmr_id, image_path: imagePath})
    }).then(res => res.json())
    .then(data => {
        if (data.status === "success") {
            const imageDiv = document.getElementById(`image-display-${fmr_id}`);
            imageDiv.innerHTML = `<img src="data:image/png;base64,${data.image_data}" class="image-preview">`;
            imageDiv.style.display = 'block';
        } else {
            alert(`Error displaying image: ${data.message}`);
        }
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
        alert("No FMRs selected for download");
        return;
    }
    const format = document.getElementById("exportFormat").value;
    const params = new URLSearchParams({format: format});
    window.open(`http://localhost:5000/export?${params.toString()}`);
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

function injectProcessingPanel(fmr_id) {
    const container = document.getElementById("dynamic-processing-panel");
    if (document.getElementById(`processing-buttons-${fmr_id}`)) return;  // Avoid duplicates

    const html = `
    <div id="processing-buttons-${fmr_id}" style="margin-bottom: 15px;">
        <hr>
        <h4>FMR-${fmr_id}</h4>
        <button onclick="displayImage(${fmr_id})">🖼️ Display Image</button>
        <button onclick="showProcessOptions(${fmr_id})">⚙️ Process</button>
        <div id="process-options-${fmr_id}" style="display: none; margin-top: 10px;">
            <div>
                <input type="radio" id="track-${fmr_id}" name="workflow-${fmr_id}" value="track">
                <label for="track-${fmr_id}">Track Progress</label><br>
                <div id="track-options-${fmr_id}" style="display: none; margin-left: 20px;">
                    <input type="radio" id="auto-${fmr_id}" name="track-mode-${fmr_id}" value="automatic" checked>
                    <label for="auto-${fmr_id}">Automatic</label><br>
                    <input type="radio" id="manual-${fmr_id}" name="track-mode-${fmr_id}" value="manual">
                    <label for="manual-${fmr_id}">Manual</label>
                </div>
            </div>
            <div>
                <input type="radio" id="extract-${fmr_id}" name="workflow-${fmr_id}" value="extract">
                <label for="extract-${fmr_id}">Extract Width</label><br>
                <div id="extract-options-${fmr_id}" style="display: none; margin-left: 20px;">
                    <input type="radio" id="bsg-${fmr_id}" name="image-type-${fmr_id}" value="BSG" checked>
                    <label for="bsg-${fmr_id}">BSG Image</label><br>
                    <input type="radio" id="pneo-${fmr_id}" name="image-type-${fmr_id}" value="PNEO">
                    <label for="pneo-${fmr_id}">PNEO Image</label>
                </div>
            </div>
            <button onclick="runProcessing(${fmr_id})" style="margin-top: 10px;">▶️ Run</button>
        </div>
        <div id="image-display-${fmr_id}" style="display: none; margin-top: 10px;"></div>
        <div id="results-${fmr_id}" style="display: none; margin-top: 10px;"></div>
    </div>
    `;

    container.insertAdjacentHTML("beforeend", html);
}
