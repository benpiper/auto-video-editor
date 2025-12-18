let currentJobId = null;
let selectedFile = null;
let eventSource = null;

// DOM elements
const uploadArea = document.getElementById('upload-area');
const videoInput = document.getElementById('video-input');
const uploadSection = document.getElementById('upload-section');
const paramsSection = document.getElementById('params-section');
const progressSection = document.getElementById('progress-section');
const downloadSection = document.getElementById('download-section');
const processBtn = document.getElementById('process-btn');
const progressFill = document.getElementById('progress-fill');
const progressMessage = document.getElementById('progress-message');
const downloadBtn = document.getElementById('download-btn');
const newVideoBtn = document.getElementById('new-video-btn');

// File upload handling
uploadArea.addEventListener('click', () => videoInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    handleFile(e.dataTransfer.files[0]);
});

videoInput.addEventListener('change', (e) => {
    handleFile(e.target.files[0]);
});

function handleFile(file) {
    if (!file || !file.type.startsWith('video/')) {
        alert('Please select a valid video file');
        return;
    }

    selectedFile = file;
    const sizeMB = (file.size / 1024 / 1024).toFixed(2);

    document.getElementById('file-info').innerHTML = `
        <strong>Selected:</strong> ${file.name} (${sizeMB} MB)
    `;
    document.getElementById('file-info').classList.remove('hidden');
    paramsSection.classList.remove('hidden');
}

// Process video
processBtn.addEventListener('click', async () => {
    if (!selectedFile) {
        alert('Please select a video file first');
        return;
    }

    processBtn.disabled = true;
    processBtn.textContent = 'Uploading...';

    const formData = new FormData();
    formData.append('video', selectedFile);
    formData.append('min_silence', document.getElementById('min-silence').value);
    formData.append('silence_thresh', document.getElementById('silence-thresh').value);
    formData.append('remove_silence', document.getElementById('remove-silence').checked);
    formData.append('filler_words', document.getElementById('filler-words').value);
    formData.append('remove_freeze', document.getElementById('remove-freeze').checked);
    formData.append('freeze_duration', document.getElementById('freeze-duration').value);

    // Background removal parameters
    const removeBg = document.getElementById('remove-background').checked;
    formData.append('remove_background', removeBg);

    if (removeBg) {
        const bgMode = document.getElementById('bg-mode').value;
        const bgMethod = document.querySelector('input[name="bg-method"]:checked').value;

        formData.append('rvm_model', document.getElementById('rvm-model').value);

        // Background type
        if (bgMode === 'color') {
            formData.append('bg_color', document.getElementById('bg-color').value);
        } else if (bgMode === 'transparent') {
            formData.append('bg_color', 'transparent');
        } else if (bgMode === 'image') {
            const bgImageFile = document.getElementById('bg-image').files[0];
            if (bgImageFile) {
                formData.append('bg_image', bgImageFile);
            } else {
                alert('Please select a background image');
                processBtn.disabled = false;
                processBtn.textContent = 'Start Processing';
                return;
            }
        }

        // Segmentation or RVM
        if (bgMethod === 'segmentation') {
            formData.append('use_segmentation', 'true');
            formData.append('seg_model', document.getElementById('seg-model').value);
            formData.append('seg_threshold', document.getElementById('seg-threshold').value);
            formData.append('seg_smooth', document.getElementById('seg-smooth').value);
        } else {
            // RVM with morphological cleanup
            formData.append('rvm_erode', document.getElementById('rvm-erode').value);
            formData.append('rvm_dilate', document.getElementById('rvm-dilate').value);
            formData.append('rvm_median', document.getElementById('rvm-median').value);
            formData.append('rvm_blur', document.getElementById('rvm-blur').value);
        }
    }

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Upload failed');
        }

        const data = await response.json();
        currentJobId = data.job_id;

        // Show progress section
        uploadSection.classList.add('hidden');
        paramsSection.classList.add('hidden');
        progressSection.classList.remove('hidden');

        // Connect to progress stream
        connectProgressStream();

    } catch (error) {
        alert('Error uploading video: ' + error.message);
        processBtn.disabled = false;
        processBtn.textContent = 'Start Processing';
    }
});

function connectProgressStream() {
    eventSource = new EventSource(`/progress/${currentJobId}`);

    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.error) {
            alert('Error: ' + data.error);
            eventSource.close();
            return;
        }

        updateProgress(data.progress, data.message);

        if (data.status === 'complete') {
            eventSource.close();
            setTimeout(showDownload, 500);
        } else if (data.status === 'error') {
            eventSource.close();
            alert('Processing failed: ' + data.message);
            setTimeout(() => location.reload(), 2000);
        }
    };

    eventSource.onerror = (error) => {
        console.error('EventSource error:', error);
        eventSource.close();
    };
}

function updateProgress(progress, message) {
    progressFill.style.width = `${progress}%`;
    progressMessage.textContent = message;
}

function showDownload() {
    progressSection.classList.add('hidden');
    downloadSection.classList.remove('hidden');
}

// Download button
downloadBtn.addEventListener('click', () => {
    window.location.href = `/download/${currentJobId}`;
});

// New video button
newVideoBtn.addEventListener('click', () => {
    location.reload();
});

// Background removal UI toggles
document.getElementById('remove-background').addEventListener('change', (e) => {
    const bgOptions = document.getElementById('bg-options');
    const rvmAdvanced = document.getElementById('rvm-advanced');
    if (e.target.checked) {
        bgOptions.style.display = 'flex';
        rvmAdvanced.style.display = 'block';
    } else {
        bgOptions.style.display = 'none';
        rvmAdvanced.style.display = 'none';
    }
});

// Silence detection UI toggle
document.getElementById('remove-silence').addEventListener('change', (e) => {
    const silenceOptions = document.getElementById('silence-options');
    if (e.target.checked) {
        silenceOptions.style.display = 'flex';
    } else {
        silenceOptions.style.display = 'none';
    }
});

document.getElementById('bg-mode').addEventListener('change', (e) => {
    const colorGroup = document.getElementById('bg-color-group');
    const imageGroup = document.getElementById('bg-image-group');

    if (e.target.value === 'color') {
        colorGroup.classList.remove('hidden');
        imageGroup.classList.add('hidden');
    } else if (e.target.value === 'image') {
        colorGroup.classList.add('hidden');
        imageGroup.classList.remove('hidden');
    } else { // transparent
        colorGroup.classList.add('hidden');
        imageGroup.classList.add('hidden');
    }
});

// Morphological cleanup preset handler
document.getElementById('morph-preset').addEventListener('change', (e) => {
    const presets = {
        none: [0, 0, 0, 0],
        light: [3, 5, 5, 0],
        aggressive: [5, 7, 7, 0]
    };

    const preset = presets[e.target.value] || [0, 0, 0, 0];
    document.getElementById('rvm-erode').value = preset[0];
    document.getElementById('rvm-dilate').value = preset[1];
    document.getElementById('rvm-median').value = preset[2];
    document.getElementById('rvm-blur').value = preset[3];

    // Update displayed values
    document.getElementById('erode-val').textContent = preset[0];
    document.getElementById('dilate-val').textContent = preset[1];
    document.getElementById('median-val').textContent = preset[2];
    document.getElementById('blur-val').textContent = preset[3];
});

// Update slider value displays
['rvm-erode', 'rvm-dilate', 'rvm-median', 'rvm-blur'].forEach(id => {
    const slider = document.getElementById(id);
    const valueId = id.replace('rvm-', '') + '-val';
    slider.addEventListener('input', (e) => {
        document.getElementById(valueId).textContent = e.target.value;
        // Reset preset to custom when manually changing
        document.getElementById('morph-preset').value = 'none';
    });
});

// Update segmentation slider value displays
document.getElementById('seg-threshold').addEventListener('input', (e) => {
    document.getElementById('threshold-val').textContent = parseFloat(e.target.value).toFixed(2);
});

document.getElementById('seg-smooth').addEventListener('input', (e) => {
    document.getElementById('smooth-val').textContent = e.target.value;
});

// Background method toggle (RVM vs Segmentation)
document.querySelectorAll('input[name="bg-method"]').forEach(radio => {
    radio.addEventListener('change', (e) => {
        const morphSection = document.getElementById('morph-cleanup-section');
        const segSection = document.getElementById('seg-options-section');
        const rvmModel = document.getElementById('rvm-model').parentElement;

        if (e.target.value === 'segmentation') {
            morphSection.style.display = 'none';
            rvmModel.style.display = 'none';
            segSection.style.display = 'block';
        } else {
            morphSection.style.display = 'block';
            rvmModel.style.display = 'block';
            segSection.style.display = 'none';
        }
    });
});
