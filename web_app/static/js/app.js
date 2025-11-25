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
    formData.append('crossfade', document.getElementById('crossfade').value);
    formData.append('bitrate', document.getElementById('bitrate').value);
    formData.append('preset', document.getElementById('preset').value);

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

// Crossfade slider
document.getElementById('crossfade').addEventListener('input', (e) => {
    document.getElementById('crossfade-value').textContent = `${e.target.value}s`;
});
