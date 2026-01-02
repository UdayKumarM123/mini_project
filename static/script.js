// DOM Elements
const uploadBox = document.getElementById('uploadBox');
const imageInput = document.getElementById('imageInput');
const classifyBtn = document.getElementById('classifyBtn');
const btnText = document.getElementById('btnText');
const btnSpinner = document.getElementById('btnSpinner');
const resultsSection = document.getElementById('resultsSection');
const previewImage = document.getElementById('previewImage');
const predictedClass = document.getElementById('predictedClass');
const confidenceValue = document.getElementById('confidenceValue');
const confidenceBadge = document.getElementById('confidenceBadge');
const probabilityBars = document.getElementById('probabilityBars');

console.log('btnSpinner:', btnSpinner);

const API_URL = 'http://localhost:5000';
let selectedFile = null;

uploadBox.addEventListener('click', () => {
    imageInput.click();
});

imageInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleFileSelect(file);
});

uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.classList.add('dragover');
});
uploadBox.addEventListener('dragleave', () => {
    uploadBox.classList.remove('dragover');
});
uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) handleFileSelect(file);
});

function handleFileSelect(file) {
    selectedFile = file;
    uploadBox.innerHTML = `
        <svg class="upload-icon" style="width: 60px; height: 60px; color: #48bb78;" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
            <polyline points="22 4 12 14.01 9 11.01"></polyline>
        </svg>
        <p class="upload-text" style="color: #48bb78;">✓ Image Selected</p>
        <p class="upload-hint">${file.name}</p>
    `;
    classifyBtn.disabled = false;
    resultsSection.hidden = true;
}

classifyBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    setLoadingState(true);
    resultsSection.hidden = true;
    try {
        const formData = new FormData();
        formData.append('image', selectedFile);
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if (data.success) displayResults(data);
    } catch (error) {
        console.error('Error:', error);
    } finally {
        setLoadingState(false);
        reallyHideSpinner();
    }
});

function displayResults(data) {
    previewImage.src = data.image;
    const formattedClass = data.predicted_class.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    predictedClass.textContent = formattedClass;
    confidenceValue.textContent = `${data.confidence}%`;

    if (data.confidence >= 80) {
        confidenceBadge.style.background = 'rgba(72, 187, 120, 0.3)';
    } else if (data.confidence >= 60) {
        confidenceBadge.style.background = 'rgba(237, 137, 54, 0.3)';
    } else {
        confidenceBadge.style.background = 'rgba(245, 101, 101, 0.3)';
    }

    if (data.road_info) {
        const roadInfo = data.road_info;
        document.getElementById('roadDescription').textContent = roadInfo.description || 'No description available';
        document.getElementById('roadDescription').style.fontWeight = 'bold';
        document.getElementById('crrRange').textContent = roadInfo.crr_range || 'N/A';
        document.getElementById('meanCrr').textContent = roadInfo.mean_crr || '0';
        document.getElementById('energyAvailable').textContent = `${roadInfo.energy_available || 0}%`;
        document.getElementById('energyReduced').textContent = `${roadInfo.energy_reduced || 0}%`;
    }

    probabilityBars.innerHTML = '';
    for (const [className, probability] of Object.entries(data.all_probabilities)) {
        const formattedName = className.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
        const item = document.createElement('div');
        item.className = 'probability-item';
        item.innerHTML = `
            <div class="probability-label">
                <span class="class-name">${formattedName}</span>
                <span class="probability-value">${probability}%</span>
            </div>
            <div class="probability-bar">
                <div class="probability-fill" style="width: ${probability}%"></div>
            </div>
        `;
        probabilityBars.appendChild(item);
    }
    resultsSection.hidden = false;
}

function setLoadingState(loading) {
    classifyBtn.disabled = loading;
    btnText.hidden = loading;
    btnSpinner.hidden = !loading;
    if (!loading) {
        btnSpinner.style.display = "none";
    } else {
        btnSpinner.style.display = "inline-block";
    }
    console.log("setLoadingState: loading=", loading, "| btnSpinner.hidden=", btnSpinner.hidden, "| btnSpinner.style.display=", btnSpinner.style.display);
}

function reallyHideSpinner() {
    btnSpinner.hidden = true;
    btnSpinner.style.display = "none";
    console.log("reallyHideSpinner called");
}