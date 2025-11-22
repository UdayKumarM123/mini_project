// DOM Elements (grab the elements that live outside uploadBox)
const uploadBox = document.getElementById('uploadBox');
const imageInput = document.getElementById('imageInput');
const classifyBtn = document.getElementById('classifyBtn');
const resultsSection = document.getElementById('resultsSection');
const previewImage = document.getElementById('previewImage');
const predictedClass = document.getElementById('predictedClass');
const confidenceValue = document.getElementById('confidenceValue');
const confidenceBadge = document.getElementById('confidenceBadge');
const probabilityBars = document.getElementById('probabilityBars');
const errorMessage = document.getElementById('errorMessage');
const errorText = document.getElementById('errorText');

// Helper to safely get the button text/spinner elements (in case DOM was altered)
function getBtnParts() {
    return {
        btnText: document.getElementById('btnText'),
        btnSpinner: document.getElementById('btnSpinner')
    };
}

const API_URL = 'http://localhost:5000';
let selectedFile = null;

// --- Upload box handlers ---
uploadBox.addEventListener('click', () => imageInput.click());

imageInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleFileSelect(file);
});

uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.classList.add('dragover');
});
uploadBox.addEventListener('dragleave', () => uploadBox.classList.remove('dragover'));
uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) handleFileSelect(file);
    else showError('Please upload a valid image file (PNG, JPG, or JPEG)');
});

function handleFileSelect(file) {
    // Validate file size
    if (file.size > 10 * 1024 * 1024) {
        showError('File size exceeds 10MB. Please upload a smaller image.');
        return;
    }

    selectedFile = file;

    // Update upload box UI (this replaces uploadBox innerHTML — safe)
    uploadBox.innerHTML = `
        <svg class="upload-icon" style="width: 60px; height: 60px; color: #48bb78;" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
            <polyline points="22 4 12 14.01 9 11.01"></polyline>
        </svg>
        <p class="upload-text" style="color: #48bb78;">✓ Image Selected</p>
        <p class="upload-hint">${file.name}</p>
    `;

    // IMPORTANT: reset spinner UI parts (in case they were toggled)
    const { btnText, btnSpinner } = getBtnParts();
    if (btnText) btnText.style.display = 'inline-block';
    if (btnSpinner) {
        btnSpinner.style.display = 'none';
        // ensure spinner animation not visible
        btnSpinner.hidden = true;
    }

    classifyBtn.disabled = false;
    resultsSection.hidden = true;
    // hide any old error
    if (errorMessage) errorMessage.hidden = true;
}

// --- Set loading state robustly ---
function setLoadingState(loading) {
    // disable/enable button
    classifyBtn.disabled = !!loading;

    // re-query parts each time to avoid stale references
    const { btnText, btnSpinner } = getBtnParts();

    if (btnText) {
        btnText.style.display = loading ? 'none' : 'inline-block';
    }
    if (btnSpinner) {
        btnSpinner.style.display = loading ? 'inline-block' : 'none';
        // keep hidden attribute in sync for older code that checks it
        btnSpinner.hidden = !loading;
    }
}

// --- Classification click handler (robust try/catch/finally) ---
classifyBtn.addEventListener('click', async () => {
    if (!selectedFile) {
        showError('Please select an image first');
        return;
    }

    // start spinner immediately (do not rely solely on setLoadingState, do both)
    setLoadingState(true);

    // hide previous UI states
    resultsSection.hidden = true;
    if (errorMessage) errorMessage.hidden = true;

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
        const resp = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            body: formData
        });

        // handle non-200 status explicitly
        if (!resp.ok) {
            const text = await resp.text().catch(() => '');
            throw new Error(`Server error: ${resp.status} ${text}`);
        }

        // parse JSON (guard for invalid JSON)
        let data;
        try {
            data = await resp.json();
        } catch (jsonErr) {
            throw new Error('Invalid JSON response from server');
        }

        if (!data || data.success === false) {
            const msg = data && data.error ? data.error : 'Classification failed';
            throw new Error(msg);
        }

        // Data OK — display results
        displayResults(data);

    } catch (err) {
        console.error('Classification error:', err);
        showError(typeof err === 'string' ? err : (err.message || 'An error occurred'));
    } finally {
        // ALWAYS stop spinner — defensive: small timeout so UI can update
        // but still guaranteed to run
        setTimeout(() => setLoadingState(false), 100);
    }
});

// --- Display results ---
function displayResults(data) {
    // data.image expected to be "data:image/..." already; if it's a filepath, adapt
    if (data.image && String(data.image).startsWith('data:')) {
        previewImage.src = data.image;
    } else if (data.image && typeof data.image === 'string' && data.image.length > 0) {
        // attempt to use as URL or local path
        previewImage.src = data.image;
    } else {
        previewImage.src = ''; // blank if absent
    }

    predictedClass.textContent = data.predicted_class || '-';
    confidenceValue.textContent = (Math.round((data.confidence || 0) * 100) / 100) + '%';

    // confidence is already percentage in your backend; handle both cases
    // (if it's e.g., 87.12 or 0.8712)
    let conf = data.confidence;
    if (typeof conf === 'number') {
        // assume if <=1 then it's fraction, convert to percent
        if (conf <= 1) conf = conf * 100;
        // else already percentage
    } else {
        conf = 0;
    }
    confidenceValue.textContent = `${Math.round(conf * 100) / 100}%`;

    // color badge
    if (conf >= 80) {
        confidenceBadge.style.background = 'rgba(72, 187, 120, 0.3)';
    } else if (conf >= 60) {
        confidenceBadge.style.background = 'rgba(237, 137, 54, 0.3)';
    } else {
        confidenceBadge.style.background = 'rgba(245, 101, 101, 0.3)';
    }

    // render probabilities — handle both list or object shapes
    probabilityBars.innerHTML = '';
    if (Array.isArray(data.all_probabilities)) {
        // expecting [{class: 'x', confidence: v}, ...]
        for (const p of data.all_probabilities) {
            appendProbabilityItem(p.class || p.name, p.confidence || p.value || 0);
        }
    } else if (typeof data.all_probabilities === 'object' && data.all_probabilities !== null) {
        // object map
        // if it's already sorted list on backend, preserve order by iterating keys
        for (const [className, probability] of Object.entries(data.all_probabilities)) {
            appendProbabilityItem(className, probability);
        }
    }

    resultsSection.hidden = false;
}

function appendProbabilityItem(className, probability) {
    const item = document.createElement('div');
    item.className = 'probability-item';
    // ensure probability is a number 0-100
    let prob = Number(probability);
    if (!isFinite(prob)) prob = 0;
    // if value seems fractional (<=1), convert to percentage
    if (prob <= 1) prob = prob * 100;

    prob = Math.round(prob * 100) / 100; // two decimals
    item.innerHTML = `
        <div class="probability-label">
            <span class="class-name">${className}</span>
            <span class="probability-value">${prob}%</span>
        </div>
        <div class="probability-bar">
            <div class="probability-fill" style="width: ${prob}%"></div>
        </div>
    `;
    probabilityBars.appendChild(item);
}

// --- Error handling ---
// We removed the big red error box UI earlier, but keep this to log if needed.
function showError(message) {
    console.warn('UI error (suppressed):', message);
    // If you later want to re-enable the visible error box, uncomment below:
    // errorText.textContent = message;
    // errorMessage.hidden = false;
}

// --- Health check disabled (Fix 2) ---
async function checkAPIHealth() {
    // intentionally disabled to avoid showing error box on load
    return;
}

// Initialize (health check disabled)
checkAPIHealth();
