/**
 * MedGPT — Frontend Application Logic
 * Handles image upload, question submission, and result display.
 */

document.addEventListener('DOMContentLoaded', () => {
    // ---------- DOM Elements ----------
    const uploadZone = document.getElementById('uploadZone');
    const uploadContent = document.getElementById('uploadContent');
    const uploadPreview = document.getElementById('uploadPreview');
    const previewImage = document.getElementById('previewImage');
    const removeImageBtn = document.getElementById('removeImage');
    const imageInput = document.getElementById('imageInput');
    const questionInput = document.getElementById('questionInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resultsSection = document.getElementById('resultsSection');
    const answerText = document.getElementById('answerText');
    const processingTime = document.getElementById('processingTime');
    const resultOriginal = document.getElementById('resultOriginal');
    const resultHeatmap = document.getElementById('resultHeatmap');
    const modelStatus = document.getElementById('modelStatus');
    const errorToast = document.getElementById('errorToast');
    const errorMessage = document.getElementById('errorMessage');
    const errorClose = document.getElementById('errorClose');

    let selectedFile = null;

    // ---------- Model Status Check ----------
    async function checkModelStatus() {
        const statusDot = modelStatus.querySelector('.status-dot');
        const statusText = modelStatus.querySelector('.status-text');

        try {
            const res = await fetch('/health');
            const data = await res.json();

            if (data.model_loaded) {
                statusDot.className = 'status-dot online';
                statusText.textContent = 'Model ready';
            } else {
                statusDot.className = 'status-dot';
                statusText.textContent = 'Model loading...';
                // Retry in 10s
                setTimeout(checkModelStatus, 10000);
            }
        } catch (e) {
            statusDot.className = 'status-dot offline';
            statusText.textContent = 'Server offline';
            // Retry in 15s
            setTimeout(checkModelStatus, 15000);
        }
    }

    checkModelStatus();

    // ---------- Image Upload ----------
    function handleFile(file) {
        if (!file || !file.type.startsWith('image/')) {
            showError('Please upload a valid image file.');
            return;
        }

        if (file.size > 50 * 1024 * 1024) {
            showError('Image file too large. Maximum size is 50MB.');
            return;
        }

        selectedFile = file;

        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            uploadContent.style.display = 'none';
            uploadPreview.style.display = 'flex';
            updateAnalyzeButton();
        };
        reader.readAsDataURL(file);
    }

    // Click to upload
    uploadZone.addEventListener('click', (e) => {
        if (e.target === removeImageBtn || e.target.closest('.btn-remove')) return;
        imageInput.click();
    });

    imageInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag and drop
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('drag-over');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('drag-over');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // Remove image
    removeImageBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        selectedFile = null;
        imageInput.value = '';
        uploadContent.style.display = 'flex';
        uploadPreview.style.display = 'none';
        updateAnalyzeButton();
    });

    // ---------- Question Input ----------
    questionInput.addEventListener('input', updateAnalyzeButton);

    // Suggestion chips
    document.querySelectorAll('.suggestion-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            questionInput.value = chip.dataset.q;
            questionInput.focus();
            updateAnalyzeButton();
        });
    });

    function updateAnalyzeButton() {
        analyzeBtn.disabled = !(selectedFile && questionInput.value.trim());
    }

    // ---------- Analysis ----------
    analyzeBtn.addEventListener('click', analyze);

    questionInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!analyzeBtn.disabled) analyze();
        }
    });

    async function analyze() {
        if (!selectedFile || !questionInput.value.trim()) return;

        // Show loading state
        const btnText = analyzeBtn.querySelector('.btn-text');
        const btnLoader = analyzeBtn.querySelector('.btn-loader');
        const btnArrow = analyzeBtn.querySelector('.btn-arrow');

        btnText.textContent = 'Analyzing...';
        btnLoader.style.display = 'inline-flex';
        btnArrow.style.display = 'none';
        analyzeBtn.disabled = true;

        // Hide previous results
        resultsSection.style.display = 'none';
        hideError();

        try {
            const formData = new FormData();
            formData.append('image', selectedFile);
            formData.append('question', questionInput.value.trim());
            formData.append('generate_heatmap', 'true');

            const response = await fetch('/predict', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.detail || `Server error: ${response.status}`);
            }

            const data = await response.json();

            // Display results
            answerText.textContent = data.answer;
            processingTime.textContent = `${data.processing_time}s`;

            // Original image
            if (data.original_image) {
                resultOriginal.src = `data:image/png;base64,${data.original_image}`;
            }

            // Heatmap
            if (data.heatmap) {
                resultHeatmap.src = `data:image/png;base64,${data.heatmap}`;
                document.querySelector('.visual-card').style.display = 'block';
            } else {
                document.querySelector('.visual-card').style.display = 'none';
            }

            resultsSection.style.display = 'flex';
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

        } catch (error) {
            showError(error.message || 'An unexpected error occurred. Please try again.');
        } finally {
            btnText.textContent = 'Analyze';
            btnLoader.style.display = 'none';
            btnArrow.style.display = 'inline-flex';
            updateAnalyzeButton();
        }
    }

    // ---------- Error Handling ----------
    function showError(message) {
        errorMessage.textContent = message;
        errorToast.style.display = 'flex';
        setTimeout(hideError, 8000);
    }

    function hideError() {
        errorToast.style.display = 'none';
    }

    errorClose.addEventListener('click', hideError);
});
