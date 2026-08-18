document.addEventListener('DOMContentLoaded', () => {
    // Check Health
    checkHardwareStatus();
    
    // Setup Navigation
    setupNavigation();

    // DOM Elements
    const uploadArea = document.getElementById('upload-area');
    const uploadPrompt = document.getElementById('upload-prompt');
    const fileInput = document.getElementById('file-input');
    const previewArea = document.getElementById('preview-area');
    const imagePreview = document.getElementById('image-preview');
    const filenameDisplay = document.getElementById('filename');
    const filesizeDisplay = document.getElementById('filesize');
    
    const analyzeBtn = document.getElementById('analyze-btn');
    const clearBtn = document.getElementById('clear-btn');
    const reuploadBtn = document.getElementById('reupload-btn');
    
    const loadingOverlay = document.getElementById('loading-overlay');
    const resultsContainer = document.getElementById('results-container');
    
    let currentFile = null;

    // Drag and Drop Events
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // Click to upload
    uploadPrompt.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            handleFile(fileInput.files[0]);
        }
    });

    // Clear Button
    clearBtn.addEventListener('click', () => {
        resetUpload();
    });

    function resetUpload() {
        currentFile = null;
        fileInput.value = '';
        uploadArea.classList.remove('hidden');
        previewArea.classList.add('hidden');
        resultsContainer.classList.add('hidden');
        
        // Reset patient inputs
        document.getElementById('patient-name').value = '';
        document.getElementById('patient-details').value = '';
        
        // Reset gradcam panels
        document.getElementById('panel-heatmap').src = '';
        document.getElementById('panel-overlay').src = '';
        const histoPanel = document.getElementById('panel-histo');
        if (histoPanel) histoPanel.classList.add('hidden');
        histoPanel.src = '';
    }

    // Reupload Button
    reuploadBtn.addEventListener('click', () => {
        fileInput.click();
    });

    // Analyze Button
    analyzeBtn.addEventListener('click', () => {
        if (currentFile) {
            uploadAndAnalyze(currentFile);
        }
    });

    function handleFile(file) {
        // Validate file type
        if (!file.type.match('image/jpeg') && !file.type.match('image/png')) {
            alert('Please upload a valid JPG or PNG image.');
            return;
        }

        currentFile = file;
        
        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            filenameDisplay.textContent = file.name;
            filesizeDisplay.textContent = formatBytes(file.size);
            
            uploadPrompt.classList.add('hidden');
            previewArea.classList.remove('hidden');
            resultsContainer.classList.add('hidden');
        };
        reader.readAsDataURL(file);
    }

    async function checkHardwareStatus() {
        const hardwareStatus = document.getElementById('hardware-status');
        try {
            const res = await fetch('http://127.0.0.1:5000/health');
            if (res.ok) {
                const data = await res.json();
                hardwareStatus.textContent = `GPU: ${data.device}`;
            } else {
                hardwareStatus.textContent = 'Backend Error';
            }
        } catch (e) {
            hardwareStatus.textContent = 'Backend Offline';
        }
    }

    async function uploadAndAnalyze(file) {
        loadingOverlay.classList.remove('hidden');
        
        const formData = new FormData();
        formData.append('image', currentFile);
        
        const patientName = document.getElementById('patient-name').value;
        const patientDetails = document.getElementById('patient-details').value;
        if (patientName) formData.append('patient_name', patientName);
        if (patientDetails) formData.append('patient_details', patientDetails);

        try {
            const response = await fetch('http://127.0.0.1:5000/predict', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.error || 'Prediction failed');
            }
            
            const data = await response.json();
            displayResults(data);
            
        } catch (error) {
            alert(`Error: ${error.message}`);
            console.error(error);
        } finally {
            loadingOverlay.classList.add('hidden');
        }
    }

    function displayResults(data) {
        switchToHomeView();
        resultsContainer.classList.remove('hidden');
        
        // Scroll to results smoothly
        resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });

        const mainClassEl = document.getElementById('main-class-result');
        const confTextEl = document.getElementById('confidence-text');
        const confCircleEl = document.getElementById('confidence-circle-path');
        const predSuccessEl = document.getElementById('prediction-success');
        const predWarningEl = document.getElementById('prediction-warning');
        const predictionTitleEl = document.getElementById('prediction-result-title');
        
        const topPredictionsCardEl = document.getElementById('top-predictions-card');
        const confidenceCircleContainerEl = document.getElementById('confidence-circle-container');
        
        const topTitleEl = document.getElementById('top-predictions-title');
        const topDescEl = document.getElementById('top-predictions-desc');
        const gradcamSection = document.getElementById('gradcam-section');
        
        const confPercent = Math.round(data.confidence * 100);
        confTextEl.textContent = `${confPercent}%`;
        confCircleEl.setAttribute('stroke-dasharray', `${confPercent}, 100`);

        if (data.status === 'uncertain') {
            predSuccessEl.classList.add('hidden');
            predWarningEl.classList.remove('hidden');
            predictionTitleEl.textContent = "Unidentified Image";
            
            topPredictionsCardEl.classList.add('hidden');
            confidenceCircleContainerEl.classList.add('hidden');
            gradcamSection.classList.add('hidden');
        } else {
            predSuccessEl.classList.remove('hidden');
            predWarningEl.classList.add('hidden');
            predictionTitleEl.textContent = "Prediction Result";
            
            topPredictionsCardEl.classList.remove('hidden');
            confidenceCircleContainerEl.classList.remove('hidden');
            mainClassEl.textContent = formatClassName(data.predicted_class);
            
            const mappedBadge = document.getElementById('mapped-status-result');
            if (mappedBadge) {
                mappedBadge.textContent = data.mapped_status;
                if (data.mapped_status === 'Cancer Present') {
                    mappedBadge.style.color = '#F44336';
                } else {
                    mappedBadge.style.color = '#4CAF50';
                }
            }
            
            topTitleEl.textContent = "Top Predictions";
            topDescEl.classList.add('hidden');
            gradcamSection.classList.remove('hidden');
            
            // Update Grad-CAM Images
            const originalImg = document.getElementById('panel-original');
            const heatmapImg = document.getElementById('panel-heatmap');
            const overlayImg = document.getElementById('panel-overlay');
            const histoImg = document.getElementById('panel-histo');

            originalImg.src = imagePreview.src;
            heatmapImg.src = `data:image/jpeg;base64,${data.gradcam.heatmap}`;
            overlayImg.src = `data:image/jpeg;base64,${data.gradcam.overlay}`;
            if (data.histopathology_match) {
                histoImg.src = `http://127.0.0.1:5000/eval_dataset/${data.histopathology_match}`;
            }
        }

        // Update Top 3 Predictions
        const topContainer = document.getElementById('top-predictions-container');
        topContainer.innerHTML = '';
        
        data.top_predictions.forEach((pred, index) => {
            const percent = (pred.probability * 100).toFixed(2);
            const html = `
                <div class="progress-item">
                    <div class="progress-header">
                        <span>${index + 1}. ${formatClassName(pred.class)}</span>
                        <span>${percent}%</span>
                    </div>
                    <div class="progress-bar-container">
                        <div class="progress-fill" style="width: ${percent}%"></div>
                    </div>
                </div>
            `;
            topContainer.innerHTML += html;
        });
    }

    function formatClassName(name) {
        return name.split('-').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
    }

    function formatBytes(bytes, decimals = 2) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    }

    // Render Confusion Matrix Map
    const matrixData = [
        [193, 4, 0, 0, 0, 0, 1, 0],
        [34, 163, 0, 1, 0, 0, 0, 0],
        [0, 0, 143, 0, 0, 55, 0, 0],
        [0, 0, 0, 189, 0, 0, 3, 6],
        [0, 0, 0, 0, 192, 0, 4, 2],
        [0, 0, 17, 0, 0, 179, 1, 1],
        [3, 0, 0, 7, 2, 0, 180, 6],
        [1, 0, 0, 2, 1, 0, 4, 190]
    ];
    
    const matrixContainer = document.getElementById('confusion-matrix-container');
    matrixContainer.style.gridTemplateColumns = `repeat(8, 1fr)`;
    
    let maxVal = 0;
    matrixData.forEach(row => row.forEach(val => { if(val > maxVal) maxVal = val; }));
    
    matrixData.forEach(row => {
        row.forEach(val => {
            const cell = document.createElement('div');
            cell.className = 'matrix-cell';
            cell.textContent = val;
            
            // Calculate color based on value intensity
            const intensity = val / maxVal;
            // Interpolate color from dark blue to cyan
            if (val > 0) {
                // simple rgb calculation for heat
                const r = Math.floor(28 + intensity * (0 - 28));
                const g = Math.floor(43 + intensity * (212 - 43));
                const b = Math.floor(84 + intensity * (255 - 84));
                cell.style.backgroundColor = `rgb(${r}, ${g}, ${b})`;
                if (intensity > 0.5) cell.style.color = '#000';
            }
            
            matrixContainer.appendChild(cell);
        });
    });

    // Render F1-Score Chart
    const f1ChartCtx = document.getElementById('f1Chart');
    if (f1ChartCtx && typeof Chart !== 'undefined') {
        new Chart(f1ChartCtx, {
            type: 'bar',
            data: {
                labels: [
                    'esophagitis', 
                    'normal-cecum', 'normal-pylorus', 'normal-z-line', 
                    'polyps', 'ulcerative-colitis'
                ],
                datasets: [
                    {
                        label: 'F1-Score',
                        data: [0.80, 0.95, 0.98, 0.83, 0.92, 0.94],
                        backgroundColor: 'rgba(155, 89, 182, 0.7)',
                        borderColor: 'rgba(155, 89, 182, 1)',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { beginAtZero: true, max: 1.0, ticks: { color: '#b0bec5' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    x: { ticks: { color: '#b0bec5', maxRotation: 45, minRotation: 45 }, grid: { display: false } }
                },
                plugins: { legend: { labels: { color: '#e0e0e0', font: { size: 14 } } } }
            }
        });
    }

    // Render Precision Chart
    const precisionChartCtx = document.getElementById('precisionChart');
    if (precisionChartCtx && typeof Chart !== 'undefined') {
        new Chart(precisionChartCtx, {
            type: 'bar',
            data: {
                labels: [
                    'esophagitis', 
                    'normal-cecum', 'normal-pylorus', 'normal-z-line', 
                    'polyps', 'ulcerative-colitis'
                ],
                datasets: [
                    {
                        label: 'Precision',
                        data: [0.89, 0.95, 0.98, 0.76, 0.93, 0.93],
                        backgroundColor: 'rgba(52, 152, 219, 0.7)',
                        borderColor: 'rgba(52, 152, 219, 1)',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { beginAtZero: true, max: 1.0, ticks: { color: '#b0bec5' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    x: { ticks: { color: '#b0bec5', maxRotation: 45, minRotation: 45 }, grid: { display: false } }
                },
                plugins: { legend: { labels: { color: '#e0e0e0', font: { size: 14 } } } }
            }
        });
    }

    // Navigation Logic
    function setupNavigation() {
        const navLinks = document.querySelectorAll('.nav-link');
        const pageViews = document.querySelectorAll('.page-view');

        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                
                // Remove active class from all links and views
                navLinks.forEach(l => l.classList.remove('active'));
                pageViews.forEach(v => v.classList.remove('active'));
                
                // Add active class to clicked link and corresponding view
                link.classList.add('active');
                const targetId = link.getAttribute('data-target');
                document.getElementById(targetId).classList.add('active');
                
                if (targetId === 'view-history') {
                    loadHistory();
                }
                
                // Scroll to top when switching views
                window.scrollTo({ top: 0, behavior: 'smooth' });
            });
        });
    }

    function switchToHomeView() {
        const homeLink = document.querySelector('.nav-link[data-target="view-home"]');
        if (homeLink && !homeLink.classList.contains('active')) {
            homeLink.click();
        }
    }

    // Gallery Logic
    const galleryClassSelect = document.getElementById('gallery-class-select');
    const galleryNextBtn = document.getElementById('gallery-next-btn');
    const galleryLoading = document.getElementById('gallery-loading');
    const galleryResults = document.getElementById('gallery-results');
    
    let galleryCurrentIndex = 0;

    galleryClassSelect.addEventListener('change', () => {
        galleryCurrentIndex = 0;
        galleryResults.classList.add('hidden');
    });

    galleryNextBtn.addEventListener('click', () => {
        const className = galleryClassSelect.value;
        fetchGallerySample(className, galleryCurrentIndex);
        galleryCurrentIndex++;
    });

    async function fetchGallerySample(className, index) {
        galleryLoading.classList.remove('hidden');
        galleryResults.classList.add('hidden');
        
        try {
            const response = await fetch(`http://127.0.0.1:5000/sample?class=${className}&index=${index}`);
            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.error || 'Failed to fetch sample');
            }
            const data = await response.json();
            displayGalleryResults(data);
        } catch (error) {
            alert(`Error: ${error.message}`);
            console.error(error);
            galleryLoading.classList.add('hidden');
        }
    }

    function displayGalleryResults(data) {
        galleryLoading.classList.add('hidden');
        galleryResults.classList.remove('hidden');

        const mainClassEl = document.getElementById('gallery-class-result');
        const confTextEl = document.getElementById('gallery-confidence-text');
        const confCircleEl = document.getElementById('gallery-confidence-circle-path');
        const predSuccessEl = document.getElementById('gallery-prediction-success');
        const predWarningEl = document.getElementById('gallery-prediction-warning');
        const actualClassEl = document.getElementById('gallery-actual-class');
        const actualClassWarnEl = document.getElementById('gallery-actual-class-warn');
        
        const galleryTopPredictionsCardEl = document.getElementById('gallery-top-predictions-card');
        const galleryConfidenceCircleContainerEl = document.getElementById('gallery-confidence-circle-container');
        
        const topTitleEl = document.getElementById('gallery-top-predictions-title');
        const topDescEl = document.getElementById('gallery-top-predictions-desc');
        const gradcamSection = document.getElementById('gallery-gradcam-section');
        
        const confPercent = Math.round(data.confidence * 100);
        confTextEl.textContent = `${confPercent}%`;
        confCircleEl.setAttribute('stroke-dasharray', `${confPercent}, 100`);

        const formattedActualClass = formatClassName(data.actual_class);

        if (data.status === 'uncertain') {
            predSuccessEl.classList.add('hidden');
            predWarningEl.classList.remove('hidden');
            actualClassWarnEl.textContent = `Actual: ${formattedActualClass} (${data.filename})`;
            
            galleryTopPredictionsCardEl.classList.add('hidden');
            galleryConfidenceCircleContainerEl.classList.add('hidden');
            gradcamSection.classList.add('hidden');
        } else {
            predSuccessEl.classList.remove('hidden');
            predWarningEl.classList.add('hidden');
            
            galleryTopPredictionsCardEl.classList.remove('hidden');
            galleryConfidenceCircleContainerEl.classList.remove('hidden');
            mainClassEl.textContent = formatClassName(data.predicted_class);
            actualClassEl.textContent = `Actual: ${formattedActualClass} (${data.filename})`;
            
            const galleryMappedBadge = document.getElementById('gallery-mapped-status');
            if (galleryMappedBadge) {
                galleryMappedBadge.textContent = data.mapped_status;
                if (data.mapped_status === 'Cancer Present') {
                    galleryMappedBadge.style.color = '#F44336';
                } else {
                    galleryMappedBadge.style.color = '#4CAF50';
                }
            }
            
            // Highlight actual vs predicted matches
            if (data.actual_class === data.predicted_class) {
                actualClassEl.style.color = '#4CAF50';
                actualClassEl.style.borderColor = '#4CAF50';
            } else {
                actualClassEl.style.color = '#F44336';
                actualClassEl.style.borderColor = '#F44336';
            }
            
            topTitleEl.textContent = "Top Predictions";
            topDescEl.classList.add('hidden');
            gradcamSection.classList.remove('hidden');
            
            // Update Grad-CAM Images
            const originalImg = document.getElementById('gallery-panel-original');
            const heatmapImg = document.getElementById('gallery-panel-heatmap');
            const overlayImg = document.getElementById('gallery-panel-overlay');
            const histoImg = document.getElementById('gallery-panel-histo');

            originalImg.src = `http://127.0.0.1:5000/eval_dataset/${data.folder_type}/${data.actual_class}/${data.filename}`;
            heatmapImg.src = `data:image/jpeg;base64,${data.gradcam.heatmap}`;
            overlayImg.src = `data:image/jpeg;base64,${data.gradcam.overlay}`;
            if (data.histopathology_match) {
                histoImg.src = `http://127.0.0.1:5000/eval_dataset/${data.histopathology_match}`;
            }
        }

        // Update Top 3 Predictions
        const topContainer = document.getElementById('gallery-top-predictions-container');
        topContainer.innerHTML = '';
        
        data.top_predictions.forEach((pred, index) => {
            const percent = (pred.probability * 100).toFixed(2);
            const html = `
                <div class="progress-item">
                    <div class="progress-header">
                        <span>${index + 1}. ${formatClassName(pred.class)}</span>
                        <span>${percent}%</span>
                    </div>
                    <div class="progress-bar-container">
                        <div class="progress-fill" style="width: ${percent}%"></div>
                    </div>
                </div>
            `;
            topContainer.innerHTML += html;
        });
    }

    // Dataset Evaluation Logic
    const evalDatasetSelect = document.getElementById('eval-dataset-select');
    const evalNextBtn = document.getElementById('eval-next-btn');
    const evalLoading = document.getElementById('eval-loading');
    const evalResults = document.getElementById('eval-results');
    const evalHistoContainer = document.getElementById('eval-histo-container');
    const evalStatsContainer = document.getElementById('eval-stats-container');
    
    let evalStats = {
        'cancer-0': { total: 0, correct: 0, incorrect: 0 },
        'cancer-1': { total: 0, correct: 0, incorrect: 0 }
    };

    const CANCER_CLASSES = ["polyps", "ulcerative-colitis"];
    const NO_CANCER_CLASSES = ["normal-cecum", "normal-pylorus", "normal-z-line", "esophagitis"];

    function mapPredictionToEvalClass(rawClass) {
        if (CANCER_CLASSES.includes(rawClass)) return "Cancer";
        if (NO_CANCER_CLASSES.includes(rawClass)) return "No Cancer";
        return "Unknown";
    }

    evalDatasetSelect.addEventListener('change', () => {
        evalResults.classList.add('hidden');
        if (evalDatasetSelect.value === 'histopathology') {
            evalStatsContainer.classList.add('hidden');
            evalHistoContainer.classList.remove('hidden');
            evalNextBtn.textContent = 'Load Random Samples';
        } else {
            evalStatsContainer.classList.remove('hidden');
            evalHistoContainer.classList.add('hidden');
            evalNextBtn.textContent = 'Evaluate Random Image';
        }
    });

    evalNextBtn.addEventListener('click', () => {
        const type = evalDatasetSelect.value;
        if (type === 'histopathology') {
            fetchHistopathology();
        } else {
            fetchEvalSample(type);
        }
    });

    async function fetchHistopathology() {
        evalLoading.classList.remove('hidden');
        evalHistoContainer.classList.add('hidden');
        
        try {
            const response = await fetch(`http://127.0.0.1:5000/eval/histopathology?count=8`);
            if (!response.ok) throw new Error('Failed to fetch samples');
            const data = await response.json();
            
            const grid = document.getElementById('histo-grid');
            grid.innerHTML = '';
            data.samples.forEach(sample => {
                const color = sample.mapped_status === 'Cancer Yes' ? '#F44336' : '#4CAF50';
                grid.innerHTML += `
                <div style="text-align: center; background: var(--card-bg); padding: 10px; border-radius: 12px; border: 1px solid var(--border-color);">
                    <img src="http://127.0.0.1:5000/eval_dataset/${sample.file_path}" style="width:100%; aspect-ratio:1; object-fit:cover; border-radius:8px; border:2px solid ${color};">
                    <p style="color: ${color}; font-weight: bold; margin-top: 10px; font-size: 1.1rem;">${sample.mapped_status}</p>
                    <p style="color: var(--text-secondary); font-size: 0.9rem; margin-top: 4px;">Conf: ${(sample.confidence * 100).toFixed(1)}%</p>
                </div>
                `;
            });
            
            evalHistoContainer.classList.remove('hidden');
        } catch (error) {
            alert(`Error: ${error.message}`);
        } finally {
            evalLoading.classList.add('hidden');
        }
    }

    async function fetchEvalSample(type) {
        evalLoading.classList.remove('hidden');
        evalResults.classList.add('hidden');
        
        try {
            const response = await fetch(`http://127.0.0.1:5000/eval/sample?type=${type}`);
            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.error || 'Failed to fetch sample');
            }
            const data = await response.json();
            displayEvalResults(data, type);
        } catch (error) {
            alert(`Error: ${error.message}`);
            console.error(error);
        } finally {
            evalLoading.classList.add('hidden');
        }
    }

    function displayEvalResults(data, type) {
        evalResults.classList.remove('hidden');
        
        const groundTruthEl = document.getElementById('eval-ground-truth');
        const predictionEl = document.getElementById('eval-prediction');
        const confidenceEl = document.getElementById('eval-confidence');
        const badgeEl = document.getElementById('eval-result-badge');
        
        const mappedPrediction = data.mapped_status === 'Cancer Present' ? 'Cancer' : 'No Cancer';
        const isCorrect = (mappedPrediction === data.ground_truth);
        
        groundTruthEl.textContent = data.ground_truth;
        predictionEl.innerHTML = `${formatClassName(data.predicted_class)} <br><span style="color: ${data.mapped_status === 'Cancer Present' ? '#F44336' : '#4CAF50'}; font-weight: bold; font-size: 1.1rem;">(${data.mapped_status})</span>`;
        
        const confPercent = Math.round(data.confidence * 100);
        confidenceEl.textContent = `${(data.confidence * 100).toFixed(1)}%`;
        
        const confCircleText = document.getElementById('eval-confidence-text');
        const confCirclePath = document.getElementById('eval-confidence-circle-path');
        confCircleText.textContent = `${confPercent}%`;
        confCirclePath.setAttribute('stroke-dasharray', `${confPercent}, 100`);
        
        if (isCorrect) {
            badgeEl.textContent = "Result: Correct";
            badgeEl.style.color = "#4CAF50";
            badgeEl.style.borderColor = "#4CAF50";
            badgeEl.style.backgroundColor = "rgba(76, 175, 80, 0.1)";
        } else {
            badgeEl.textContent = "Result: Incorrect";
            badgeEl.style.color = "#F44336";
            badgeEl.style.borderColor = "#F44336";
            badgeEl.style.backgroundColor = "rgba(244, 67, 54, 0.1)";
        }
        
        // Update images
        document.getElementById('eval-panel-original').src = `http://127.0.0.1:5000/eval_dataset/${data.file_path}`;
        document.getElementById('eval-panel-heatmap').src = `data:image/jpeg;base64,${data.gradcam.heatmap}`;
        document.getElementById('eval-panel-overlay').src = `data:image/jpeg;base64,${data.gradcam.overlay}`;
        if (data.histopathology_match) {
            document.getElementById('eval-panel-histo').src = `http://127.0.0.1:5000/eval_dataset/${data.histopathology_match}`;
        }
        
        // Update Stats
        updateStats(type, isCorrect);
    }

    function updateStats(type, isCorrect) {
        evalStats[type].total++;
        if (isCorrect) evalStats[type].correct++;
        else evalStats[type].incorrect++;
        
        // Draw stats
        ['cancer-0', 'cancer-1'].forEach(t => {
            const prefix = t === 'cancer-0' ? 'stat-c0' : 'stat-c1';
            const s = evalStats[t];
            document.getElementById(`${prefix}-total`).textContent = s.total;
            document.getElementById(`${prefix}-correct`).textContent = s.correct;
            document.getElementById(`${prefix}-incorrect`).textContent = s.incorrect;
            document.getElementById(`${prefix}-accuracy`).textContent = s.total > 0 ? Math.round((s.correct/s.total)*100) + '%' : '0%';
        });
        
        const total = evalStats['cancer-0'].total + evalStats['cancer-1'].total;
        const correct = evalStats['cancer-0'].correct + evalStats['cancer-1'].correct;
        const incorrect = evalStats['cancer-0'].incorrect + evalStats['cancer-1'].incorrect;
        
        document.getElementById('stat-all-total').textContent = total;
        document.getElementById('stat-all-correct').textContent = correct;
        document.getElementById('stat-all-incorrect').textContent = incorrect;
        document.getElementById('stat-all-accuracy').textContent = total > 0 ? Math.round((correct/total)*100) + '%' : '0%';
    }

    // --- History Logic ---
    const reportModal = document.getElementById('report-modal');
    const closeModalBtn = document.getElementById('close-modal-btn');
    
    closeModalBtn.addEventListener('click', () => {
        reportModal.classList.add('hidden');
    });
    
    // Close modal when clicking outside content
    reportModal.addEventListener('click', (e) => {
        if (e.target === reportModal) {
            reportModal.classList.add('hidden');
        }
    });

    async function loadHistory() {
        const historyGrid = document.getElementById('history-grid');
        historyGrid.innerHTML = '<p style="color: var(--text-secondary); text-align: center; grid-column: 1 / -1;">Loading history...</p>';
        
        try {
            const response = await fetch('http://127.0.0.1:5000/history');
            const data = await response.json();
            
            if (!data.history || data.history.length === 0) {
                historyGrid.innerHTML = '<p style="color: var(--text-secondary); text-align: center; grid-column: 1 / -1;">No history found. Analyze an image on the Home page first!</p>';
                return;
            }
            
            historyGrid.innerHTML = '';
            historyGrid.style.gridTemplateColumns = '1fr'; // Force 1 column for detailed rows
            
            data.history.forEach(item => {
                const isCancer = item.mapped_status === 'Cancer Present';
                const statusColor = isCancer ? '#F44336' : '#4CAF50';
                
                const card = document.createElement('div');
                card.style.cssText = `background: var(--bg-tertiary); border-radius: 12px; padding: 1.5rem; border: 1px solid var(--border-color); display: flex; flex-direction: column; gap: 1rem;`;
                
                const date = new Date(item.timestamp);
                const timeString = date.toLocaleTimeString() + ' | ' + date.toLocaleDateString();
                
                const histoHtml = item.histopathology_match 
                    ? `<div style="flex: 1;"><p style="font-size: 0.75rem; color: var(--text-secondary); margin-bottom: 5px;">Match</p><img src="http://127.0.0.1:5000/eval_dataset/${item.histopathology_match}" style="width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 6px;"></div>` 
                    : '';

                card.innerHTML = `
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 1px solid var(--border-color); padding-bottom: 1rem;">
                        <div>
                            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 5px;">
                                <span style="background: var(--accent-blue); color: white; padding: 4px 8px; border-radius: 4px; font-family: monospace; font-size: 0.9rem; font-weight: bold;">${item.record_id || 'PR-???'}</span>
                                <span style="color: var(--text-secondary); font-size: 0.9rem;">${timeString}</span>
                            </div>
                            <h3 style="color: var(--text-primary); margin: 5px 0;">${item.patient_name || 'Anonymous Patient'}</h3>
                            <p style="color: var(--text-secondary); font-size: 0.9rem; max-width: 600px;">${item.patient_details || 'No additional details provided.'}</p>
                        </div>
                        <div style="text-align: right;">
                            <div style="color: ${statusColor}; font-weight: bold; font-size: 1.2rem; margin-bottom: 5px;">${item.mapped_status}</div>
                            <div style="color: var(--text-primary); text-transform: capitalize; font-size: 1rem;">${item.predicted_class}</div>
                            <div style="color: var(--accent-blue); font-weight: bold; font-size: 1.1rem; margin-top: 5px;">${(item.confidence * 100).toFixed(1)}% Confidence</div>
                        </div>
                    </div>
                    
                    <div style="display: flex; gap: 1rem; margin-top: 0.5rem;">
                        <div style="flex: 1;">
                            <p style="font-size: 0.75rem; color: var(--text-secondary); margin-bottom: 5px;">Original</p>
                            <img src="http://127.0.0.1:5000/history_image/${item.image_filename}" style="width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 6px;">
                        </div>
                        <div style="flex: 1;">
                            <p style="font-size: 0.75rem; color: var(--text-secondary); margin-bottom: 5px;">Heatmap</p>
                            <img src="data:image/jpeg;base64,${item.gradcam.heatmap}" style="width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 6px;">
                        </div>
                        <div style="flex: 1;">
                            <p style="font-size: 0.75rem; color: var(--text-secondary); margin-bottom: 5px;">Overlay</p>
                            <img src="data:image/jpeg;base64,${item.gradcam.overlay}" style="width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 6px;">
                        </div>
                        ${histoHtml}
                    </div>
                    
                    <button class="btn btn-secondary" style="width: 100%; padding: 10px; margin-top: 0.5rem;" onclick='openReportModal(${JSON.stringify(item).replace(/'/g, "&#39;")})'>Expand Full Report</button>
                `;
                
                historyGrid.appendChild(card);
            });
            
        } catch (error) {
            historyGrid.innerHTML = `<p style="color: #F44336; text-align: center; grid-column: 1 / -1;">Error loading history: ${error.message}</p>`;
        }
    }

    // Expose openReportModal globally so inline onclick can use it
    window.openReportModal = function(item) {
        document.getElementById('modal-record-id').textContent = item.record_id || 'PR-???';
        document.getElementById('modal-patient-name').textContent = item.patient_name || 'Anonymous';
        document.getElementById('modal-patient-details').textContent = item.patient_details || 'No additional details provided.';
        
        document.getElementById('modal-prediction').textContent = item.predicted_class;
        
        const statusEl = document.getElementById('modal-status');
        statusEl.textContent = item.mapped_status;
        const isCancer = item.mapped_status === 'Cancer Present';
        statusEl.style.backgroundColor = isCancer ? 'rgba(244, 67, 54, 0.2)' : 'rgba(76, 175, 80, 0.2)';
        statusEl.style.color = isCancer ? '#F44336' : '#4CAF50';
        statusEl.style.border = `1px solid ${isCancer ? '#F44336' : '#4CAF50'}`;
        
        document.getElementById('modal-confidence').textContent = (item.confidence * 100).toFixed(1) + '%';
        
        document.getElementById('modal-original').src = `http://127.0.0.1:5000/history_image/${item.image_filename}`;
        document.getElementById('modal-heatmap').src = `data:image/jpeg;base64,${item.gradcam.heatmap}`;
        document.getElementById('modal-overlay').src = `data:image/jpeg;base64,${item.gradcam.overlay}`;
        
        const histoImg = document.getElementById('modal-histo');
        if (item.histopathology_match) {
            histoImg.src = `http://127.0.0.1:5000/eval_dataset/${item.histopathology_match}`;
            histoImg.style.display = 'block';
        } else {
            histoImg.style.display = 'none';
        }
        
        reportModal.classList.remove('hidden');
    };
});
