// Prediction Form
const predictForm = document.getElementById("predict-form");
const fileInput = document.getElementById("code-file");
const baseFileInput = document.getElementById("base-file");
const selectedFileLabel = document.getElementById("selected-file");
const selectedBaseFileLabel = document.getElementById("selected-base-file");
const commitMessageField = document.getElementById("commit-message");
const resultsContainer = document.getElementById("results");

// Training Form
const trainingForm = document.getElementById("training-form");
const trainingProgress = document.getElementById("training-progress");
const trainingResults = document.getElementById("training-results");
const progressBarFill = document.getElementById("progress-bar-fill");
const progressText = document.getElementById("progress-text");
const progressMessage = document.getElementById("progress-message");
const metricsGrid = document.getElementById("metrics-grid");

// File input handlers
fileInput.addEventListener("change", () => {
    if (fileInput.files.length === 0) {
        selectedFileLabel.textContent = "No file selected";
        return;
    }
    selectedFileLabel.textContent = fileInput.files[0].name;
});

if (baseFileInput) {
    baseFileInput.addEventListener("change", () => {
        if (baseFileInput.files.length === 0) {
            selectedBaseFileLabel.textContent = "No parent file selected";
            return;
        }
        selectedBaseFileLabel.textContent = baseFileInput.files[0].name;
    });
}

// Training Form Handler
trainingForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    
    const repoUrl = document.getElementById("repo-url").value.trim();
    const modelType = document.getElementById("model-type").value;
    
        // Get advanced configuration
        const startDate = document.getElementById("start-date")?.value || "";
        const endDate = document.getElementById("end-date")?.value || "";
        const commitHash = document.getElementById("commit-hash")?.value.trim() || "";
        const bugKeywords = document.getElementById("bug-keywords")?.value.trim() || "";
        const granularity = document.querySelector('input[name="granularity"]:checked')?.value || 'file';
        const selectedFeatures = Array.from(document.querySelectorAll('input[name="features"]:checked')).map(cb => cb.value);
        const maxCommits = document.getElementById("max-commits")?.value || "";
    
    // Show progress dashboard
    trainingProgress.style.display = "block";
    trainingResults.style.display = "none";
    
    // Clear commits display
    const commitsContainer = document.getElementById("commits-container");
    if (commitsContainer) {
        commitsContainer.style.display = "none";
        commitsContainer.innerHTML = "";
    }
    
    // Reset stop button state
    const stopBtn = document.getElementById("stop-training-btn");
    if (stopBtn) {
        stopBtn.style.display = "none";
        stopBtn.disabled = false;
        stopBtn.innerHTML = '<span class="btn-icon">⏹</span> Stop Training';
    }
    
    // Clear any existing polling interval
    if (trainingPollInterval) {
        clearInterval(trainingPollInterval);
        trainingPollInterval = null;
    }
    
    // Clear current job ID
    window.currentTrainingJobId = null;
    
    updateProgress(0, "Starting training...");
    
    // Disable form
    trainingForm.querySelector("button").disabled = true;
    
    // Add log entry
    if (window.addLogEntry) {
        window.addLogEntry("Starting training pipeline...", "info");
        if (repoUrl) {
            window.addLogEntry(`Repository: ${repoUrl}`, "info");
        }
        window.addLogEntry(`Model Type: ${modelType}`, "info");
        window.addLogEntry(`Granularity: ${granularity}`, "info");
    }
    
    try {
        // Use FormData to send all fields
        const formData = new FormData();
        if (repoUrl) {
            formData.append("repo_url", repoUrl);
        }
        formData.append("model_type", modelType);
        
        // Add advanced configuration
        if (startDate) formData.append("start_date", startDate);
        if (endDate) formData.append("end_date", endDate);
        if (commitHash) formData.append("commit_hash", commitHash);
        if (bugKeywords) formData.append("bug_keywords", bugKeywords);
        if (maxCommits) formData.append("max_commits", maxCommits);
        formData.append("granularity", granularity);
        
        // Add Jira configuration
        const jiraUrl = document.getElementById("jira-url")?.value.trim() || "";
        const jiraUsername = document.getElementById("jira-username")?.value.trim() || "";
        const jiraApiToken = document.getElementById("jira-api-token")?.value.trim() || "";
        const enableJira = document.getElementById("enable-jira")?.checked || false;
        
        if (enableJira && jiraUrl && jiraUsername && jiraApiToken) {
            formData.append("jira_url", jiraUrl);
            formData.append("jira_username", jiraUsername);
            formData.append("jira_api_token", jiraApiToken);
        }
        
        selectedFeatures.forEach(feature => {
            formData.append("features", feature);
        });
        
        // Start training
        const response = await fetch("/api/train", {
            method: "POST",
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || "Training failed to start");
        }
        
        const jobId = data.job_id;
        console.log("Training started with job ID:", jobId);
        
        // Store job ID for stop functionality
        window.currentTrainingJobId = jobId;
        
        // Show and reset stop button (ensure it's fully reset)
        const stopBtn = document.getElementById("stop-training-btn");
        if (stopBtn) {
            // First reset the button state completely
            stopBtn.disabled = false;
            stopBtn.innerHTML = '<span class="btn-icon">⏹</span> Stop Training';
            // Then show it
            stopBtn.style.display = "inline-flex";
        }
        
        // Poll for status
        pollTrainingStatus(jobId);
        
    } catch (error) {
        updateProgress(0, `Error: ${error.message}`, "error");
        trainingForm.querySelector("button").disabled = false;
        // Reset stop button on error
        const stopBtn = document.getElementById("stop-training-btn");
        if (stopBtn) {
            stopBtn.style.display = "none";
            stopBtn.disabled = false;
            stopBtn.innerHTML = '<span class="btn-icon">⏹</span> Stop Training';
        }
        // Clear polling interval
        if (trainingPollInterval) {
            clearInterval(trainingPollInterval);
            trainingPollInterval = null;
        }
        window.currentTrainingJobId = null;
    }
});

// Store polling interval for cleanup
let trainingPollInterval = null;

// Poll training status
async function pollTrainingStatus(jobId) {
    // Clear any existing interval
    if (trainingPollInterval) {
        clearInterval(trainingPollInterval);
    }
    
    trainingPollInterval = setInterval(async () => {
        try {
            const response = await fetch(`/api/training/status/${jobId}`);
            const status = await response.json();
            
            if (!response.ok) {
                throw new Error(status.error || "Failed to get status");
            }
            
            // Update progress
            const progress = status.progress || 0;
            updateProgress(progress, status.message || "Processing...");
            
            // Update steps based on progress
            updateSteps(progress);
            
            // Check if commits are available (after cloning)
            if (status.commits && status.commits.length > 0 && progress >= 6) {
                showCommits(status.commits, status.repo_url);
            }
            
            // Check if completed, failed, or cancelled
            if (status.status === "completed") {
                clearInterval(trainingPollInterval);
                trainingPollInterval = null;
                updateProgress(100, "Training completed successfully!", "success");
                showTrainingResults(status.results);
                trainingForm.querySelector("button").disabled = false;
                hideStopButton();
            } else if (status.status === "failed") {
                clearInterval(trainingPollInterval);
                trainingPollInterval = null;
                updateProgress(0, `Training failed: ${status.error}`, "error");
                trainingForm.querySelector("button").disabled = false;
                hideStopButton();
            } else if (status.status === "cancelled") {
                clearInterval(trainingPollInterval);
                trainingPollInterval = null;
                updateProgress(progress, "Training cancelled by user", "error");
                trainingForm.querySelector("button").disabled = false;
                hideStopButton();
            }
            
        } catch (error) {
            clearInterval(trainingPollInterval);
            trainingPollInterval = null;
            updateProgress(0, `Error: ${error.message}`, "error");
            trainingForm.querySelector("button").disabled = false;
            hideStopButton();
        }
    }, 2000); // Poll every 2 seconds
}

// Stop training function
async function stopTraining() {
    const jobId = window.currentTrainingJobId;
    if (!jobId) {
        alert("No active training job to stop");
        return;
    }
    
    if (!confirm("Are you sure you want to stop the training? This action cannot be undone.")) {
        return;
    }
    
    const stopBtn = document.getElementById("stop-training-btn");
    if (stopBtn) {
        stopBtn.disabled = true;
        stopBtn.innerHTML = '<span class="btn-icon">⏳</span> Stopping...';
    }
    
    try {
        const response = await fetch(`/api/training/stop/${jobId}`, {
            method: "POST"
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || "Failed to stop training");
        }
        
        if (window.addLogEntry) {
            window.addLogEntry("Training stopped by user", "warning");
        }
        
        // The polling will detect the cancelled status and update the UI
        hideStopButton();
        
    } catch (error) {
        console.error("Error stopping training:", error);
        alert(`Failed to stop training: ${error.message}`);
        if (stopBtn) {
            stopBtn.disabled = false;
            stopBtn.innerHTML = '<span class="btn-icon">⏹</span> Stop Training';
        }
    }
}

function hideStopButton() {
    const stopBtn = document.getElementById("stop-training-btn");
    if (stopBtn) {
        stopBtn.style.display = "none";
        stopBtn.disabled = false;
        stopBtn.innerHTML = '<span class="btn-icon">⏹</span> Stop Training';
    }
    window.currentTrainingJobId = null;
}

// Update progress bar
function updateProgress(percent, message, status = "running") {
    progressBarFill.style.width = `${percent}%`;
    progressText.textContent = `${percent.toFixed(1)}%`;
    progressMessage.textContent = message;
    
    // Add log entry
    if (window.addLogEntry) {
        const logType = status === "error" ? "error" : status === "completed" ? "success" : "info";
        window.addLogEntry(`${message} (${percent.toFixed(1)}%)`, logType);
    }
    
    // Update color based on status
    progressBarFill.className = "progress-bar-fill";
    if (status === "success") {
        progressBarFill.classList.add("success");
    } else if (status === "error") {
        progressBarFill.classList.add("error");
    }
}

// Update step indicators
function updateSteps(progress) {
    const steps = [
        { id: "step-szz", threshold: 20 },
        { id: "step-features", threshold: 40 },
        { id: "step-training", threshold: 80 },
        { id: "step-complete", threshold: 100 }
    ];
    
    steps.forEach((step, index) => {
        const stepEl = document.getElementById(step.id);
        const icon = stepEl.querySelector(".step-icon");
        
        if (progress >= step.threshold) {
            icon.textContent = "✅";
            stepEl.classList.add("completed");
        } else if (progress >= (index > 0 ? steps[index - 1].threshold : 0)) {
            icon.textContent = "⏳";
            stepEl.classList.add("active");
        } else {
            icon.textContent = "⏸️";
            stepEl.classList.remove("active", "completed");
        }
    });
}

// Show commits in UI
function showCommits(commits, repoUrl) {
    const commitsContainer = document.getElementById("commits-container");
    if (!commitsContainer) return;
    
    commitsContainer.style.display = "block";
    commitsContainer.innerHTML = `
        <h3>Repository Commits</h3>
        <p class="repo-info">Repository: <strong>${repoUrl || 'N/A'}</strong></p>
        <p class="commits-count">Showing <strong>${commits.length}</strong> commit(s)</p>
        <div class="commits-list">
            ${commits.map(commit => `
                <div class="commit-card">
                    <div class="commit-header">
                        <span class="commit-hash" title="${commit.hash}">${commit.short_hash || commit.hash.substring(0, 7)}</span>
                        <span class="commit-author">${commit.author || 'Unknown'}</span>
                        <span class="commit-date">${formatDate(commit.date)}</span>
                    </div>
                    <div class="commit-message">${escapeHtml(commit.message || 'No message')}</div>
                </div>
            `).join('')}
        </div>
    `;
}

function formatDate(dateStr) {
    if (!dateStr) return 'N/A';
    try {
        const date = new Date(dateStr);
        return date.toLocaleString();
    } catch (e) {
        return dateStr;
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Show training results
function showTrainingResults(results) {
    if (!results) return;
    
    trainingResults.style.display = "block";
    
    const metrics = [
        { label: "Best Model", value: results.best_model || "N/A" },
        { label: "F1-Score", value: (results.best_f1 || 0).toFixed(4) },
        { label: "Accuracy", value: ((results.best_accuracy || 0) * 100).toFixed(2) + "%" },
        { label: "Recall", value: ((results.best_recall || 0) * 100).toFixed(2) + "%" },
        { label: "Precision", value: ((results.best_precision || 0) * 100).toFixed(2) + "%" },
        { label: "ROC-AUC", value: (results.best_roc_auc || 0).toFixed(4) }
    ];
    
    metricsGrid.innerHTML = metrics.map(metric => `
        <div class="metric-card">
            <div class="metric-label">${metric.label}</div>
            <div class="metric-value">${metric.value}</div>
        </div>
    `).join("");
    
    // Show performance metrics chart
    if (typeof visualizations !== 'undefined') {
        const vizContainer = document.getElementById('training-visualizations');
        if (vizContainer) {
            visualizations.renderPerformanceMetrics('training-visualizations', {
                accuracy: results.best_accuracy || 0,
                precision: results.best_precision || 0,
                recall: results.best_recall || 0,
                f1: results.best_f1 || 0,
                roc_auc: results.best_roc_auc || 0
            });
        }
    }
}

// Prediction Form Handler
predictForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    if (fileInput.files.length === 0) {
        alert("Please choose a code file to upload.");
        return;
    }

    const formData = new FormData();
    formData.append("code_file", fileInput.files[0]);
    const selectedModel = predictForm.elements["model"].value;
    formData.append("model", selectedModel);
    
    // Add function-level option
    const functionLevelCheckbox = document.getElementById("function-level-checkbox");
    if (functionLevelCheckbox && functionLevelCheckbox.checked) {
        formData.append("function_level", "true");
    }
    
    if (baseFileInput && baseFileInput.files.length > 0) {
        formData.append("base_code", baseFileInput.files[0]);
    }
    if (commitMessageField) {
        formData.append("commit_message", commitMessageField.value || "");
    }

    toggleFormDisabled(true);
    showLoadingState();

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            const errorText = await response.text();
            let errorMessage = "Prediction failed";
            try {
                const errorJson = JSON.parse(errorText);
                errorMessage = errorJson.error || errorMessage;
            } catch (e) {
                errorMessage = errorText || errorMessage;
            }
            throw new Error(errorMessage);
        }

        const responseText = await response.text();
        let payload;
        try {
            payload = JSON.parse(responseText);
        } catch (e) {
            console.error("JSON parse error:", e, "Response:", responseText);
            throw new Error("Invalid response from server. Please try again.");
        }

        window.lastPredictionData = payload; // Store for table view toggle
        window.currentResultsData = payload; // Also store here for function-level
        renderResults(payload);
    } catch (error) {
        console.error("Prediction error:", error);
        renderError(error.message);
    } finally {
        toggleFormDisabled(false);
    }
});

function toggleFormDisabled(disabled) {
    Array.from(predictForm.elements).forEach((el) => {
        el.disabled = disabled;
    });
}

function showLoadingState() {
    resultsContainer.innerHTML = `
        <div class="card result-card">
            <p>Running predictions…</p>
        </div>
    `;
}

function renderError(message) {
    resultsContainer.innerHTML = `
        <div class="card result-card error-banner">
            <strong>Error:</strong> ${message}
        </div>
    `;
}

function renderResults(data) {
    // Check view preference (default to card view)
    const useTableView = window.currentViewMode === 'table';
    
    // Handle function-level results
    if (data.function_level && data.predictions && Array.isArray(data.predictions)) {
        // Group predictions by function name
        const functionGroups = {};
        data.predictions.forEach(funcPred => {
            const funcName = funcPred.function_name || 'Unknown';
            if (!functionGroups[funcName]) {
                functionGroups[funcName] = {
                    function_name: funcName,
                    class_name: funcPred.class_name || '',
                    function_start_line: funcPred.function_start_line || 'N/A',
                    function_end_line: funcPred.function_end_line || 'N/A',
                    is_method: funcPred.is_method || false,
                    predictions: []
                };
            }
            functionGroups[funcName].predictions.push(funcPred);
        });
        
        const functionGroupsArray = Object.values(functionGroups);
        
        // Function-level results - show each function separately
        // Deduplicate predictions by model name within each function
        const functionResultsHtml = functionGroupsArray.map((funcGroup, idx) => {
            const funcName = funcGroup.function_name;
            const seenModelNames = new Set();
            const uniquePredictions = funcGroup.predictions.filter((funcPred) => {
                const modelName = funcPred.name || 'Unknown Model';
                if (seenModelNames.has(modelName)) {
                    return false; // Skip duplicate
                }
                seenModelNames.add(modelName);
                return true;
            });
            
            const predictionsHtml = uniquePredictions.map((funcPred) => {
                const probability = parseFloat(funcPred.probability || 0) * 100;
                const probabilityStr = probability.toFixed(2);
                const barWidth = Math.min(100, Math.max(0, probability));
                const barColor = probability >= 50 ? 'var(--error)' : probability >= 25 ? '#fbbf24' : 'var(--success)';
                const label = funcPred.label || 'CLEAN';
                const labelLower = label.toLowerCase();
                
                return `
                    <div class="prediction-card">
                        <div class="prediction-header">
                            <h3 class="prediction-model-name">${funcPred.name || 'Model'}</h3>
                        </div>
                        <div class="prediction-content">
                            <div class="probability-info">
                                <span class="probability-label">Bug Probability:</span>
                                <span class="probability-value">${probabilityStr}%</span>
                            </div>
                            <div class="probability-bar-container">
                                <div class="probability-bar" style="width: ${barWidth}%; background: ${barColor};"></div>
                            </div>
                            <div class="prediction-badge-container">
                                <button class="prediction-btn ${labelLower}" type="button">
                                    ${label}
                                </button>
                            </div>
                        </div>
                    </div>
                `;
            }).join("");
            
            return `
                <div class="function-result-section">
                    <h4 class="function-name">Function: ${funcName}</h4>
                    ${funcGroup.class_name ? `<p class="function-class">Class: ${funcGroup.class_name}</p>` : ''}
                    <p class="function-lines">Lines: ${funcGroup.function_start_line} - ${funcGroup.function_end_line}</p>
                    <div class="function-predictions-grid">
                        ${predictionsHtml}
                    </div>
                </div>
            `;
        }).join("");
        
        // Store data for table view toggle
        window.currentResultsData = data;
        
        resultsContainer.innerHTML = `
            <div class="results-header">
                <h2>Function-Level Results for: ${data.filename || 'unknown'}</h2>
                <div class="view-toggle">
                    <button class="toggle-btn ${!useTableView ? 'active' : ''}" onclick="switchView('card')">
                        <span>📊</span> Card View
                    </button>
                    <button class="toggle-btn ${useTableView ? 'active' : ''}" onclick="switchView('table')">
                        <span>📋</span> Table View
                    </button>
                </div>
            </div>
            <div class="function-results-container">
                <p class="function-count-info">Found ${functionGroupsArray.length} function(s) with ${data.function_count || data.predictions.length} total prediction(s)</p>
                ${functionResultsHtml}
            </div>
        `;
        
        // If table view is requested, switch to it
        if (useTableView && typeof ResultsTable !== 'undefined') {
            setTimeout(() => {
                if (!window.resultsTable) {
                    window.resultsTable = new ResultsTable('results');
                }
                window.resultsTable.render(data);
            }, 100);
        }
        return;
    }
    
    // Handle table view (works for both file-level and function-level)
    if (useTableView && typeof ResultsTable !== 'undefined') {
        // Use advanced table view
        if (!window.resultsTable) {
            window.resultsTable = new ResultsTable('results');
        }
        window.resultsTable.render(data);
        return; // Exit early when using table view
    }
    
    // Card view with progress bars (matching image)
    const statsHtml = Object.entries(data.stats || {})
        .map(
            ([label, value]) => `
            <div class="code-metric-card">
                <div class="metric-label">${formatMetricLabel(label)}</div>
                <div class="metric-value">${Number(value).toFixed(2)}</div>
            </div>
        `
        )
        .join("");

    // Deduplicate predictions by model name for file-level card view
    const seenModelNames = new Set();
    const uniquePredictions = Object.values(data.predictions || {}).filter((prediction) => {
        const modelName = prediction.name || 'Unknown Model';
        if (seenModelNames.has(modelName)) {
            return false; // Skip duplicate
        }
        seenModelNames.add(modelName);
        return true;
    });
    
    const predictionsHtml = uniquePredictions
        .map((prediction) => {
            const probability = parseFloat(prediction.probability) * 100;
            const probabilityStr = probability.toFixed(2);
            const barWidth = Math.min(100, Math.max(0, probability));
            const barColor = probability >= 50 ? 'var(--error)' : probability >= 25 ? '#fbbf24' : 'var(--success)';
            
            return `
                <div class="prediction-card">
                    <div class="prediction-header">
                        <h3 class="prediction-model-name">${prediction.name}</h3>
                    </div>
                    <div class="prediction-content">
                        <div class="probability-info">
                            <span class="probability-label">Bug Probability:</span>
                            <span class="probability-value">${probabilityStr}%</span>
                        </div>
                        <div class="probability-bar-container">
                            <div class="probability-bar" style="width: ${barWidth}%; background: ${barColor};"></div>
                        </div>
                        <div class="prediction-badge-container">
                            <button class="prediction-btn ${prediction.label.toLowerCase()}" type="button">
                                ${prediction.label}
                            </button>
                        </div>
                    </div>
                </div>
            `;
        })
        .join("");

    // Add visualizations if available
    let visualizationsHtml = '';
    if (data.confusion_matrix) {
        visualizationsHtml += `<div id="confusion-matrix-container"></div>`;
    }
    if (data.feature_importance) {
        visualizationsHtml += `<div id="feature-importance-container"></div>`;
    }
    
    resultsContainer.innerHTML = `
        <div class="results-header">
            <h2>Results for: ${data.filename || fileInput.files[0].name}</h2>
            <div class="view-toggle">
                <button class="toggle-btn ${!useTableView ? 'active' : ''}" onclick="switchView('card')">
                    <span>📊</span> Card View
                </button>
                <button class="toggle-btn ${useTableView ? 'active' : ''}" onclick="switchView('table')">
                    <span>📋</span> Table View
                </button>
            </div>
        </div>
        <div class="code-metrics-section">
            <h3>Code Metrics</h3>
            <div class="code-metrics-grid">${statsHtml}</div>
        </div>
        <div class="predictions-section">
            <h3>Bug Probability Prediction</h3>
            <div class="predictions-grid">${predictionsHtml}</div>
        </div>
        ${visualizationsHtml}
    `;
    
    // Render visualizations
    if (data.confusion_matrix && typeof visualizations !== 'undefined') {
        visualizations.renderConfusionMatrix('confusion-matrix-container', data.confusion_matrix);
    }
    if (data.feature_importance && typeof visualizations !== 'undefined') {
        visualizations.renderFeatureImportance('feature-importance-container', data.feature_importance);
    }
}

function formatMetricLabel(label) {
    // Convert snake_case to Title Case
    return label
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
        .join(' ');
}

function switchView(viewMode) {
    window.currentViewMode = viewMode;
    // Try to get data from either location
    const lastData = window.lastPredictionData || window.currentResultsData;
    if (lastData) {
        renderResults(lastData);
    }
}

// Initialize view mode (default to card view)
window.currentViewMode = 'card';
