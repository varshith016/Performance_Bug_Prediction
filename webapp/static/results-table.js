/**
 * Advanced Results Table Component
 * Provides sorting, filtering, and table view for prediction results
 */

class ResultsTable {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.data = [];
        this.sortColumn = null;
        this.sortDirection = 'asc';
        this.filterValue = 'all'; // 'all', 'buggy', 'clean'
    }

    render(data) {
        // Store original data for filename access
        this.originalData = data;
        this.data = Array.isArray(data) ? data : this._convertToArray(data);
        
        if (this.data.length === 0) {
            this.container.innerHTML = '<p>No results to display</p>';
            return;
        }

        const filename = (this.originalData && this.originalData.filename) || 'unknown';
        const html = `
            <div class="results-header">
                <h2>Results for: ${filename}</h2>
                <div class="view-toggle">
                    <button class="toggle-btn" onclick="switchView('card')">
                        <span>📊</span> Card View
                    </button>
                    <button class="toggle-btn active" onclick="switchView('table')">
                        <span>📋</span> Table View
                    </button>
                </div>
            </div>
            <div class="results-table-container">
                <div class="table-controls">
                    <div class="filter-controls">
                        <label>Filter:</label>
                        <select id="results-filter" class="form-input">
                            <option value="all">All Results</option>
                            <option value="buggy">Buggy Only</option>
                            <option value="clean">Clean Only</option>
                        </select>
                    </div>
                    <div class="sort-controls">
                        <label>Sort by:</label>
                        <select id="results-sort" class="form-input">
                            <option value="filename">File Name</option>
                            <option value="probability">Probability</option>
                            <option value="model">Model</option>
                        </select>
                    </div>
                </div>
                <div class="table-wrapper">
                    <table class="results-table">
                        <thead>
                            <tr>
                                <th onclick="if(window.resultsTable) window.resultsTable.sort('filename')">
                                    ${this.originalData && this.originalData.function_level ? 'Function' : 'File Name'} <span class="sort-indicator"></span>
                                </th>
                                ${this.originalData && this.originalData.function_level ? `
                                <th onclick="if(window.resultsTable) window.resultsTable.sort('class_name')">
                                    Class <span class="sort-indicator"></span>
                                </th>
                                <th onclick="if(window.resultsTable) window.resultsTable.sort('function_lines')">
                                    Lines <span class="sort-indicator"></span>
                                </th>
                                ` : ''}
                                <th onclick="if(window.resultsTable) window.resultsTable.sort('model')">
                                    Model <span class="sort-indicator"></span>
                                </th>
                                <th onclick="if(window.resultsTable) window.resultsTable.sort('probability')">
                                    Probability <span class="sort-indicator"></span>
                                </th>
                                <th onclick="if(window.resultsTable) window.resultsTable.sort('label')">
                                    Prediction <span class="sort-indicator"></span>
                                </th>
                            </tr>
                        </thead>
                        <tbody id="results-table-body">
                            ${this._renderRows()}
                        </tbody>
                    </table>
                </div>
                <div class="table-summary">
                    <p>Showing <strong>${this._getFilteredData().length}</strong> of <strong>${this.data.length}</strong> results</p>
                </div>
            </div>
        `;

        this.container.innerHTML = html;
        this._attachEventListeners();
    }

    _convertToArray(data) {
        // Convert prediction results to array format
        const results = [];
        const seenModels = new Set(); // Track seen models to prevent duplicates
        
        // Handle function-level results
        if (data.function_level && Array.isArray(data.predictions)) {
            data.predictions.forEach((funcPred) => {
                const modelName = funcPred.name || 'Unknown Model';
                const funcName = funcPred.function_name || 'Unknown';
                // Use function name + model name as unique key to prevent duplicates
                // Backend already deduplicates by model name, so we just need to deduplicate per function
                const uniqueKey = `${data.filename || 'unknown'}_${funcName}_${modelName}`;
                
                if (!seenModels.has(uniqueKey)) {
                    seenModels.add(uniqueKey);
                    results.push({
                        filename: data.filename || 'unknown',
                        function_name: funcName,
                        class_name: funcPred.class_name || '',
                        function_lines: `${funcPred.function_start_line || 'N/A'}-${funcPred.function_end_line || 'N/A'}`,
                        model: modelName,
                        probability: funcPred.probability || 0,
                        label: funcPred.label || 'CLEAN',
                        modelId: funcPred.model_id || ''
                    });
                }
            });
            return results;
        }
        
        // Handle file-level results
        if (data.predictions && typeof data.predictions === 'object' && !Array.isArray(data.predictions)) {
            Object.entries(data.predictions).forEach(([modelId, prediction]) => {
                // Use model name as unique identifier (backend already deduplicates by name)
                const modelName = prediction.name || modelId;
                const uniqueKey = `${data.filename || 'unknown'}_${modelName}`;
                
                if (!seenModels.has(uniqueKey)) {
                    seenModels.add(uniqueKey);
                    results.push({
                        filename: data.filename || 'unknown',
                        model: modelName,
                        probability: prediction.probability || 0,
                        label: prediction.label || 'CLEAN',
                        modelId: modelId
                    });
                }
            });
        }
        
        return results;
    }

    _getFilteredData() {
        let filtered = [...this.data];
        
        // Apply filter
        if (this.filterValue === 'buggy') {
            filtered = filtered.filter(r => r.label === 'BUGGY');
        } else if (this.filterValue === 'clean') {
            filtered = filtered.filter(r => r.label === 'CLEAN');
        }
        
        // Apply sort
        if (this.sortColumn) {
            filtered.sort((a, b) => {
                let aVal = a[this.sortColumn];
                let bVal = b[this.sortColumn];
                
                // Handle different data types
                if (typeof aVal === 'number' && typeof bVal === 'number') {
                    return this.sortDirection === 'asc' ? aVal - bVal : bVal - aVal;
                }
                
                aVal = String(aVal || '').toLowerCase();
                bVal = String(bVal || '').toLowerCase();
                
                if (this.sortDirection === 'asc') {
                    return aVal.localeCompare(bVal);
                } else {
                    return bVal.localeCompare(aVal);
                }
            });
        }
        
        return filtered;
    }

    _renderRows() {
        const filtered = this._getFilteredData();
        
        const isFunctionLevel = this.originalData && this.originalData.function_level;
        const colCount = isFunctionLevel ? 6 : 4;
        
        if (filtered.length === 0) {
            return `<tr><td colspan="${colCount}" class="no-results">No results match the filter</td></tr>`;
        }
        
        return filtered.map(row => `
            <tr>
                <td>${this._escapeHtml(row.filename)}</td>
                <td>${this._escapeHtml(row.model)}</td>
                <td>
                    <div class="probability-cell">
                        <span class="probability-value">${(row.probability * 100).toFixed(2)}%</span>
                        <div class="probability-bar-mini">
                            <div class="probability-bar-fill" 
                                 style="width: ${(row.probability * 100).toFixed(2)}%; background: ${row.probability >= 0.5 ? 'var(--error)' : 'var(--success)'};"></div>
                        </div>
                    </div>
                </td>
                <td>
                    <button class="prediction-btn ${row.label.toLowerCase()}" type="button">
                        ${row.label}
                    </button>
                </td>
            </tr>
        `).join('');
    }

    sort(column) {
        if (this.sortColumn === column) {
            this.sortDirection = this.sortDirection === 'asc' ? 'desc' : 'asc';
        } else {
            this.sortColumn = column;
            this.sortDirection = 'asc';
        }
        this._updateTable();
    }

    filter(value) {
        this.filterValue = value;
        this._updateTable();
    }

    _updateTable() {
        const tbody = document.getElementById('results-table-body');
        if (tbody) {
            tbody.innerHTML = this._renderRows();
        }
        this._updateSortIndicators();
        this._updateSummary();
    }

    _updateSortIndicators() {
        document.querySelectorAll('.sort-indicator').forEach(indicator => {
            indicator.textContent = '';
        });
        
        if (this.sortColumn) {
            const headers = document.querySelectorAll('th');
            headers.forEach(header => {
                if (header.textContent.includes(this._getColumnName(this.sortColumn))) {
                    const indicator = header.querySelector('.sort-indicator');
                    if (indicator) {
                        indicator.textContent = this.sortDirection === 'asc' ? ' ↑' : ' ↓';
                    }
                }
            });
        }
    }

    _getColumnName(column) {
        const mapping = {
            'filename': 'File Name',
            'model': 'Model',
            'probability': 'Probability',
            'label': 'Prediction'
        };
        return mapping[column] || column;
    }

    _updateSummary() {
        const summary = document.querySelector('.table-summary p');
        if (summary) {
            summary.innerHTML = `Showing <strong>${this._getFilteredData().length}</strong> of <strong>${this.data.length}</strong> results`;
        }
    }

    _attachEventListeners() {
        const filterSelect = document.getElementById('results-filter');
        const sortSelect = document.getElementById('results-sort');
        
        if (filterSelect) {
            filterSelect.addEventListener('change', (e) => {
                this.filter(e.target.value);
            });
        }
        
        if (sortSelect) {
            sortSelect.addEventListener('change', (e) => {
                this.sort(e.target.value);
            });
        }
    }

    viewDetails(modelId) {
        // Show detailed view for a specific result
        console.log('View details for:', modelId);
        // Can be extended to show modal or navigate to detail page
    }

    _escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Global instance
window.resultsTable = null;

