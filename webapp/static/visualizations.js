/**
 * Visualizations Component
 * Creates charts for confusion matrix, feature importance, and performance metrics
 */

class Visualizations {
    constructor() {
        this.charts = {};
    }

    renderConfusionMatrix(containerId, confusionMatrix, labels = ['CLEAN', 'BUGGY']) {
        const container = document.getElementById(containerId);
        if (!container) return;

        const [tn, fp, fn, tp] = confusionMatrix;
        const total = tn + fp + fn + tp;

        const html = `
            <div class="chart-container">
                <h3>Confusion Matrix</h3>
                <div class="confusion-matrix">
                    <table class="confusion-table">
                        <thead>
                            <tr>
                                <th></th>
                                <th>Predicted: ${labels[0]}</th>
                                <th>Predicted: ${labels[1]}</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <th>Actual: ${labels[0]}</th>
                                <td class="correct">${tn}<br><small>${((tn/total)*100).toFixed(1)}%</small></td>
                                <td class="incorrect">${fp}<br><small>${((fp/total)*100).toFixed(1)}%</small></td>
                            </tr>
                            <tr>
                                <th>Actual: ${labels[1]}</th>
                                <td class="incorrect">${fn}<br><small>${((fn/total)*100).toFixed(1)}%</small></td>
                                <td class="correct">${tp}<br><small>${((tp/total)*100).toFixed(1)}%</small></td>
                            </tr>
                        </tbody>
                    </table>
                    <div class="matrix-legend">
                        <div class="legend-item">
                            <span class="legend-color correct"></span>
                            <span>Correct Predictions</span>
                        </div>
                        <div class="legend-item">
                            <span class="legend-color incorrect"></span>
                            <span>Incorrect Predictions</span>
                        </div>
                    </div>
                </div>
            </div>
        `;

        container.innerHTML = html;
    }

    renderFeatureImportance(containerId, importanceData) {
        const container = document.getElementById(containerId);
        if (!container) return;

        // Sort features by importance
        const sorted = Object.entries(importanceData)
            .map(([name, value]) => ({ name, value }))
            .sort((a, b) => b.value - a.value)
            .slice(0, 10); // Top 10 features

        const maxValue = Math.max(...sorted.map(f => f.value));

        const barsHtml = sorted.map(feature => {
            const width = (feature.value / maxValue) * 100;
            return `
                <div class="feature-bar-row">
                    <div class="feature-name">${feature.name.replace(/_/g, ' ')}</div>
                    <div class="feature-bar-container">
                        <div class="feature-bar" style="width: ${width}%"></div>
                        <span class="feature-value">${feature.value.toFixed(4)}</span>
                    </div>
                </div>
            `;
        }).join('');

        const html = `
            <div class="chart-container">
                <h3>Feature Importance (Top 10)</h3>
                <div class="feature-importance-chart">
                    ${barsHtml}
                </div>
            </div>
        `;

        container.innerHTML = html;
    }

    renderPerformanceMetrics(containerId, metrics) {
        const container = document.getElementById(containerId);
        if (!container) return;

        const metricsList = [
            { label: 'Accuracy', value: metrics.accuracy, color: '#52c41a' },
            { label: 'Precision', value: metrics.precision, color: '#1890ff' },
            { label: 'Recall', value: metrics.recall, color: '#722ed1' },
            { label: 'F1-Score', value: metrics.f1, color: '#fa8c16' },
            { label: 'ROC-AUC', value: metrics.roc_auc, color: '#eb2f96' }
        ];

        const barsHtml = metricsList.map(metric => {
            const width = metric.value * 100;
            return `
                <div class="metric-bar-row">
                    <div class="metric-label">${metric.label}</div>
                    <div class="metric-bar-container">
                        <div class="metric-bar" style="width: ${width}%; background-color: ${metric.color}"></div>
                        <span class="metric-value">${(metric.value * 100).toFixed(2)}%</span>
                    </div>
                </div>
            `;
        }).join('');

        const html = `
            <div class="chart-container">
                <h3>Model Performance Metrics</h3>
                <div class="performance-chart">
                    ${barsHtml}
                </div>
            </div>
        `;

        container.innerHTML = html;
    }

    renderROCCurve(containerId, fpr, tpr, auc) {
        const container = document.getElementById(containerId);
        if (!container) return;

        // Simple SVG-based ROC curve
        const points = fpr.map((x, i) => ({ x: x * 100, y: (1 - tpr[i]) * 100 }));
        const pathData = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x},${p.y}`).join(' ');

        const html = `
            <div class="chart-container">
                <h3>ROC Curve (AUC: ${auc.toFixed(4)})</h3>
                <div class="roc-curve-container">
                    <svg viewBox="0 0 100 100" class="roc-svg">
                        <!-- Diagonal line (random classifier) -->
                        <line x1="0" y1="100" x2="100" y2="0" stroke="#ccc" stroke-width="0.5" stroke-dasharray="2,2"/>
                        <!-- ROC curve -->
                        <path d="${pathData} L 100,0 L 0,0 Z" fill="rgba(24, 144, 255, 0.2)" stroke="#1890ff" stroke-width="1"/>
                        <!-- Axes -->
                        <line x1="0" y1="100" x2="100" y2="100" stroke="#333" stroke-width="0.5"/>
                        <line x1="0" y1="100" x2="0" y2="0" stroke="#333" stroke-width="0.5"/>
                        <!-- Labels -->
                        <text x="50" y="105" text-anchor="middle" font-size="3">False Positive Rate</text>
                        <text x="-50" y="50" text-anchor="middle" font-size="3" transform="rotate(-90 -50 50)">True Positive Rate</text>
                    </svg>
                </div>
            </div>
        `;

        container.innerHTML = html;
    }
}

// Global instance
const visualizations = new Visualizations();

