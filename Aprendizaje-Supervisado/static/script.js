const API_URL = window.location.origin;

let authToken = localStorage.getItem('authToken');
let currentUser = JSON.parse(localStorage.getItem('currentUser') || 'null');
let currentDataset = null;
let executedModels = [];

if (authToken && currentUser) {
    showDashboard();
    loadDatasets();
}

function showNotification(message, type = 'success') {
    const notification = document.getElementById('notification');
    notification.textContent = message;
    notification.className = `notification ${type} show`;
    
    setTimeout(() => {
        notification.classList.remove('show');
    }, 3000);
}

function showLogin() {
    document.getElementById('login-form').classList.add('active');
    document.getElementById('register-form').classList.remove('active');
}

function showRegister() {
    document.getElementById('register-form').classList.add('active');
    document.getElementById('login-form').classList.remove('active');
}

async function handleRegister(event) {
    event.preventDefault();
    
    const name = document.getElementById('register-name').value;
    const email = document.getElementById('register-email').value;
    const student_code = document.getElementById('register-code').value;
    const password = document.getElementById('register-password').value;
    
    try {
        const response = await fetch(`${API_URL}/api/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, email, student_code, password })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showNotification('Registro exitoso. Por favor inicia sesión.', 'success');
            showLogin();
        } else {
            showNotification(data.error || 'Error en el registro', 'error');
        }
    } catch (error) {
        showNotification('Error de conexión: ' + error.message, 'error');
    }
}

async function handleLogin(event) {
    event.preventDefault();
    
    const email = document.getElementById('login-email').value;
    const password = document.getElementById('login-password').value;
    
    try {
        const response = await fetch(`${API_URL}/api/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            authToken = data.token;
            currentUser = data.user;
            localStorage.setItem('authToken', authToken);
            localStorage.setItem('currentUser', JSON.stringify(currentUser));
            
            showDashboard();
            loadDatasets();
            showNotification('Bienvenido ' + currentUser.name, 'success');
        } else {
            showNotification(data.error || 'Error en el login', 'error');
        }
    } catch (error) {
        showNotification('Error de conexión: ' + error.message, 'error');
    }
}

function logout() {
    authToken = null;
    currentUser = null;
    localStorage.removeItem('authToken');
    localStorage.removeItem('currentUser');
    
    document.getElementById('auth-section').style.display = 'flex';
    document.getElementById('dashboard-section').style.display = 'none';
    showNotification('Sesión cerrada', 'success');
}

function showDashboard() {
    document.getElementById('auth-section').style.display = 'none';
    document.getElementById('dashboard-section').style.display = 'block';
    document.getElementById('user-name').textContent = currentUser.name;
}

function showSection(section) {
    document.querySelectorAll('.content-section').forEach(s => s.classList.remove('active'));
    document.getElementById(`${section}-section`).classList.add('active');
    
    document.querySelectorAll('.menu a').forEach(a => a.classList.remove('active'));
    event.target.classList.add('active');
    
    if (section === 'regression' || section === 'logistic' || section === 'tree') {
        loadDatasetsForModel(section);
    }
    
    if (section === 'report') {
        updateReportModels();
    }
}

async function loadDatasets() {
    try {
        const response = await fetch(`${API_URL}/api/datasets`, {
            headers: { 'Authorization': `Bearer ${authToken}` }
        });
        
        const data = await response.json();
        
        if (response.ok) {
            const list = document.getElementById('datasets-list');
            if (data.datasets.length === 0) {
                list.innerHTML = '<p>No hay datasets disponibles. Carga o genera datos para empezar.</p>';
            } else {
                list.innerHTML = data.datasets.map(ds => `
                    <div style="padding: 10px; border-bottom: 1px solid #eee;">
                        <strong>${ds.name}</strong> - ${new Date(ds.created_at).toLocaleString()}
                    </div>
                `).join('');
            }
        }
    } catch (error) {
        showNotification('Error al cargar datasets: ' + error.message, 'error');
    }
}

async function uploadFile() {
    const fileInput = document.getElementById('file-upload');
    const file = fileInput.files[0];
    
    if (!file) {
        showNotification('Selecciona un archivo', 'error');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch(`${API_URL}/api/upload-data`, {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${authToken}` },
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showNotification('Datos cargados exitosamente', 'success');
            currentDataset = { id: data.dataset_id, columns: data.columns, preview: data.preview };
            showDataPreview(data);
            loadDatasets();
        } else {
            showNotification(data.error || 'Error al cargar archivo', 'error');
        }
    } catch (error) {
        showNotification('Error de conexión: ' + error.message, 'error');
    }
}

async function generateRandomData() {
    const rows = document.getElementById('random-rows').value;
    const columns = document.getElementById('random-cols').value;
    
    try {
        const response = await fetch(`${API_URL}/api/generate-random-data`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${authToken}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ rows: parseInt(rows), columns: parseInt(columns) })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showNotification('Datos generados exitosamente', 'success');
            currentDataset = { id: data.dataset_id, columns: data.columns, preview: data.preview };
            showDataPreview(data);
            loadDatasets();
        } else {
            showNotification(data.error || 'Error al generar datos', 'error');
        }
    } catch (error) {
        showNotification('Error de conexión: ' + error.message, 'error');
    }
}

function showDataPreview(data) {
    const preview = document.getElementById('data-preview');
    preview.style.display = 'block';
    
    const content = document.getElementById('preview-content');
    
    const table = `
        <p><strong>Filas:</strong> ${data.rows} | <strong>Columnas:</strong> ${data.columns.join(', ')}</p>
        <table>
            <thead>
                <tr>${data.columns.map(col => `<th>${col}</th>`).join('')}</tr>
            </thead>
            <tbody>
                ${data.preview.slice(0, 10).map(row => `
                    <tr>${data.columns.map(col => `<td>${row[col]}</td>`).join('')}</tr>
                `).join('')}
            </tbody>
        </table>
    `;
    
    content.innerHTML = table;
}

async function loadDatasetsForModel(modelType) {
    try {
        const response = await fetch(`${API_URL}/api/datasets`, {
            headers: { 'Authorization': `Bearer ${authToken}` }
        });
        
        const data = await response.json();
        
        if (response.ok) {
            const selectId = `${modelType}-dataset`;
            const select = document.getElementById(selectId);
            
            select.innerHTML = '<option value="">Selecciona un dataset</option>' + 
                data.datasets.map(ds => `<option value="${ds.id}">${ds.name}</option>`).join('');
            
            select.onchange = () => loadDatasetColumns(modelType, select.value);
        }
    } catch (error) {
        showNotification('Error al cargar datasets: ' + error.message, 'error');
    }
}

async function loadDatasetColumns(modelType, datasetId) {
    if (!datasetId) return;
    
    try {
        const response = await fetch(`${API_URL}/api/datasets`, {
            headers: { 'Authorization': `Bearer ${authToken}` }
        });
        
        const data = await response.json();
        
        if (currentDataset && currentDataset.id == datasetId) {
            updateColumnSelectors(modelType, currentDataset.columns);
        }
    } catch (error) {
        showNotification('Error: ' + error.message, 'error');
    }
}

function updateColumnSelectors(modelType, columns) {
    const targetSelect = document.getElementById(`${modelType}-target`);
    const featuresDiv = document.getElementById(`${modelType}-features`);
    
    targetSelect.innerHTML = '<option value="">Selecciona variable objetivo</option>' + 
        columns.map(col => `<option value="${col}">${col}</option>`).join('');
    
    featuresDiv.innerHTML = columns.map(col => `
        <label style="display: block; margin: 5px 0;">
            <input type="checkbox" name="${modelType}-feature" value="${col}"> ${col}
        </label>
    `).join('');
}

async function runRegression() {
    const datasetId = document.getElementById('reg-dataset').value;
    const modelType = document.getElementById('reg-model').value;
    const targetColumn = document.getElementById('reg-target').value;
    const featureCheckboxes = document.querySelectorAll('input[name="reg-feature"]:checked');
    const featureColumns = Array.from(featureCheckboxes).map(cb => cb.value);
    
    if (!datasetId || !targetColumn) {
        showNotification('Selecciona dataset y variable objetivo', 'error');
        return;
    }
    
    try {
        const response = await fetch(`${API_URL}/api/regression`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${authToken}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ dataset_id: datasetId, model_type: modelType, target_column: targetColumn, feature_columns: featureColumns })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showNotification('Modelo ejecutado exitosamente', 'success');
            displayRegressionResults(data);
            executedModels.push(data);
        } else {
            showNotification(data.error || 'Error al ejecutar modelo', 'error');
        }
    } catch (error) {
        showNotification('Error de conexión: ' + error.message, 'error');
    }
}

function displayRegressionResults(data) {
    const results = document.getElementById('reg-results');
    const content = document.getElementById('reg-results-content');
    
    results.style.display = 'block';
    
    const metricsHtml = `
        <div class="metrics-grid">
            <div class="metric-item">
                <div class="metric-label">R² (Test)</div>
                <div class="metric-value">${data.metrics.r2_test.toFixed(4)}</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">MSE (Test)</div>
                <div class="metric-value">${data.metrics.mse_test.toFixed(4)}</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">RMSE (Test)</div>
                <div class="metric-value">${data.metrics.rmse_test.toFixed(4)}</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">MAE (Test)</div>
                <div class="metric-value">${data.metrics.mae_test.toFixed(4)}</div>
            </div>
        </div>
        ${data.intercept !== null ? `<p><strong>Intercepto:</strong> ${data.intercept.toFixed(4)}</p>` : ''}
        <img src="data:image/png;base64,${data.plot}" class="results-image" alt="Gráfica de resultados">
    `;
    
    content.innerHTML = metricsHtml;
}

async function runLogistic() {
    const datasetId = document.getElementById('log-dataset').value;
    const targetColumn = document.getElementById('log-target').value;
    const featureCheckboxes = document.querySelectorAll('input[name="log-feature"]:checked');
    const featureColumns = Array.from(featureCheckboxes).map(cb => cb.value);
    
    if (!datasetId || !targetColumn) {
        showNotification('Selecciona dataset y variable objetivo', 'error');
        return;
    }
    
    try {
        const response = await fetch(`${API_URL}/api/logistic`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${authToken}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ dataset_id: datasetId, target_column: targetColumn, feature_columns: featureColumns })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showNotification('Modelo ejecutado exitosamente', 'success');
            displayLogisticResults(data);
            executedModels.push(data);
        } else {
            showNotification(data.error || 'Error al ejecutar modelo', 'error');
        }
    } catch (error) {
        showNotification('Error de conexión: ' + error.message, 'error');
    }
}

function displayLogisticResults(data) {
    const results = document.getElementById('log-results');
    const content = document.getElementById('log-results-content');
    
    results.style.display = 'block';
    
    const metricsHtml = `
        <div class="metrics-grid">
            <div class="metric-item">
                <div class="metric-label">Accuracy</div>
                <div class="metric-value">${data.metrics.accuracy.toFixed(4)}</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Precision</div>
                <div class="metric-value">${data.metrics.precision.toFixed(4)}</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Recall</div>
                <div class="metric-value">${data.metrics.recall.toFixed(4)}</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">F1-Score</div>
                <div class="metric-value">${data.metrics.f1_score.toFixed(4)}</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">AUC</div>
                <div class="metric-value">${data.metrics.auc.toFixed(4)}</div>
            </div>
        </div>
        <p><strong>Clases:</strong> ${data.classes.join(', ')}</p>
        <img src="data:image/png;base64,${data.plot}" class="results-image" alt="Curva ROC">
    `;
    
    content.innerHTML = metricsHtml;
}

async function runDecisionTree() {
    const datasetId = document.getElementById('tree-dataset').value;
    const modelType = document.getElementById('tree-model').value;
    const targetColumn = document.getElementById('tree-target').value;
    const maxDepth = document.getElementById('tree-depth').value || null;
    const featureCheckboxes = document.querySelectorAll('input[name="tree-feature"]:checked');
    const featureColumns = Array.from(featureCheckboxes).map(cb => cb.value);
    
    if (!datasetId || !targetColumn) {
        showNotification('Selecciona dataset y variable objetivo', 'error');
        return;
    }
    
    try {
        const response = await fetch(`${API_URL}/api/decision-tree`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${authToken}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                dataset_id: datasetId, 
                model_type: modelType, 
                target_column: targetColumn, 
                feature_columns: featureColumns,
                max_depth: maxDepth ? parseInt(maxDepth) : null
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showNotification('Árbol de decisión generado exitosamente', 'success');
            displayTreeResults(data);
            executedModels.push(data);
        } else {
            showNotification(data.error || 'Error al ejecutar modelo', 'error');
        }
    } catch (error) {
        showNotification('Error de conexión: ' + error.message, 'error');
    }
}

function displayTreeResults(data) {
    const results = document.getElementById('tree-results');
    const content = document.getElementById('tree-results-content');
    
    results.style.display = 'block';
    
    const metricsKeys = Object.keys(data.metrics);
    const metricsHtml = `
        <div class="metrics-grid">
            ${metricsKeys.map(key => `
                <div class="metric-item">
                    <div class="metric-label">${key}</div>
                    <div class="metric-value">${data.metrics[key].toFixed ? data.metrics[key].toFixed(4) : data.metrics[key]}</div>
                </div>
            `).join('')}
        </div>
        ${data.algorithm_info ? `<p><em>${data.algorithm_info}</em></p>` : ''}
        <h4>Visualización del Árbol</h4>
        <img src="data:image/png;base64,${data.plot}" class="results-image" alt="Árbol de decisión">
        <h4>Importancia de Variables</h4>
        <img src="data:image/png;base64,${data.importance_plot}" class="results-image" alt="Importancia de variables">
    `;
    
    content.innerHTML = metricsHtml;
}

function updateReportModels() {
    const container = document.getElementById('report-models');
    
    if (executedModels.length === 0) {
        container.innerHTML = '<p>No hay modelos ejecutados. Ejecuta al menos un modelo antes de generar un reporte.</p>';
        return;
    }
    
    container.innerHTML = executedModels.map((model, idx) => `
        <label style="display: block; margin: 10px 0; padding: 10px; background: #f5f7fa; border-radius: 6px;">
            <input type="checkbox" name="report-model" value="${idx}" checked> 
            <strong>${model.model_type}</strong>
        </label>
    `).join('');
}

async function generateReport() {
    const selectedCheckboxes = document.querySelectorAll('input[name="report-model"]:checked');
    const selectedIndices = Array.from(selectedCheckboxes).map(cb => parseInt(cb.value));
    
    if (selectedIndices.length === 0) {
        showNotification('Selecciona al menos un modelo para el reporte', 'error');
        return;
    }
    
    const selectedModels = selectedIndices.map(idx => executedModels[idx]);
    
    try {
        const response = await fetch(`${API_URL}/api/generate-report`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${authToken}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ report_data: { models: selectedModels } })
        });
        
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `reporte_ml_${new Date().getTime()}.pdf`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
            
            showNotification('Reporte PDF generado exitosamente', 'success');
        } else {
            showNotification('Error al generar reporte', 'error');
        }
    } catch (error) {
        showNotification('Error de conexión: ' + error.message, 'error');
    }
}
