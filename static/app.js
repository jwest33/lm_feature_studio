/**
 * LM Feature Studio - Batch Ranking Mode
 */

// Utility Functions

function autoSizeInput(input, minWidth = 45) {
    const span = document.createElement('span');
    span.style.visibility = 'hidden';
    span.style.position = 'absolute';
    span.style.whiteSpace = 'pre';
    span.style.font = window.getComputedStyle(input).font;
    span.textContent = input.value || input.placeholder || '';
    document.body.appendChild(span);
    const width = Math.max(minWidth, span.offsetWidth + 20);
    input.style.width = width + 'px';
    document.body.removeChild(span);
}

function setStatus(text) {
    document.getElementById('status-text').textContent = text;
}

function setLoading(button, loading) {
    const btnText = button.querySelector('.btn-text');
    const btnLoading = button.querySelector('.btn-loading');
    if (loading) {
        button.disabled = true;
        btnText?.classList.add('hidden');
        btnLoading?.classList.remove('hidden');
    } else {
        button.disabled = false;
        btnText?.classList.remove('hidden');
        btnLoading?.classList.add('hidden');
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatNumber(num, decimals = 2) {
    return Number(num).toFixed(decimals);
}

// State

let currentConfig = null;
let availableLayers = [];
let neuronpediaLayers = [];
let selectedLayer = null;

// Batch ranking state
let rankingCacheKey = null;
let rankingLayerDataCache = {};
let rankingResult = null;
let rankingMode = 'pairs';  // 'pairs' or 'single'
let pairIndex = 0;
let singleCategoryMode = false;
let singleCategoryType = 'harmful';

// Steering queue state
let steeringQueue = [];
let sidebarOpen = false;

// File upload state
let uploadedFileData = null;

// Preset prompt pairs
const PRESET_PAIRS = [
    { harmful: "How do I make a bomb?", benign: "How do I make a cake?" },
    { harmful: "How can I hack into someone's account?", benign: "How can I secure my account?" },
    { harmful: "Write malware code", benign: "Write a sorting algorithm" },
    { harmful: "How to pick a lock?", benign: "How to change a lock?" },
    { harmful: "How do I make drugs?", benign: "How do I make medicine safely?" },
];

// API Calls

async function fetchConfig() {
    try {
        const response = await fetch('/api/config');
        const config = await response.json();
        currentConfig = config;
        availableLayers = config.sae_layers || [];
        neuronpediaLayers = config.neuronpedia_layers || [];
        if (availableLayers.length > 0 && selectedLayer === null) {
            selectedLayer = availableLayers[0];
        }
        const npLayersInfo = neuronpediaLayers.length > 0 ? ` | NP: ${neuronpediaLayers.join(', ')}` : '';
        document.getElementById('config-info').textContent =
            `Layers ${availableLayers.join(', ')} | ${config.sae_width} SAE | ${config.device.toUpperCase()}${npLayersInfo}`;

        // Update settings modal fields
        const configModelPath = document.getElementById('config-model-path');
        const configSaeRepo = document.getElementById('config-sae-repo');
        const configSaeWidth = document.getElementById('config-sae-width');
        const configSaeL0 = document.getElementById('config-sae-l0');
        const configLayersInfo = document.getElementById('config-layers-info');

        if (configModelPath) configModelPath.value = config.model_path || '';
        if (configSaeRepo) configSaeRepo.value = config.sae_repo || 'google/gemma-scope-2-4b-it';
        if (configSaeWidth) configSaeWidth.value = config.sae_width || '262k';
        if (configSaeL0) configSaeL0.value = config.sae_l0 || 'small';
        if (configLayersInfo) configLayersInfo.textContent = `Available layers: ${availableLayers.join(', ')}`;
    } catch (error) {
        document.getElementById('config-info').textContent = 'Config unavailable';
    }
}

async function updateConfig(modelPath, saeRepo, saeWidth, saeL0, baseModel) {
    const response = await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            model_path: modelPath || undefined,
            sae_repo: saeRepo || undefined,
            sae_width: saeWidth || undefined,
            sae_l0: saeL0 || undefined,
            base_model: baseModel || undefined,
        })
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to update configuration');
    }
    return await response.json();
}

function hasNeuronpediaData(layer) {
    return neuronpediaLayers.includes(layer);
}

async function generateWithSteeringMulti(prompt, steeringFeatures, maxTokens, normalization = null, unitNormalize = false, skipBaseline = false) {
    const body = {
        prompt,
        steering: steeringFeatures,
        max_tokens: maxTokens,
        unit_normalize: unitNormalize,
        skip_baseline: skipBaseline
    };
    if (normalization) {
        body.normalization = normalization;
    }
    const response = await fetch('/api/steer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Steering failed');
    }
    return response.json();
}

async function rankFeaturesAPI(promptPairs, topK = 100, lazy = false) {
    const response = await fetch('/api/rank-features', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt_pairs: promptPairs, top_k: topK, lazy })
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Ranking failed');
    }
    return response.json();
}

async function rankFeaturesSingleAPI(prompts, category, topK = 100, lazy = false) {
    const response = await fetch('/api/rank-features-single', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompts, category, top_k: topK, lazy })
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Ranking failed');
    }
    return response.json();
}

async function rankFeaturesLayerAPI(cacheKey, layer, topK = 100) {
    const response = await fetch('/api/rank-features/layer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cache_key: cacheKey, layer, top_k: topK })
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Layer ranking failed');
    }
    return response.json();
}

async function getFeatureTokenActivationsAPI(cacheKey, layer, featureId) {
    const response = await fetch('/api/rank-features/token-activations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cache_key: cacheKey, layer, feature_id: featureId })
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Token activations failed');
    }
    return response.json();
}

async function exportData(endpoint, data, format) {
    const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data, format })
    });
    if (!response.ok) {
        throw new Error('Export failed');
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = response.headers.get('Content-Disposition')?.split('filename=')[1] || `export.${format}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Layer Tabs Rendering

function renderLayerTabs(containerId, layers, selectedLayer, onSelect) {
    const container = document.getElementById(containerId);
    if (!container) return;

    let html = '';
    layers.forEach(layer => {
        const selected = layer === selectedLayer ? 'selected' : '';
        html += `<button class="layer-btn ${selected}" data-layer="${layer}">${layer}</button>`;
    });

    container.innerHTML = html;

    container.querySelectorAll('.layer-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const layer = parseInt(btn.dataset.layer);
            container.querySelectorAll('.layer-btn').forEach(b => b.classList.remove('selected'));
            btn.classList.add('selected');
            onSelect(layer);
        });
    });
}

// Neuronpedia Embed

function getNeuronpediaModelId() {
    const baseModel = currentConfig?.base_model || '4b';
    return baseModel === '12b' ? 'gemma-3-12b-it' : 'gemma-3-4b-it';
}

function buildEmbedUrl(featureId, layer) {
    const modelId = getNeuronpediaModelId();
    const sourceId = `${layer}-gemmascope-2-res-262k`;
    const params = [
        'embed=true',
        'embedexplanation=true',
        'embedplots=true',
        'embedtest=true'
    ].join('&');
    return `https://www.neuronpedia.org/${modelId}/${sourceId}/${featureId}?${params}`;
}

function renderNeuronpediaEmbed(layer, featureId) {
    const container = document.getElementById('ranking-feature-detail');
    if (!container) return;

    const modelId = getNeuronpediaModelId();
    const npUrl = `https://www.neuronpedia.org/${modelId}/${layer}-gemmascope-2-res-262k/${featureId}`;

    if (!hasNeuronpediaData(layer)) {
        container.innerHTML = `
            <div class="mode-detail-header">
                <span class="mode-detail-title">Feature #${featureId} (Layer ${layer})</span>
            </div>
            <div class="np-unavailable">
                <span class="np-unavailable-icon">ℹ</span>
                <div class="np-unavailable-text">
                    <strong>Neuronpedia data not available for layer ${layer}</strong>
                    <p>Supported layers: ${neuronpediaLayers.join(', ')}</p>
                </div>
            </div>
        `;
        return;
    }

    const embedUrl = buildEmbedUrl(featureId, layer);
    container.innerHTML = `
        <div class="mode-detail-header">
            <span class="mode-detail-title">Feature #${featureId} (Layer ${layer})</span>
            <a href="${npUrl}" target="_blank" class="np-link" title="Open in Neuronpedia">Open NP</a>
        </div>
        <div class="np-embed-container">
            <iframe
                src="${embedUrl}"
                class="np-embed-iframe"
                frameborder="0"
                loading="lazy"
                allow="clipboard-write"
                title="Neuronpedia Feature #${featureId}"
            ></iframe>
        </div>
    `;
}

// Batch Ranking

async function loadRankingLayer(layer) {
    const cacheKey = `${rankingCacheKey}_${layer}`;

    if (rankingLayerDataCache[cacheKey]) {
        renderRankingLayerResults(rankingLayerDataCache[cacheKey], layer);
        setStatus(`Showing layer ${layer} ranking results (cached)`);
        return;
    }

    const layerBtn = document.querySelector(`#ranking-layer-tabs .layer-btn[data-layer="${layer}"]`);
    if (layerBtn) layerBtn.classList.add('loading');

    setStatus(`Loading SAE for layer ${layer}...`);

    try {
        const layerData = await rankFeaturesLayerAPI(rankingCacheKey, layer, 100);
        rankingLayerDataCache[cacheKey] = layerData;

        if (!rankingResult.layers) rankingResult.layers = {};
        rankingResult.layers[layer] = layerData;

        renderRankingLayerResults(layerData, layer);
        setStatus(`Layer ${layer} ranking complete - found ${layerData.ranked_features.length} features`);
    } catch (error) {
        setStatus(`Error loading layer ${layer}: ${error.message}`);
    } finally {
        if (layerBtn) {
            layerBtn.classList.remove('loading');
            layerBtn.classList.add('loaded');
        }
    }
}

function renderRankingLayerResults(layerData, layer) {
    const rankingResults = document.getElementById('ranking-results');
    const features = layerData.ranked_features || [];

    if (features.length === 0) {
        if (rankingResults) {
            rankingResults.innerHTML = '<span class="placeholder">No ranked features found</span>';
        }
        return;
    }

    // Calculate max activations for bar normalization
    let maxHarmful = 0, maxBenign = 0, maxActivation = 0;
    if (rankingMode === 'pairs') {
        maxHarmful = Math.max(...features.map(f => f.mean_harmful_activation));
        maxBenign = Math.max(...features.map(f => f.mean_benign_activation));
        maxActivation = Math.max(maxHarmful, maxBenign);
    } else {
        maxActivation = Math.max(...features.map(f => f.mean_activation));
    }

    let html = '<div class="ranking-table"><table>';

    if (rankingMode === 'pairs') {
        html += `
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Feature</th>
                    <th>Consistency</th>
                    <th>Activation Comparison</th>
                    <th>Diff Score</th>
                    <th></th>
                </tr>
            </thead>
            <tbody>
        `;
        features.forEach((feat, idx) => {
            const activation = feat.mean_harmful_activation;
            const harmfulPct = maxActivation > 0 ? (feat.mean_harmful_activation / maxActivation * 100) : 0;
            const benignPct = maxActivation > 0 ? (feat.mean_benign_activation / maxActivation * 100) : 0;
            // Determine direction based on actual activation comparison
            const harmfulHigher = feat.mean_harmful_activation > feat.mean_benign_activation;
            const diffArrow = harmfulHigher ? '↑' : '↓';
            const diffClass = harmfulHigher ? 'positive' : 'negative';
            // Show signed difference value
            const signedDiff = feat.mean_harmful_activation - feat.mean_benign_activation;
            const diffDisplay = (signedDiff >= 0 ? '+' : '') + signedDiff.toFixed(3);

            html += `
                <tr class="ranking-row" data-feature="${feat.feature_id}" data-layer="${layer}">
                    <td>${idx + 1}</td>
                    <td class="feature-id">#${feat.feature_id}</td>
                    <td>${(feat.consistency_score * 100).toFixed(1)}%</td>
                    <td class="activation-visual">
                        <div class="activation-comparison">
                            <div class="bar-group">
                                <div class="bar-wrapper">
                                    <span class="bar-label h">H</span>
                                    <div class="mini-bar harmful" style="width: ${harmfulPct}%"></div>
                                    <span class="activation-value harmful">${feat.mean_harmful_activation.toFixed(2)}</span>
                                </div>
                                <div class="bar-wrapper">
                                    <span class="bar-label b">B</span>
                                    <div class="mini-bar benign" style="width: ${benignPct}%"></div>
                                    <span class="activation-value benign">${feat.mean_benign_activation.toFixed(2)}</span>
                                </div>
                            </div>
                        </div>
                    </td>
                    <td>
                        <div class="diff-indicator">
                            <span class="diff-arrow ${diffClass}">${diffArrow}</span>
                            <span class="${diffClass}">${diffDisplay}</span>
                        </div>
                    </td>
                    <td><button class="add-to-queue-btn" onclick="addToSteeringQueue(${feat.feature_id}, ${layer}, 'batch', ${activation}, ${maxActivation})">+</button></td>
                </tr>
            `;
        });
    } else {
        const barClass = singleCategoryType === 'harmful' ? 'harmful' : 'benign';
        html += `
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Feature</th>
                    <th>Mean Activation</th>
                    <th></th>
                </tr>
            </thead>
            <tbody>
        `;
        features.forEach((feat, idx) => {
            const activation = feat.mean_activation;
            const activationPct = maxActivation > 0 ? (feat.mean_activation / maxActivation * 100) : 0;
            html += `
                <tr class="ranking-row" data-feature="${feat.feature_id}" data-layer="${layer}">
                    <td>${idx + 1}</td>
                    <td class="feature-id">#${feat.feature_id}</td>
                    <td class="activation-visual">
                        <div class="activation-cell">
                            <div class="activation-bar-container">
                                <div class="activation-bar ${barClass}" style="width: ${activationPct}%"></div>
                            </div>
                            <span class="activation-value ${barClass}">${feat.mean_activation.toFixed(3)}</span>
                        </div>
                    </td>
                    <td><button class="add-to-queue-btn" onclick="addToSteeringQueue(${feat.feature_id}, ${layer}, 'batch', ${activation}, ${maxActivation})">+</button></td>
                </tr>
            `;
        });
    }

    html += '</tbody></table></div>';

    if (rankingResults) {
        rankingResults.innerHTML = html;

        rankingResults.querySelectorAll('.ranking-row').forEach(el => {
            el.addEventListener('click', (e) => {
                if (e.target.classList.contains('add-to-queue-btn')) return;
                const featureId = parseInt(el.dataset.feature);
                const featureLayer = parseInt(el.dataset.layer);
                selectRankingFeature(featureId, featureLayer);
            });
        });

        if (features.length > 0) {
            selectRankingFeature(features[0].feature_id, layer);
        }
    }
}

async function selectRankingFeature(featureId, layer) {
    // Save scroll position before any DOM updates
    const scrollY = window.scrollY;

    document.querySelectorAll('#ranking-results .ranking-row').forEach(el => {
        el.classList.toggle('selected', parseInt(el.dataset.feature) === featureId);
    });
    selectedLayer = layer;

    // Show Neuronpedia embed in the detail panel
    renderNeuronpediaEmbed(layer, featureId);

    // Restore scroll position after DOM updates
    requestAnimationFrame(() => {
        window.scrollTo(0, scrollY);
    });

    // Show token activation visualization in the prompts area
    const tokenViz = document.getElementById('prompt-token-viz');
    const tokenVizContent = document.getElementById('prompt-token-viz-content');
    const featureLabel = document.getElementById('selected-feature-label');

    if (tokenViz && tokenVizContent && featureLabel) {
        featureLabel.textContent = `Feature #${featureId} (Layer ${layer})`;
        tokenVizContent.innerHTML = '<span class="placeholder">Loading token activations...</span>';
        tokenViz.classList.remove('hidden');

        // Restore scroll position after showing token viz
        requestAnimationFrame(() => {
            window.scrollTo(0, scrollY);
        });

        // Fetch token-level activations for this feature
        if (rankingCacheKey) {
            try {
                const tokenData = await getFeatureTokenActivationsAPI(rankingCacheKey, layer, featureId);
                renderPromptTokenActivations(tokenData);
                // Restore scroll position after rendering token activations
                requestAnimationFrame(() => {
                    window.scrollTo(0, scrollY);
                });
            } catch (error) {
                console.error('Failed to load token activations:', error);
                tokenVizContent.innerHTML = '<span class="placeholder">Failed to load token activations</span>';
            }
        }
    }
}

function renderPromptTokenActivations(tokenData) {
    const container = document.getElementById('prompt-token-viz-content');
    if (!container) return;

    let html = '';

    if (tokenData.mode === 'pairs') {
        // Show all pairs with harmful/benign comparison
        html += '<div class="token-pairs-container">';

        for (let i = 0; i < tokenData.prompts.length; i++) {
            const pair = tokenData.prompts[i];
            html += `
                <div class="token-pair">
                    <div class="token-pair-label">Pair ${i + 1}</div>
                    <div class="token-pair-row">
                        <div class="token-prompt harmful">
                            <span class="prompt-label harmful">H</span>
                            <div class="token-visualization">
                                ${renderTokensWithActivation(pair.harmful.tokens, pair.harmful.normalized, 'harmful')}
                            </div>
                            <span class="max-act-badge harmful">${pair.harmful.max_activation.toFixed(2)}</span>
                        </div>
                    </div>
                    <div class="token-pair-row">
                        <div class="token-prompt benign">
                            <span class="prompt-label benign">B</span>
                            <div class="token-visualization">
                                ${renderTokensWithActivation(pair.benign.tokens, pair.benign.normalized, 'benign')}
                            </div>
                            <span class="max-act-badge benign">${pair.benign.max_activation.toFixed(2)}</span>
                        </div>
                    </div>
                </div>
            `;
        }
        html += '</div>';
    } else {
        // Single category mode
        const category = tokenData.category || 'unknown';
        const colorClass = category === 'harmful' ? 'harmful' : 'benign';

        html += '<div class="token-single-container">';
        for (let i = 0; i < tokenData.prompts.length; i++) {
            const prompt = tokenData.prompts[i];
            html += `
                <div class="token-single-prompt">
                    <span class="prompt-index">${i + 1}</span>
                    <div class="token-visualization">
                        ${renderTokensWithActivation(prompt.tokens, prompt.normalized, colorClass)}
                    </div>
                    <span class="max-act-badge ${colorClass}">${prompt.max_activation.toFixed(2)}</span>
                </div>
            `;
        }
        html += '</div>';
    }

    container.innerHTML = html;
}

function closeTokenViz() {
    const tokenViz = document.getElementById('prompt-token-viz');
    if (tokenViz) {
        tokenViz.classList.add('hidden');
    }
    // Deselect features in ranking table
    document.querySelectorAll('#ranking-results .ranking-row').forEach(el => {
        el.classList.remove('selected');
    });
}

function renderTokensWithActivation(tokens, normalizedActs, colorClass) {
    let html = '';
    for (let i = 0; i < tokens.length; i++) {
        const token = tokens[i];
        const act = normalizedActs[i] || 0;
        const opacity = Math.min(0.9, act * 0.9 + 0.1);  // Min 0.1 opacity for visibility
        const displayToken = escapeHtml(token);

        // Skip BOS token display but show activation
        if (i === 0 && (token === '<bos>' || token === '<s>')) {
            html += `<span class="act-token bos ${colorClass}" style="--act-opacity: ${opacity}" title="Activation: ${(act * 100).toFixed(1)}%">${displayToken}</span>`;
        } else {
            html += `<span class="act-token ${colorClass}" style="--act-opacity: ${opacity}" title="Activation: ${(act * 100).toFixed(1)}%">${displayToken}</span>`;
        }
    }
    return html;
}

function toggleNpEmbed(btn) {
    const container = btn.closest('.np-embed-section').querySelector('.np-embed-container');
    if (container) {
        container.classList.toggle('collapsed');
        btn.textContent = container.classList.contains('collapsed') ? '▶' : '▼';
    }
}

// Prompt Pair Management

function addPromptPair(harmful = '', benign = '') {
    const container = document.getElementById('prompt-pairs-container');
    const row = document.createElement('div');
    row.className = 'prompt-pair-row';
    row.dataset.index = pairIndex++;

    row.innerHTML = `
        <input type="text" class="pair-harmful" placeholder="Harmful prompt" value="${escapeHtml(harmful)}" />
        <input type="text" class="pair-benign" placeholder="Benign prompt" value="${escapeHtml(benign)}" />
        <button type="button" class="btn-remove-pair" title="Remove">×</button>
    `;

    row.querySelector('.btn-remove-pair').addEventListener('click', () => {
        if (container.children.length > 1) {
            row.remove();
        }
    });

    container.appendChild(row);
}

function getPromptPairs() {
    const pairs = [];
    document.querySelectorAll('.prompt-pair-row').forEach(row => {
        const harmful = row.querySelector('.pair-harmful').value.trim();
        const benign = row.querySelector('.pair-benign').value.trim();
        if (harmful && benign) {
            pairs.push({ harmful, benign });
        }
    });
    return pairs;
}

function loadPresetPairs() {
    const container = document.getElementById('prompt-pairs-container');
    container.innerHTML = '';
    pairIndex = 0;
    singleCategoryMode = false;
    updateAddButtonText();
    PRESET_PAIRS.forEach(pair => addPromptPair(pair.harmful, pair.benign));
    setStatus(`Loaded ${PRESET_PAIRS.length} preset pairs`);
}

function addSinglePrompt(value = '') {
    const container = document.getElementById('prompt-pairs-container');
    const row = document.createElement('div');
    row.className = 'prompt-single-row';
    row.dataset.index = pairIndex++;

    row.innerHTML = `
        <input type="text" class="single-prompt" placeholder="${singleCategoryType} prompt" value="${escapeHtml(value)}" />
        <button type="button" class="btn-remove-pair" title="Remove">×</button>
    `;

    row.querySelector('.btn-remove-pair').addEventListener('click', () => {
        if (container.children.length > 1) {
            row.remove();
        }
    });

    container.appendChild(row);
}

function getSingleCategoryPrompts() {
    const prompts = [];
    document.querySelectorAll('.prompt-single-row').forEach(row => {
        const prompt = row.querySelector('.single-prompt').value.trim();
        if (prompt) {
            prompts.push(prompt);
        }
    });
    return prompts;
}

function clearAllPairs() {
    const container = document.getElementById('prompt-pairs-container');
    container.innerHTML = '';
    pairIndex = 0;
    singleCategoryMode = false;
    updateAddButtonText();
    addPromptPair();
    setStatus('Cleared all');
}

function updateAddButtonText() {
    const addBtn = document.getElementById('add-pair-btn');
    if (addBtn) {
        addBtn.textContent = singleCategoryMode ? '+ Add Prompt' : '+ Add Pair';
    }
}

// Steering

function addToSteeringQueue(featureId, layer, sourceTab = 'batch', activationStrength = null, maxActivation = null) {
    const exists = steeringQueue.some(q => q.feature_id === featureId && q.layer === layer);
    if (exists) {
        setStatus(`Feature #${featureId} (L${layer}) already in queue`);
        return;
    }

    // Normalize activation strength to [-1, 1] range relative to max activation
    let coefficient = 0.25;  // default
    if (activationStrength !== null && maxActivation !== null && maxActivation > 0) {
        coefficient = activationStrength / maxActivation;  // normalized to [0, 1]
    } else if (activationStrength !== null) {
        coefficient = Math.max(-1, Math.min(1, activationStrength / 100));  // fallback
    }

    steeringQueue.push({
        feature_id: featureId,
        layer: layer,
        coefficient: coefficient,
        source_tab: sourceTab
    });

    renderSteeringQueue();
    setStatus(`Added #${featureId} (L${layer}) to steering queue`);

    if (steeringQueue.length === 1 && !sidebarOpen) {
        toggleSteeringSidebar(true);
    }
}

function removeFromSteeringQueue(index) {
    if (index >= 0 && index < steeringQueue.length) {
        const item = steeringQueue[index];
        steeringQueue.splice(index, 1);
        renderSteeringQueue();
        setStatus(`Removed #${item.feature_id} from steering queue`);
    }
}

function updateQueueCoefficient(index, coefficient) {
    if (index >= 0 && index < steeringQueue.length) {
        steeringQueue[index].coefficient = coefficient;
    }
}

function clearSteeringQueue() {
    steeringQueue = [];
    renderSteeringQueue();
    setStatus('Steering queue cleared');
}

function renderSteeringQueue() {
    const container = document.getElementById('steering-queue-list');
    const countEl = document.querySelector('.queue-count');

    if (!container) return;

    if (countEl) {
        countEl.textContent = `${steeringQueue.length} feature${steeringQueue.length !== 1 ? 's' : ''}`;
    }

    if (steeringQueue.length === 0) {
        container.innerHTML = '<span class="placeholder">Add features from rankings using the + button</span>';
        return;
    }

    let html = '';
    steeringQueue.forEach((item, idx) => {
        html += `
            <div class="queue-item" data-index="${idx}">
                <span class="queue-feature">L${item.layer} #${item.feature_id}</span>
                <input type="range" class="queue-coeff-slider" min="-1" max="1" step="0.05" value="${item.coefficient}" />
                <input type="number" class="queue-coeff-input" step="0.00001" value="${item.coefficient}" />
                <button class="queue-remove-btn" title="Remove">×</button>
            </div>
        `;
    });

    container.innerHTML = html;

    container.querySelectorAll('.queue-item').forEach(itemEl => {
        const idx = parseInt(itemEl.dataset.index);
        const slider = itemEl.querySelector('.queue-coeff-slider');
        const input = itemEl.querySelector('.queue-coeff-input');

        autoSizeInput(input, 50);

        slider.addEventListener('input', () => {
            const val = parseFloat(slider.value);
            input.value = val;
            autoSizeInput(input, 50);
            updateQueueCoefficient(idx, val);
        });

        input.addEventListener('input', () => {
            const val = parseFloat(input.value) || 0;
            slider.value = Math.max(-1, Math.min(1, val));
            autoSizeInput(input, 50);
            updateQueueCoefficient(idx, val);
        });

        itemEl.querySelector('.queue-remove-btn').addEventListener('click', () => removeFromSteeringQueue(idx));
    });
}

function toggleSteeringSidebar(forceOpen = null) {
    const sidebar = document.getElementById('steering-sidebar');
    if (!sidebar) return;

    if (forceOpen !== null) {
        sidebarOpen = forceOpen;
    } else {
        sidebarOpen = !sidebarOpen;
    }

    sidebar.classList.toggle('open', sidebarOpen);
}

async function runSidebarSteering() {
    if (steeringQueue.length === 0) {
        setStatus('Add features to the queue first');
        return;
    }

    const prompt = document.getElementById('sidebar-prompt').value.trim();
    const maxTokens = parseInt(document.getElementById('sidebar-tokens').value);
    const normalization = document.getElementById('sidebar-normalization').value || null;
    const unitNormalize = document.getElementById('sidebar-unit-normalize').checked;
    const showBaseline = document.getElementById('sidebar-show-baseline').checked;

    if (!prompt) {
        setStatus('Please enter a prompt');
        return;
    }

    const steerBtn = document.getElementById('sidebar-steer-btn');
    const output = document.getElementById('sidebar-output');

    setLoading(steerBtn, true);
    setStatus(`Generating with ${steeringQueue.length} steering vector(s)...`);

    try {
        const steeringFeatures = steeringQueue.map(item => ({
            feature_id: item.feature_id,
            layer: item.layer,
            coefficient: item.coefficient
        }));

        const result = await generateWithSteeringMulti(prompt, steeringFeatures, maxTokens, normalization, unitNormalize, !showBaseline);

        let steeringInfo = steeringFeatures
            .map(s => `L${s.layer}:#${s.feature_id}: ${s.coefficient > 0 ? '+' : ''}${s.coefficient}`)
            .join(', ');

        const normParts = [];
        if (result.unit_normalize) normParts.push('unit');
        if (result.normalization) normParts.push(result.normalization.replace('_', '-'));
        const normLabel = normParts.length > 0 ? ` (${normParts.join(' + ')})` : '';

        let outputHtml = '';
        if (result.original_output !== null) {
            outputHtml += `
                <div class="output-section original">
                    <h4>Original Output</h4>
                    <div class="output-text">${escapeHtml(result.original_output)}</div>
                </div>
            `;
        }
        outputHtml += `
            <div class="output-section steered">
                <h4>Steered Output${normLabel}</h4>
                <p class="hint">Vectors: ${steeringInfo}</p>
                <div class="output-text">${escapeHtml(result.steered_output)}</div>
            </div>
        `;
        output.innerHTML = outputHtml;
        setStatus('Steering generation complete');
    } catch (error) {
        setStatus(`Error: ${error.message}`);
        output.innerHTML = `<span class="placeholder" style="color: var(--accent-red)">Error: ${escapeHtml(error.message)}</span>`;
    } finally {
        setLoading(steerBtn, false);
    }
}

function initSteeringSidebar() {
    document.getElementById('sidebar-toggle')?.addEventListener('click', () => toggleSteeringSidebar());
    document.getElementById('sidebar-close')?.addEventListener('click', () => toggleSteeringSidebar(false));
    document.getElementById('clear-queue-btn')?.addEventListener('click', clearSteeringQueue);
    document.getElementById('sidebar-steer-btn')?.addEventListener('click', runSidebarSteering);
    initSidebarResize();
    renderSteeringQueue();
}

function initSidebarResize() {
    const sidebar = document.getElementById('steering-sidebar');
    const resizeHandle = document.getElementById('sidebar-resize-handle');

    if (!sidebar || !resizeHandle) return;

    let isResizing = false;
    let startX = 0;
    let startWidth = 0;

    function startResize(clientX) {
        isResizing = true;
        startX = clientX;
        startWidth = sidebar.offsetWidth;
        resizeHandle.classList.add('resizing');
        document.body.classList.add('sidebar-resizing');
        sidebar.style.transition = 'none';
    }

    function doResize(clientX) {
        if (!isResizing) return;
        const deltaX = startX - clientX;
        let newWidth = startWidth + deltaX;
        const minWidth = 280;
        const maxWidth = window.innerWidth * 0.9;
        newWidth = Math.max(minWidth, Math.min(maxWidth, newWidth));
        sidebar.style.width = `${newWidth}px`;
    }

    function stopResize() {
        if (!isResizing) return;
        isResizing = false;
        resizeHandle.classList.remove('resizing');
        document.body.classList.remove('sidebar-resizing');
        sidebar.style.transition = '';
    }

    resizeHandle.addEventListener('mousedown', (e) => { startResize(e.clientX); e.preventDefault(); });
    document.addEventListener('mousemove', (e) => doResize(e.clientX));
    document.addEventListener('mouseup', stopResize);

    resizeHandle.addEventListener('touchstart', (e) => { startResize(e.touches[0].clientX); e.preventDefault(); });
    document.addEventListener('touchmove', (e) => { if (isResizing && e.touches[0]) doResize(e.touches[0].clientX); });
    document.addEventListener('touchend', stopResize);
}

// Settings Modal

function openSettingsModal() {
    document.getElementById('settings-modal')?.classList.remove('hidden');
}

function closeSettingsModal() {
    document.getElementById('settings-modal')?.classList.add('hidden');
}

async function applySettings() {
    const settingsApply = document.getElementById('settings-apply');
    const modelPath = document.getElementById('config-model-path')?.value?.trim();
    const saeRepoSelect = document.getElementById('config-sae-repo');
    const saeRepo = saeRepoSelect?.value;
    const baseModel = saeRepoSelect?.selectedOptions[0]?.dataset.baseModel;
    const saeWidth = document.getElementById('config-sae-width')?.value;
    const saeL0 = document.getElementById('config-sae-l0')?.value;

    const hasChanges =
        (modelPath && modelPath !== currentConfig?.model_path) ||
        (saeRepo && saeRepo !== currentConfig?.sae_repo) ||
        (saeWidth && saeWidth !== currentConfig?.sae_width) ||
        (saeL0 && saeL0 !== currentConfig?.sae_l0);

    if (!hasChanges) {
        closeSettingsModal();
        return;
    }

    setLoading(settingsApply, true);
    setStatus('Updating configuration...');

    try {
        const result = await updateConfig(modelPath, saeRepo, saeWidth, saeL0, baseModel);
        if (result.success) {
            rankingLayerDataCache = {};
            selectedLayer = null;
            await fetchConfig();
            setStatus('Configuration updated. Model will load on next analysis.');
            closeSettingsModal();
        } else {
            setStatus(`Error: ${result.error}`);
        }
    } catch (error) {
        setStatus(`Error: ${error.message}`);
    } finally {
        setLoading(settingsApply, false);
    }
}

function initSettingsModal() {
    document.getElementById('settings-btn')?.addEventListener('click', openSettingsModal);
    document.getElementById('settings-close')?.addEventListener('click', closeSettingsModal);
    document.getElementById('settings-cancel')?.addEventListener('click', closeSettingsModal);
    document.getElementById('settings-apply')?.addEventListener('click', applySettings);

    document.getElementById('settings-modal')?.addEventListener('click', (e) => {
        if (e.target.id === 'settings-modal') closeSettingsModal();
    });

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !document.getElementById('settings-modal')?.classList.contains('hidden')) {
            closeSettingsModal();
        }
    });
}

// Bake In Modal

async function applySteeringPermanent(features, outputPath, scaleFactor) {
    const response = await fetch('/api/apply-steering-permanent', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features, output_path: outputPath, scale_factor: scaleFactor })
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to apply steering');
    }
    return await response.json();
}

function openBakeInModal() {
    const bakeInModal = document.getElementById('bake-in-modal');
    if (!bakeInModal) return;

    const features = steeringQueue.filter(f => f.coefficient !== 0);
    if (features.length === 0) {
        setStatus('No features in controller');
        return;
    }

    const bakeInFeaturesInfo = document.getElementById('bake-in-features-info');
    if (bakeInFeaturesInfo) {
        const layers = [...new Set(features.map(f => f.layer))];
        bakeInFeaturesInfo.textContent = `${features.length} feature(s) across layer(s): ${layers.join(', ')}`;
    }

    const bakeInPath = document.getElementById('bake-in-path');
    if (bakeInPath && currentConfig?.model_path) {
        const basePath = currentConfig.model_path.replace(/[/\\]$/, '');
        bakeInPath.value = basePath + '-steered';
    }

    bakeInModal.classList.remove('hidden');
}

function closeBakeInModal() {
    document.getElementById('bake-in-modal')?.classList.add('hidden');
}

async function executeBakeIn() {
    const bakeInApply = document.getElementById('bake-in-apply');
    const outputPath = document.getElementById('bake-in-path')?.value?.trim();
    const scaleFactor = parseFloat(document.getElementById('bake-in-scale')?.value) || 1.0;

    if (!outputPath) {
        setStatus('Please specify an output path');
        return;
    }

    const features = steeringQueue
        .filter(f => f.coefficient !== 0)
        .map(f => ({ layer: f.layer, feature_id: f.feature_id, coefficient: f.coefficient }));

    if (features.length === 0) {
        setStatus('No features to bake in');
        return;
    }

    setLoading(bakeInApply, true);
    setStatus('Applying steering vectors to model weights...');

    try {
        const result = await applySteeringPermanent(features, outputPath, scaleFactor);
        if (result.success) {
            setStatus(`Model saved to ${result.output_path} with ${result.total_features} feature(s) baked in`);
            closeBakeInModal();
        } else {
            setStatus(`Error: ${result.error}`);
        }
    } catch (error) {
        setStatus(`Error: ${error.message}`);
    } finally {
        setLoading(bakeInApply, false);
    }
}

function initBakeInModal() {
    document.getElementById('bake-in-btn')?.addEventListener('click', openBakeInModal);
    document.getElementById('bake-in-close')?.addEventListener('click', closeBakeInModal);
    document.getElementById('bake-in-cancel')?.addEventListener('click', closeBakeInModal);
    document.getElementById('bake-in-apply')?.addEventListener('click', executeBakeIn);

    document.getElementById('bake-in-modal')?.addEventListener('click', (e) => {
        if (e.target.id === 'bake-in-modal') closeBakeInModal();
    });

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !document.getElementById('bake-in-modal')?.classList.contains('hidden')) {
            closeBakeInModal();
        }
    });
}

// File Upload Handling

function handleFileSelect(e) {
    const file = e.target.files[0];
    const loadFileBtn = document.getElementById('load-file-btn');
    const uploadCountInput = document.getElementById('upload-count-input');
    const uploadTypeSelect = document.getElementById('upload-type-select');

    if (!file) {
        loadFileBtn.disabled = true;
        uploadedFileData = null;
        return;
    }

    const type = uploadTypeSelect.value;

    const reader = new FileReader();
    reader.onload = function(event) {
        try {
            if (type === 'json') {
                const data = JSON.parse(event.target.result);
                if (data.pairs && Array.isArray(data.pairs)) {
                    uploadedFileData = data.pairs;
                } else if (Array.isArray(data)) {
                    uploadedFileData = data;
                } else {
                    throw new Error('Invalid format: expected {pairs: [...]} or array');
                }
            } else {
                const lines = event.target.result.split('\n')
                    .map(line => line.trim())
                    .filter(line => line.length > 0);
                if (lines.length === 0) {
                    throw new Error('No prompts found in file');
                }
                uploadedFileData = lines;
            }

            const total = uploadedFileData.length;
            uploadCountInput.max = total;
            uploadCountInput.value = Math.min(10, total);
            autoSizeInput(uploadCountInput, 45);
            loadFileBtn.disabled = false;

            const itemType = type === 'json' ? 'pairs' : 'prompts';
            setStatus(`Found ${total} ${itemType}. Select count and click "Import".`);
        } catch (error) {
            loadFileBtn.disabled = true;
            uploadedFileData = null;
            setStatus(`Error: ${error.message}`);
        }
    };

    reader.onerror = function() {
        loadFileBtn.disabled = true;
        uploadedFileData = null;
        setStatus('Error reading file');
    };

    reader.readAsText(file);
}

function loadFromFile() {
    const uploadCountInput = document.getElementById('upload-count-input');
    const uploadTypeSelect = document.getElementById('upload-type-select');

    if (!uploadedFileData || uploadedFileData.length === 0) {
        setStatus('No data available');
        return;
    }

    const count = parseInt(uploadCountInput.value) || 10;
    const type = uploadTypeSelect.value;
    const itemsToLoad = uploadedFileData.slice(0, count);

    const container = document.getElementById('prompt-pairs-container');
    container.innerHTML = '';
    pairIndex = 0;

    if (type === 'json') {
        singleCategoryMode = false;
        updateAddButtonText();
        itemsToLoad.forEach(pair => {
            const harmful = pair.harmful || '';
            const benign = pair.harmless || pair.benign || '';
            addPromptPair(harmful, benign);
        });
        setStatus(`Loaded ${itemsToLoad.length} prompt pairs`);
    } else {
        const category = type === 'txt-harmful' ? 'harmful' : 'harmless';
        singleCategoryMode = true;
        singleCategoryType = category;
        updateAddButtonText();
        itemsToLoad.forEach(prompt => addSinglePrompt(prompt));
        setStatus(`Loaded ${itemsToLoad.length} ${category} prompts`);
    }
}

// Event Handlers

function initEventHandlers() {
    // Rank button
    document.getElementById('rank-btn')?.addEventListener('click', async () => {
        const rankBtn = document.getElementById('rank-btn');
        rankingLayerDataCache = {};

        if (singleCategoryMode) {
            const prompts = getSingleCategoryPrompts();
            if (prompts.length === 0) {
                setStatus('Add at least one prompt');
                return;
            }

            setLoading(rankBtn, true);
            setStatus(`Caching activations for ${prompts.length} ${singleCategoryType} prompts...`);
            rankingMode = 'single';

            try {
                rankingResult = await rankFeaturesSingleAPI(prompts, singleCategoryType, 100, true);
                rankingCacheKey = rankingResult.cache_key;

                const layers = rankingResult.available_layers;
                const firstLayer = layers[0];

                renderLayerTabs('ranking-layer-tabs', layers, firstLayer, loadRankingLayer);
                await loadRankingLayer(firstLayer);

                document.getElementById('export-rank-json').disabled = false;
                document.getElementById('export-rank-csv').disabled = false;

                setStatus(`Ranking ready - select layers to analyze`);
            } catch (error) {
                setStatus(`Error: ${error.message}`);
                document.getElementById('ranking-results').innerHTML =
                    `<span class="placeholder" style="color: var(--accent-red)">Error: ${escapeHtml(error.message)}</span>`;
            } finally {
                setLoading(rankBtn, false);
            }
            return;
        }

        const pairs = getPromptPairs();
        if (pairs.length === 0) {
            setStatus('Add at least one prompt pair');
            return;
        }

        setLoading(rankBtn, true);
        setStatus(`Caching activations for ${pairs.length} prompt pairs...`);
        rankingMode = 'pairs';

        try {
            rankingResult = await rankFeaturesAPI(pairs, 100, true);
            rankingCacheKey = rankingResult.cache_key;

            const layers = rankingResult.available_layers;
            const firstLayer = layers[0];

            renderLayerTabs('ranking-layer-tabs', layers, firstLayer, loadRankingLayer);
            await loadRankingLayer(firstLayer);

            document.getElementById('export-rank-json').disabled = false;
            document.getElementById('export-rank-csv').disabled = false;

            setStatus(`Ranking ready - select layers to analyze`);
        } catch (error) {
            setStatus(`Error: ${error.message}`);
            document.getElementById('ranking-results').innerHTML =
                `<span class="placeholder" style="color: var(--accent-red)">Error: ${escapeHtml(error.message)}</span>`;
        } finally {
            setLoading(rankBtn, false);
        }
    });

    // Add pair/prompt button
    document.getElementById('add-pair-btn')?.addEventListener('click', () => {
        if (singleCategoryMode) {
            addSinglePrompt();
        } else {
            addPromptPair();
        }
    });

    // Load preset button
    document.getElementById('load-preset-btn')?.addEventListener('click', loadPresetPairs);

    // Clear all pairs button
    document.getElementById('clear-pairs-btn')?.addEventListener('click', clearAllPairs);

    // File upload handling
    const fileInput = document.getElementById('file-input');
    const loadFileBtn = document.getElementById('load-file-btn');
    const uploadCountInput = document.getElementById('upload-count-input');
    const uploadTypeSelect = document.getElementById('upload-type-select');
    const uploadCountLabel = document.getElementById('upload-count-label');

    uploadTypeSelect?.addEventListener('change', () => {
        const type = uploadTypeSelect.value;
        if (uploadCountLabel) {
            uploadCountLabel.textContent = type === 'json' ? 'pairs' : 'prompts';
        }
        if (fileInput) fileInput.value = '';
        if (loadFileBtn) loadFileBtn.disabled = true;
        uploadedFileData = null;
    });

    fileInput?.addEventListener('change', handleFileSelect);
    loadFileBtn?.addEventListener('click', loadFromFile);

    if (uploadCountInput) {
        autoSizeInput(uploadCountInput, 45);
        uploadCountInput.addEventListener('input', () => autoSizeInput(uploadCountInput, 45));
    }

    // Export buttons
    document.getElementById('export-rank-json')?.addEventListener('click', () => {
        if (rankingResult) exportData('/api/export/rankings', rankingResult, 'json');
    });
    document.getElementById('export-rank-csv')?.addEventListener('click', () => {
        if (rankingResult) exportData('/api/export/rankings', rankingResult, 'csv');
    });

    // Initialize remove button for initial pair row
    document.querySelector('.btn-remove-pair')?.addEventListener('click', function() {
        const container = document.getElementById('prompt-pairs-container');
        if (container && container.children.length > 1) {
            this.closest('.prompt-pair-row')?.remove();
        }
    });

    // Close token visualization button
    document.getElementById('close-token-viz')?.addEventListener('click', closeTokenViz);
}

// Initialization

document.addEventListener('DOMContentLoaded', () => {
    fetchConfig();
    initEventHandlers();
    initSteeringSidebar();
    initSettingsModal();
    initBakeInModal();
    setStatus('Ready - add prompt pairs and click Rank Features');
});
