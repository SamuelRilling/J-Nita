import {
    getActiveStage, setActiveStage, triggerStageComplete,
    resetPipelineState, wait
} from './pipeline-state.js';
import { runOcr, pingProvider } from './api-client.js';
import { providers } from './provider-registry.js';

// ── State ─────────────────────────────────────────────────
let currentImage            = null;
let conditionedImage        = null;
let originalImageElement    = null;
let conditionedImageElement = null;
let ocrRawText      = '';
let outputMode      = 'markdown';
let selectedProvider = null;
let conditionController = null;
let ocrController       = null;

// Per-provider parameter values (keyed by provider id) and health-check status
const providerParamValues = {};
const providerStatus = {};

// ── Utility ───────────────────────────────────────────────
function humanizeError(err) {
    const msg = err.message || '';
    if (/fetch|network|failed to fetch/i.test(msg)) return 'Network error. Check your connection.';
    if (/timeout|timed out/i.test(msg)) return 'Request timed out. Try a smaller image.';
    if (/abort/i.test(msg)) return 'Request was cancelled.';
    if (/502/i.test(msg)) return msg;
    if (/413/i.test(msg)) return 'Image too large (30 MB limit).';
    return msg || 'Something went wrong. Please try again.';
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
}

// ── Log helpers ───────────────────────────────────────────
function createLogLine(container, message) {
    const line = document.createElement('div');
    line.className = 'progress-line';
    line.innerHTML = `<span class="progress-icon">⏳</span><span class="progress-message">${message}</span>`;
    container.appendChild(line);
    container.scrollTop = container.scrollHeight;
    return line;
}

function markLogLine(line, status, message) {
    const icon = line.querySelector('.progress-icon');
    const text = line.querySelector('.progress-message');
    if (status === 'complete') { line.classList.add('complete'); icon.textContent = '✅'; }
    else if (status === 'error') { line.classList.add('error'); icon.textContent = '⚠️'; }
    else { icon.textContent = '⏳'; }
    if (message) text.textContent = message;
}

function showOverlay(id) { document.getElementById(id).classList.add('active'); }
function hideOverlay(id) { document.getElementById(id).classList.remove('active'); }

async function runStage(container, message, delay = 520) {
    const line = createLogLine(container, message);
    await wait(delay);
    markLogLine(line, 'complete');
    return line;
}

async function waitForPromiseWithUpdates(container, promise, message) {
    const line = createLogLine(container, message);
    let resolved = false;
    while (!resolved) {
        try {
            const result = await Promise.race([promise, wait(1200)]);
            if (result !== undefined) {
                resolved = true;
                markLogLine(line, 'complete', 'Backend response received.');
                return result;
            }
            markLogLine(line, 'pending', 'Waiting for backend response...');
        } catch (e) {
            markLogLine(line, 'error', 'Backend request failed.');
            throw e;
        }
    }
    return await promise;
}

// ── Canvas rendering ──────────────────────────────────────
function drawImageToCanvas(img, canvas, options = {}) {
    const ctx = canvas.getContext('2d');
    let maxW = options.maxWidth  || canvas.parentElement.clientWidth  - 4;
    let maxH = options.maxHeight || canvas.parentElement.clientHeight - 4;
    if (maxW <= 0) maxW = 320;
    if (maxH <= 0) maxH = 240;
    const ratio = Math.min(maxW / img.width, maxH / img.height);
    const w = img.width * ratio, h = img.height * ratio;
    const ox = (maxW - w) / 2, oy = (maxH - h) / 2;
    canvas.width = maxW; canvas.height = maxH;
    ctx.clearRect(0, 0, maxW, maxH);
    ctx.drawImage(img, ox, oy, w, h);
    return { ctx, width: w, height: h, offsetX: ox, offsetY: oy, scale: ratio };
}

function displayConditionedImage(img) {
    const canvas = document.getElementById('conditionedCanvas');
    const placeholder = document.getElementById('conditionedPlaceholder');
    placeholder.style.display = 'none';
    canvas.style.display = 'block';
    drawImageToCanvas(img, canvas);
}

function renderBoundingBox(img) {
    const canvas = document.getElementById('boundingCanvas');
    const { ctx, width, height, offsetX, offsetY } = drawImageToCanvas(img, canvas, {
        maxWidth: canvas.clientWidth, maxHeight: canvas.clientHeight
    });
    ctx.strokeStyle = '#c98a2a';
    ctx.lineWidth = 2;
    const ix = width * 0.08, iy = height * 0.08;
    ctx.strokeRect(offsetX + ix, offsetY + iy, width - ix * 2, height - iy * 2);
}

function renderCropPreview(img) {
    const canvas = document.getElementById('cropCanvas');
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w; canvas.height = h;
    const cw = img.width * 0.8, ch = img.height * 0.8;
    const sx = (img.width - cw) / 2, sy = (img.height - ch) / 2;
    canvas.getContext('2d').drawImage(img, sx, sy, cw, ch, 0, 0, w, h);
}

function renderFilterPreview(img) {
    const canvas = document.getElementById('filterCanvas');
    const { ctx } = drawImageToCanvas(img, canvas, {
        maxWidth: canvas.clientWidth, maxHeight: canvas.clientHeight
    });
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const d = imageData.data;
    for (let i = 0; i < d.length; i += 4) {
        const avg = (d[i] + d[i+1] + d[i+2]) / 3;
        const v = avg > 170 ? 255 : avg * 0.9;
        d[i] = d[i+1] = d[i+2] = v;
    }
    ctx.putImageData(imageData, 0, 0);
}

function renderFinalPreview(img) {
    const canvas = document.getElementById('finalCanvas');
    drawImageToCanvas(img, canvas, { maxWidth: canvas.clientWidth, maxHeight: canvas.clientHeight });
}

function renderAdvancedCanvases() {
    if (!originalImageElement || !conditionedImageElement) return;
    renderBoundingBox(originalImageElement);
    renderCropPreview(originalImageElement);
    renderFilterPreview(originalImageElement);
    renderFinalPreview(conditionedImageElement);
}

function clearCanvas(id) {
    const c = document.getElementById(id);
    c.getContext('2d').clearRect(0, 0, c.width, c.height);
}

// ── File handling ─────────────────────────────────────────
function handleDragOver(e) {
    e.preventDefault(); e.stopPropagation();
    e.currentTarget.classList.add('dragover');
}

function handleDragLeave(e) {
    e.currentTarget.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault(); e.stopPropagation();
    e.currentTarget.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFile(files[0]);
}

function handleFileSelect(e) {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        showToast('Please upload an image file', 'error');
        return;
    }
    const reader = new FileReader();
    reader.onload = e => {
        const img = new Image();
        img.onload = () => {
            currentImage = e.target.result;
            originalImageElement = img;

            document.getElementById('uploadArea').classList.add('hidden');
            const wrapper = document.getElementById('inputCanvasWrapper');
            wrapper.classList.remove('hidden');
            const canvas = document.getElementById('originalCanvas');
            canvas.style.display = 'block';
            drawImageToCanvas(img, canvas);

            const overlay = document.getElementById('conditionBtnOverlay');
            overlay.classList.add('has-image');
            document.getElementById('conditionBtn').disabled = false;

            conditionedImage = null;
            ocrRawText = '';
            document.getElementById('ocrBtn').disabled = true;
            document.getElementById('rerunBtn').disabled = true;
            document.getElementById('advancedBtn').disabled = true;
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

// ── Condition image ───────────────────────────────────────
async function conditionImage(skipAnimation = false) {
    if (!currentImage) { showToast('Upload an image first', 'error'); return; }

    const btn = document.getElementById('conditionBtn');
    btn.disabled = true;
    btn.innerHTML = '<span class="loading" aria-hidden="true"></span> Conditioning...';
    document.getElementById('cancelConditionBtn').classList.remove('hidden');

    const logContainer = document.getElementById('conditionLog');
    logContainer.innerHTML = '';
    showOverlay('conditionOverlay');
    document.getElementById('conditionedCanvas').style.display = 'none';
    document.getElementById('conditionedPlaceholder').style.display = 'flex';
    resetAdvancedPanel();

    conditionController = new AbortController();
    const timeoutId = setTimeout(() => conditionController.abort(), 90000);
    const strength    = parseInt(document.getElementById('strength').value);
    const compression = parseInt(document.getElementById('png_compression').value);

    const requestPromise = fetch('/api/condition', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: conditionController.signal,
        body: JSON.stringify({ image: currentImage, params: { strength, png_compression: compression } })
    });

    try {
        if (!skipAnimation) {
            await triggerStageComplete('input');
            if (conditionController.signal.aborted) { hideOverlay('conditionOverlay'); return; }
            setActiveStage('process');
        }

        const stages = [
            'Checking image orientation',
            'Detecting page edges',
            'Locating document boundaries',
            'Cropping to document area',
            'Scaling for OCR',
            'Applying noise reduction and contrast',
            'Sharpening text strokes',
        ];
        for (const s of stages) await runStage(logContainer, s, 520);

        const response = await waitForPromiseWithUpdates(logContainer, requestPromise, 'Waiting for backend');

        if (!response.ok) {
            let errMsg = `Backend error ${response.status}`;
            try { const e = await response.json(); if (e.error) errMsg = e.error; } catch (_) {}
            throw new Error(errMsg);
        }

        const data = await response.json();
        if (data.conditioned_image) {
            conditionedImage = data.conditioned_image;
            const img = new Image();
            img.onload = () => {
                conditionedImageElement = img;
                displayConditionedImage(img);
                document.getElementById('ocrBtn').disabled = false;
                document.getElementById('rerunBtn').disabled = false;
                document.getElementById('advancedBtn').disabled = false;
                renderAdvancedCanvases();
                hideOverlay('conditionOverlay');
                showToast('Image conditioned', 'success');
            };
            img.src = data.conditioned_image;
        } else {
            const line = createLogLine(logContainer, 'Conditioning failed');
            markLogLine(line, 'error', data.error || 'Failed to condition image');
            showToast(data.error || 'Conditioning failed', 'error');
        }
    } catch (err) {
        const line = createLogLine(logContainer, 'Conditioning failed');
        const msg = err.name === 'AbortError' ? 'Request timed out. Try a smaller image.' : humanizeError(err);
        markLogLine(line, 'error', msg);
        showToast(msg, 'error');
    } finally {
        clearTimeout(timeoutId);
        conditionController = null;
        btn.disabled = false;
        btn.innerHTML = 'Condition Image';
        document.getElementById('cancelConditionBtn').classList.add('hidden');
    }
}

// ── Process OCR ───────────────────────────────────────────
async function processOCR() {
    if (!conditionedImage) { showToast('Condition the image first', 'error'); return; }

    const btn = document.getElementById('ocrBtn');
    btn.disabled = true;
    btn.innerHTML = '<span class="loading" aria-hidden="true"></span> Running...';
    document.getElementById('cancelOcrBtn').classList.remove('hidden');

    const logContainer = document.getElementById('ocrLog');
    logContainer.innerHTML = '';
    showOverlay('ocrOverlay');
    const textContent = document.getElementById('textContent');
    textContent.textContent = 'OCR running...';
    textContent.classList.add('empty');

    try {
        await triggerStageComplete('process');
        setActiveStage('result');

        const stages = [
            'Preparing image for OCR',
            `Using provider: ${selectedProvider || 'none configured'}`,
            'Sending to OCR service',
            'Extracting text',
            'Formatting output',
        ];
        for (const s of stages) await runStage(logContainer, s, 520);

        const data = await runOcr(selectedProvider || 'no-provider', conditionedImage, providerParamValues[selectedProvider] || {});
        ocrRawText = data.markdown || data.text || '';
        hideOverlay('ocrOverlay');
        renderOutputText();
        showToast('OCR complete', 'success');
    } catch (err) {
        const line = createLogLine(logContainer, 'OCR failed');
        const msg = humanizeError(err);
        markLogLine(line, 'error', msg);
        showToast(msg, 'error');
    } finally {
        ocrController = null;
        btn.disabled = false;
        btn.innerHTML = 'Run OCR';
        document.getElementById('cancelOcrBtn').classList.add('hidden');
    }
}

// ── Output rendering ──────────────────────────────────────
function renderOutputText() {
    const el = document.getElementById('textContent');
    el.classList.remove('empty', 'preview-mode');
    if (!ocrRawText) { el.classList.add('empty'); el.textContent = 'No text extracted.'; return; }
    if (outputMode === 'preview' && typeof marked !== 'undefined') {
        el.classList.add('preview-mode');
        el.innerHTML = marked.parse(ocrRawText);
    } else if (outputMode === 'plain') {
        el.textContent = ocrRawText
            .replace(/#{1,6}\s+/g, '')
            .replace(/\*\*(.+?)\*\*/g, '$1')
            .replace(/\*(.+?)\*/g, '$1')
            .replace(/`(.+?)`/g, '$1')
            .replace(/\[(.+?)\]\(.+?\)/g, '$1');
    } else {
        el.textContent = ocrRawText;
    }
}

// ── Export ────────────────────────────────────────────────
function copyResult() {
    if (!ocrRawText) { showToast('No text to copy', 'error'); return; }
    navigator.clipboard.writeText(ocrRawText).then(() => {
        showToast('Copied to clipboard', 'success');
    }).catch(() => {
        showToast('Copy failed. Use browser copy (Ctrl+A, Ctrl+C).', 'error');
    });
    document.getElementById('exportDropdown').classList.add('hidden');
}

async function downloadResult(format) {
    if (!ocrRawText) { showToast('No text to export', 'error'); return; }
    document.getElementById('exportDropdown').classList.add('hidden');
    if (format === 'md') {
        const blob = new Blob([ocrRawText], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);
        const a = Object.assign(document.createElement('a'), { href: url, download: 'ocr_result.md' });
        a.click(); URL.revokeObjectURL(url);
    } else {
        try {
            const r = await fetch(`/api/export/${format}`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ markdown_text: ocrRawText })
            });
            if (r.ok) {
                const blob = await r.blob();
                const url = URL.createObjectURL(blob);
                const a = Object.assign(document.createElement('a'), { href: url, download: `ocr_result.${format}` });
                a.click(); URL.revokeObjectURL(url);
            } else {
                showToast('Export failed', 'error');
            }
        } catch (e) { showToast(humanizeError(e), 'error'); }
    }
}

// ── Provider dropdown ─────────────────────────────────────
function renderApiDropdown() {
    const list = document.getElementById('apiModelList');
    list.innerHTML = '';
    if (providers.length === 0) {
        const msg = document.createElement('div');
        msg.className = 'dropdown-item';
        msg.style.cssText = 'cursor:default; color: var(--muted); font-style: italic;';
        msg.textContent = 'No providers configured.';
        list.appendChild(msg);
        return;
    }

    // Group by tier; render free before paid
    const tiers = {};
    providers.forEach(p => { (tiers[p.tier] = tiers[p.tier] || []).push(p); });
    const tierOrder = ['free', 'paid'];
    const activeKeys = tierOrder.filter(t => tiers[t]);
    const multiTier = activeKeys.length > 1;

    activeKeys.forEach(tier => {
        const group = document.createElement('div');

        if (multiTier) {
            const header = document.createElement('div');
            header.className = 'tier-label';
            header.textContent = tier === 'free' ? 'Free' : 'Paid';
            group.appendChild(header);
        }

        tiers[tier].forEach(p => {
            // Initialise param values from spec defaults on first render
            if (!providerParamValues[p.id]) {
                providerParamValues[p.id] = {};
                (p.parameters || []).forEach(spec => {
                    providerParamValues[p.id][spec.key] = spec.default;
                });
            }

            const status = providerStatus[p.id] || 'untested';
            const statusChar = { untested: '?', testing: '…', success: '✓', failed: '✗' }[status] || '?';
            const isSelected = p.id === selectedProvider;

            const item = document.createElement('div');
            item.className = `api-model-item${isSelected ? ' is-selected' : ''}`;
            item.dataset.provider = p.id;

            const icon = document.createElement('span');
            icon.className = `model-status-icon model-status-${status}`;
            icon.textContent = statusChar;
            item.appendChild(icon);

            const nameEl = document.createElement('span');
            nameEl.className = 'model-name';
            nameEl.textContent = p.displayName;
            item.appendChild(nameEl);

            if (p.qualitativeTags && p.qualitativeTags.length) {
                const tags = document.createElement('span');
                tags.className = 'model-tags';
                p.qualitativeTags.forEach(tag => {
                    const pill = document.createElement('span');
                    pill.className = 'model-tag';
                    pill.textContent = tag;
                    tags.appendChild(pill);
                });
                item.appendChild(tags);
            }

            const testBtn = document.createElement('button');
            testBtn.className = 'btn btn-xs btn-ghost';
            testBtn.textContent = 'Test';
            testBtn.addEventListener('click', e => { e.stopPropagation(); testProvider(p.id); });
            item.appendChild(testBtn);

            item.addEventListener('click', () => selectProvider(p.id));
            group.appendChild(item);

            // Parameter panel — shown only for the selected provider
            if (isSelected && p.parameters && p.parameters.length) {
                const paramsPanel = document.createElement('div');
                paramsPanel.className = 'provider-params';
                p.parameters.forEach(spec => {
                    paramsPanel.appendChild(renderParameter(spec, providerParamValues[p.id][spec.key], val => {
                        providerParamValues[p.id][spec.key] = val;
                    }));
                });
                group.appendChild(paramsPanel);
            }
        });

        list.appendChild(group);
    });
}

function renderParameter(spec, currentValue, onChange) {
    const row = document.createElement('div');
    row.className = 'param-row';
    if (spec.help) row.title = spec.help;

    const label = document.createElement('span');
    label.className = 'param-label';
    label.textContent = spec.label;
    row.appendChild(label);

    if (spec.type === 'select') {
        const sel = document.createElement('select');
        sel.className = 'param-select';
        (spec.options || []).forEach(opt => {
            const option = document.createElement('option');
            option.value = opt.value;
            option.textContent = opt.label;
            if (opt.value === currentValue) option.selected = true;
            sel.appendChild(option);
        });
        sel.addEventListener('change', () => onChange(sel.value));
        row.appendChild(sel);
    } else if (spec.type === 'toggle') {
        const lbl = document.createElement('label');
        lbl.className = 'param-toggle-label';
        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.className = 'param-toggle';
        cb.checked = !!currentValue;
        cb.addEventListener('change', () => onChange(cb.checked));
        lbl.appendChild(cb);
        row.appendChild(lbl);
    }

    return row;
}

async function testProvider(id) {
    providerStatus[id] = 'testing';
    renderApiDropdown();
    try {
        const result = await pingProvider(id);
        providerStatus[id] = result.ok ? 'success' : 'failed';
        if (!result.ok && result.reason) {
            showToast(`${id}: ${result.reason}`, 'error');
        }
    } catch {
        providerStatus[id] = 'failed';
    }
    renderApiDropdown();
}

function selectProvider(id) {
    selectedProvider = id;
    const p = providers.find(pr => pr.id === id);
    document.getElementById('apiModelLabel').textContent = p ? p.displayName.split(' ')[0] : id;
    renderApiDropdown();
    showToast(`Provider: ${p ? p.displayName : id}`, 'info');
}

// ── Panel toggles ─────────────────────────────────────────
function toggleAdjustDrawer() {
    const drawer = document.getElementById('adjustDrawer');
    const btn = document.getElementById('adjustBtn');
    const open = drawer.classList.toggle('open');
    btn.setAttribute('aria-expanded', open);
    drawer.setAttribute('aria-hidden', !open);
}

function toggleAdvancedPanel() {
    const panel = document.getElementById('advancedPanel');
    const btn   = document.getElementById('advancedBtn');
    const open  = panel.classList.toggle('open');
    btn.textContent = open ? 'Hide' : 'Advanced';
    btn.setAttribute('aria-expanded', open);
    panel.setAttribute('aria-hidden', !open);
    if (open) renderAdvancedCanvases();
}

function resetAdvancedPanel() {
    const panel = document.getElementById('advancedPanel');
    panel.classList.remove('open');
    document.getElementById('advancedBtn').textContent = 'Advanced';
    document.getElementById('advancedBtn').setAttribute('aria-expanded', 'false');
    ['boundingCanvas','cropCanvas','filterCanvas','finalCanvas'].forEach(clearCanvas);
}

function toggleApiDropdown() {
    const dd  = document.getElementById('apiDropdown');
    const btn = document.getElementById('apiBtn');
    const open = dd.classList.toggle('hidden') ? false : true;
    btn.setAttribute('aria-expanded', open);
    if (open) renderApiDropdown();
    document.getElementById('exportDropdown').classList.add('hidden');
}

function toggleExportDropdown() {
    const dd  = document.getElementById('exportDropdown');
    const btn = document.getElementById('exportBtn');
    const open = dd.classList.toggle('hidden') ? false : true;
    btn.setAttribute('aria-expanded', open);
    document.getElementById('apiDropdown').classList.add('hidden');
}

function updateRangeValue(input) {
    const span = document.getElementById(input.id + '_value');
    if (span) span.textContent = input.value;
}

// ── Reset ─────────────────────────────────────────────────
function resetAll() {
    if (conditionController) conditionController.abort();
    if (ocrController) ocrController.abort();

    currentImage = conditionedImage = originalImageElement = conditionedImageElement = null;
    ocrRawText = '';

    document.getElementById('uploadArea').classList.remove('hidden');
    document.getElementById('inputCanvasWrapper').classList.add('hidden');
    document.getElementById('conditionBtnOverlay').classList.remove('has-image');
    document.getElementById('conditionBtn').disabled = false;
    document.getElementById('conditionBtn').innerHTML = 'Condition Image';
    document.getElementById('fileInput').value = '';

    document.getElementById('conditionedCanvas').style.display = 'none';
    document.getElementById('conditionedPlaceholder').style.display = 'flex';
    document.getElementById('ocrBtn').disabled = true;
    document.getElementById('rerunBtn').disabled = true;
    document.getElementById('advancedBtn').disabled = true;
    hideOverlay('conditionOverlay');
    document.getElementById('conditionLog').innerHTML = '';
    document.getElementById('adjustDrawer').classList.remove('open');
    resetAdvancedPanel();

    const tc = document.getElementById('textContent');
    tc.textContent = 'OCR results will appear here';
    tc.className = 'text-content empty';
    hideOverlay('ocrOverlay');
    document.getElementById('ocrLog').innerHTML = '';

    resetPipelineState();
    setActiveStage('input');
}

// ── Output type switcher ──────────────────────────────────
document.getElementById('outputType').addEventListener('click', e => {
    const btn = e.target.closest('.seg-btn');
    if (!btn) return;
    document.querySelectorAll('.seg-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    outputMode = btn.dataset.mode;
    if (ocrRawText) renderOutputText();
});

// ── Event wiring ──────────────────────────────────────────
document.getElementById('fileInput').addEventListener('change', handleFileSelect);

const uploadArea = document.getElementById('uploadArea');
uploadArea.addEventListener('click', () => document.getElementById('fileInput').click());
uploadArea.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') document.getElementById('fileInput').click(); });
uploadArea.addEventListener('dragover', handleDragOver);
uploadArea.addEventListener('dragleave', handleDragLeave);
uploadArea.addEventListener('drop', handleDrop);

document.getElementById('conditionBtn').addEventListener('click', () => conditionImage(false));
document.getElementById('ocrBtn').addEventListener('click', processOCR);
document.getElementById('rerunBtn').addEventListener('click', () => conditionImage(true));
document.getElementById('cancelConditionBtn').addEventListener('click', () => { if (conditionController) conditionController.abort(); });
document.getElementById('cancelOcrBtn').addEventListener('click', () => { if (ocrController) ocrController.abort(); });
document.getElementById('resetBtn').addEventListener('click', resetAll);
document.getElementById('adjustBtn').addEventListener('click', toggleAdjustDrawer);
document.getElementById('advancedBtn').addEventListener('click', toggleAdvancedPanel);
document.getElementById('apiBtn').addEventListener('click', toggleApiDropdown);
document.getElementById('exportBtn').addEventListener('click', toggleExportDropdown);
document.getElementById('batchOverlayClose').addEventListener('click', () => {
    document.getElementById('batchReorderOverlay').classList.add('hidden');
});

document.getElementById('strength').addEventListener('input', e => updateRangeValue(e.target));
document.getElementById('png_compression').addEventListener('input', e => updateRangeValue(e.target));

document.getElementById('exportCopy').addEventListener('click', copyResult);
document.getElementById('exportMd').addEventListener('click', () => downloadResult('md'));
document.getElementById('exportPdf').addEventListener('click', () => downloadResult('pdf'));
document.getElementById('exportDocx').addEventListener('click', () => downloadResult('docx'));

// Closed panel tabs — click to reopen
['input', 'process', 'result'].forEach(stage => {
    const tab = document.getElementById(`tab-${stage}`);
    tab.addEventListener('click', () => {
        if (getActiveStage() !== stage) setActiveStage(stage);
    });
    tab.addEventListener('keydown', e => {
        if ((e.key === 'Enter' || e.key === ' ') && getActiveStage() !== stage) setActiveStage(stage);
    });
});

// Close dropdowns when clicking outside
document.addEventListener('click', e => {
    if (!e.target.closest('#apiBtn') && !e.target.closest('#apiDropdown')) {
        document.getElementById('apiDropdown').classList.add('hidden');
        document.getElementById('apiBtn').setAttribute('aria-expanded', 'false');
    }
    if (!e.target.closest('#exportBtn') && !e.target.closest('#exportDropdown')) {
        document.getElementById('exportDropdown').classList.add('hidden');
        document.getElementById('exportBtn').setAttribute('aria-expanded', 'false');
    }
});

// Escape key — close overlays and dropdowns
document.addEventListener('keydown', e => {
    if (e.key !== 'Escape') return;
    document.getElementById('batchReorderOverlay').classList.add('hidden');
    document.getElementById('apiDropdown').classList.add('hidden');
    document.getElementById('apiBtn').setAttribute('aria-expanded', 'false');
    document.getElementById('exportDropdown').classList.add('hidden');
    document.getElementById('exportBtn').setAttribute('aria-expanded', 'false');
});

// Initialise stripe state
setActiveStage('input');
