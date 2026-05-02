/**
 * Condition an image via the backend's HF Space proxy.
 * @param {string} imageBase64 - data URL or raw base64 string
 * @param {Object} [params] - conditioning parameters (strength, etc.)
 */
export async function conditionImage(imageBase64, params = {}) {
    const resp = await fetch('/api/condition', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imageBase64, params })
    });
    if (!resp.ok) {
        const errBody = await resp.json().catch(() => ({}));
        throw new Error(errBody.error || `HTTP ${resp.status}`);
    }
    return resp.json();
}

/**
 * Run OCR via the backend's provider router.
 * @param {string} providerId
 * @param {string} imageBase64
 * @param {Object} [options]
 */
export async function runOcr(providerId, imageBase64, options = {}) {
    const resp = await fetch('/api/ocr', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ providerId, image: imageBase64, options })
    });
    if (!resp.ok) {
        const errBody = await resp.json().catch(() => ({}));
        throw new Error(errBody.error || `HTTP ${resp.status}`);
    }
    return resp.json();
}

/**
 * Ping a provider for health/availability.
 * @param {string} providerId
 */
export async function pingProvider(providerId) {
    const resp = await fetch('/api/health', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ providerId })
    });
    return resp.json();
}
