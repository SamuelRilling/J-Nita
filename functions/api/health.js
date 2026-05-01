import { findProvider } from "./_shared/providers.js";

export async function onRequestPost({ request, env }) {
    let body;
    try { body = await request.json(); } catch { body = {}; }
    const { providerId } = body;

    if (!providerId) {
        return jsonResponse({ error: "Missing providerId" }, 400);
    }

    const provider = findProvider(providerId);
    if (!provider) {
        return jsonResponse({
            ok: false,
            latencyMs: 0,
            reason: "Provider not found (no providers implemented yet)"
        }, 200);  // 200 because the API itself worked; ok:false signals provider issue
    }

    return jsonResponse({ error: "Not yet implemented" }, 501);
}

function jsonResponse(obj, status = 200) {
    return new Response(JSON.stringify(obj), {
        status,
        headers: { "Content-Type": "application/json" }
    });
}
