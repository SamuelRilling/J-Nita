import { findProvider } from "./_shared/providers.js";

export async function onRequestPost({ request, env }) {
    let body;
    try {
        body = await request.json();
    } catch {
        return jsonResponse({ error: "Invalid JSON body" }, 400);
    }

    const { providerId, image, options } = body;
    if (!providerId) {
        return jsonResponse({ error: "Missing providerId" }, 400);
    }

    const provider = findProvider(providerId);
    if (!provider) {
        return jsonResponse({
            error: "No providers are implemented yet. This endpoint will be wired up in Phase 3."
        }, 501);
    }

    // Phase 3+ will actually call provider.call() here
    return jsonResponse({ error: "Not yet implemented" }, 501);
}

function jsonResponse(obj, status = 200) {
    return new Response(JSON.stringify(obj), {
        status,
        headers: { "Content-Type": "application/json" }
    });
}
