import { callGradioSpace } from "./_shared/hf-space-client.js";

export async function onRequestPost({ request, env }) {
    let body;
    try {
        body = await request.json();
    } catch {
        return jsonResponse({ error: "Invalid JSON body" }, 400);
    }

    const { image, params } = body;
    if (!image || typeof image !== "string") {
        return jsonResponse({ error: "Missing 'image' (base64 string) in body" }, 400);
    }

    // Defaults match config.json's image_conditioning section
    const p = params || {};
    const args = [
        image,                                  // image_b64
        p.strength ?? 10,                       // strength
        p.adaptive_block_size ?? 11,            // adaptive_block_size
        p.adaptive_C ?? 2,                      // adaptive_C
        p.morph_iterations ?? 1,                // morph_iterations
        p.target_width ?? 1280,                 // target_width
        p.target_height ?? 1792,                // target_height
        p.png_compression ?? 2,                 // png_compression
        p.return_stages ?? false                // return_stages
    ];

    try {
        const start = Date.now();
        const result = await callGradioSpace(env.HF_SPACE_BASE, "condition_image", args);
        const latencyMs = Date.now() - start;
        return jsonResponse({ ...result, latencyMs }, 200);
    } catch (err) {
        return jsonResponse({ error: err.message }, 502);
    }
}

function jsonResponse(obj, status = 200) {
    return new Response(JSON.stringify(obj), {
        status,
        headers: { "Content-Type": "application/json" }
    });
}
