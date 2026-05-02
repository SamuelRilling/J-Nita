/**
 * Call a Gradio 5 endpoint on a Hugging Face Space.
 * Implements the two-step pattern:
 *   1. POST /gradio_api/call/<fn> → {event_id}
 *   2. GET /gradio_api/call/<fn>/<event_id> → SSE stream, parse data line
 *
 * @param {string} baseUrl - e.g. "https://user-space.hf.space"
 * @param {string} fnName  - e.g. "condition_image"
 * @param {any[]} dataArgs - positional args matching the Gradio function signature
 * @returns {Promise<any>}  - the parsed result from the SSE "data:" line
 */
export async function callGradioSpace(baseUrl, fnName, dataArgs) {
    // Step 1: POST to get event_id
    const postResp = await fetch(`${baseUrl}/gradio_api/call/${fnName}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ data: dataArgs })
    });
    if (!postResp.ok) {
        throw new Error(`HF Space POST failed: ${postResp.status} ${postResp.statusText}`);
    }
    const { event_id } = await postResp.json();
    if (!event_id) {
        throw new Error("HF Space did not return event_id");
    }

    // Step 2: GET to retrieve result via SSE
    const getResp = await fetch(`${baseUrl}/gradio_api/call/${fnName}/${event_id}`, {
        method: "GET"
    });
    if (!getResp.ok) {
        throw new Error(`HF Space GET failed: ${getResp.status} ${getResp.statusText}`);
    }

    // Parse SSE: look for "event: complete" followed by "data: <json>"
    const text = await getResp.text();
    const lines = text.split("\n");
    let foundComplete = false;
    for (const line of lines) {
        if (line.startsWith("event: complete")) {
            foundComplete = true;
            continue;
        }
        if (foundComplete && line.startsWith("data: ")) {
            const payload = line.slice(6); // strip "data: "
            const parsed = JSON.parse(payload);
            // Gradio wraps single returns in an array of length 1
            return Array.isArray(parsed) ? parsed[0] : parsed;
        }
    }
    throw new Error(`HF Space response did not contain expected SSE data line. Raw: ${text.slice(0, 500)}`);
}
