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
  if (!image || typeof image !== "string") {
    return jsonResponse({ error: "Missing 'image' (base64 string) in body" }, 400);
  }

  const provider = findProvider(providerId);
  if (!provider) {
    return jsonResponse({ error: `Unknown provider: ${providerId}` }, 400);
  }

  let imageBuffer;
  try {
    const b64 = image.includes(",") ? image.split(",", 2)[1] : image;
    const binaryString = atob(b64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    imageBuffer = bytes.buffer;
  } catch (err) {
    return jsonResponse({ error: `Invalid base64 image: ${err.message}` }, 400);
  }

  try {
    const result = await provider.call(imageBuffer, options || {}, env);
    return jsonResponse(result, 200);
  } catch (err) {
    return jsonResponse({ error: err.message, providerId }, 502);
  }
}

function jsonResponse(obj, status = 200) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { "Content-Type": "application/json" }
  });
}
