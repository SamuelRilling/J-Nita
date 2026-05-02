/** @typedef {import('../types.js').Provider} Provider */
/** @typedef {import('../types.js').Env} Env */

/** @type {Provider} */
export const ocrSpaceEngine3 = {
  id: "ocr-space-e3",
  displayName: "OCR.space (Engine 3)",
  tier: "free",
  category: "handwriting",
  qualitativeTags: ["Free", "Fast", "Handwriting", "Latin scripts", "US-hosted"],
  hosting: "us",
  languages: ["eng", "ger", "spa", "fre", "ita", "por", "dut", "pol", "rus"],
  parameters: [
    {
      key: "language",
      label: "Language",
      type: "select",
      default: "eng",
      help: "Primary language of the handwritten text",
      options: [
        { value: "eng", label: "English" },
        { value: "ger", label: "German" },
        { value: "spa", label: "Spanish" },
        { value: "fre", label: "French" },
        { value: "ita", label: "Italian" },
        { value: "por", label: "Portuguese" },
        { value: "dut", label: "Dutch" },
        { value: "pol", label: "Polish" },
        { value: "rus", label: "Russian" }
      ]
    },
    {
      key: "scale",
      label: "Upscale before OCR",
      type: "toggle",
      default: true,
      help: "Improves accuracy for low-resolution images. Slightly slower."
    },
    {
      key: "isTable",
      label: "Detect as table",
      type: "toggle",
      default: false,
      help: "Enable if the image contains tabular data"
    }
  ],

  /**
   * Run OCR on an image via OCR.space Engine 3.
   * @param {ArrayBuffer} imageBuffer
   * @param {{language?:string, scale?:boolean, isTable?:boolean}} options
   * @param {Env} env
   */
  async call(imageBuffer, options, env) {
    if (!env.OCR_SPACE_API_KEY) {
      throw new Error("OCR_SPACE_API_KEY not configured in environment");
    }

    const formData = new FormData();
    formData.append("file", new Blob([imageBuffer], { type: "image/png" }), "image.png");
    formData.append("apikey", env.OCR_SPACE_API_KEY);
    formData.append("OCREngine", "3");
    formData.append("language", options.language || "eng");
    formData.append("scale", options.scale === false ? "false" : "true");
    formData.append("isTable", options.isTable ? "true" : "false");
    formData.append("isOverlayRequired", "false");
    formData.append("detectOrientation", "true");
    formData.append("filetype", "PNG");

    const start = Date.now();
    const response = await fetch("https://api.ocr.space/parse/image", {
      method: "POST",
      body: formData
    });
    const latencyMs = Date.now() - start;

    if (!response.ok) {
      throw new Error(`OCR.space HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();

    if (data.IsErroredOnProcessing) {
      const errMsg = Array.isArray(data.ErrorMessage)
        ? data.ErrorMessage.join("; ")
        : (data.ErrorMessage || "Unknown OCR.space error");
      throw new Error(`OCR.space: ${errMsg}`);
    }

    if (!data.ParsedResults || data.ParsedResults.length === 0) {
      throw new Error("OCR.space returned no parsed results");
    }

    const text = data.ParsedResults[0].ParsedText || "";

    return {
      text,
      markdown: text,
      latencyMs,
      providerId: "ocr-space-e3",
      raw: data
    };
  },

  /**
   * Health check.
   * @param {Env} env
   */
  async ping(env) {
    const start = Date.now();

    if (!env.OCR_SPACE_API_KEY) {
      return {
        ok: false,
        latencyMs: 0,
        reason: "OCR_SPACE_API_KEY not configured"
      };
    }

    // Tiny 8x8 white PNG, base64-encoded inline.
    const tinyPngBase64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";
    const binary = atob(tinyPngBase64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

    try {
      const formData = new FormData();
      formData.append("file", new Blob([bytes], { type: "image/png" }), "ping.png");
      formData.append("apikey", env.OCR_SPACE_API_KEY);
      formData.append("OCREngine", "1");
      formData.append("isOverlayRequired", "false");

      console.log("DEBUG ping: sending request");
      const response = await fetch("https://api.ocr.space/parse/image", {
        method: "POST",
        body: formData
      });
      const latencyMs = Date.now() - start;

      console.log("DEBUG ping: response status =", response.status);
      
      // Read the body regardless of status, so we can see what OCR.space said
      const bodyText = await response.text();
      console.log("DEBUG ping: response body =", bodyText.slice(0, 500));

      if (!response.ok) {
        return { ok: false, latencyMs, reason: `HTTP ${response.status}: ${bodyText.slice(0, 200)}` };
      }

      let data;
      try {
        data = JSON.parse(bodyText);
      } catch (e) {
        return { ok: false, latencyMs, reason: `Invalid JSON response: ${bodyText.slice(0, 100)}` };
      }

      if (data.IsErroredOnProcessing) {
        const errMsg = Array.isArray(data.ErrorMessage)
          ? data.ErrorMessage[0]
          : (data.ErrorMessage || "Unknown error");
        if (/api\s*key|unauthorized|invalid/i.test(errMsg)) {
          return { ok: false, latencyMs, reason: errMsg };
        }
        return { ok: true, latencyMs };
      }

      return { ok: true, latencyMs };
    } catch (err) {
      return {
        ok: false,
        latencyMs: Date.now() - start,
        reason: err.message || "Network error"
      };
    }
  }
  
};