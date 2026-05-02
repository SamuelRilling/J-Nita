/**
 * Client-side provider registry — metadata only.
 * Mirror of the server registry minus the call/ping functions, which can
 * only run server-side (where API keys live).
 *
 * @typedef {import('../../functions/api/_shared/types.js').ProviderMetadata} ProviderMetadata
 *
 * @type {ProviderMetadata[]}
 */
export const providers = [
  {
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
    ]
  }
];
