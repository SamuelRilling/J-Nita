/**
 * @typedef {"select"|"slider"|"text"|"toggle"} ParameterType
 *
 * @typedef {Object} ParameterSpec
 * @property {string} key
 * @property {string} label
 * @property {ParameterType} type
 * @property {any} default
 * @property {Array<{value:any,label:string}>} [options]  // for "select"
 * @property {number} [min]                                // for "slider"
 * @property {number} [max]                                // for "slider"
 * @property {number} [step]                               // for "slider"
 * @property {string} [help]
 *
 * @typedef {Object} OcrResult
 * @property {string} text                  - Plain text output
 * @property {string} markdown              - Markdown-formatted (may equal text)
 * @property {number} [confidence]          - 0-1 if provider returns one
 * @property {number} latencyMs             - Measured by the router
 * @property {string} providerId            - Echoed back
 * @property {Object} [raw]                 - Provider's raw response
 *
 * @typedef {Object} HealthResult
 * @property {boolean} ok
 * @property {number} latencyMs
 * @property {string} [reason]              - Set when !ok
 *
 * @typedef {"free"|"paid"} ProviderTier
 * @typedef {"handwriting"|"general-ocr"|"vision-llm"|"self-hosted"} ProviderCategory
 * @typedef {"us"|"eu"|"global"|"self"} HostingRegion
 *
 * @typedef {Object} ProviderMetadata
 * @property {string} id                    - Stable kebab-case id
 * @property {string} displayName
 * @property {ProviderTier} tier
 * @property {ProviderCategory} category
 * @property {string[]} qualitativeTags
 * @property {HostingRegion} hosting
 * @property {string[]} languages
 * @property {ParameterSpec[]} parameters
 *
 * @typedef {Object} Provider
 * @extends ProviderMetadata
 * @property {(image: ArrayBuffer, options: Object, env: Env) => Promise<OcrResult>} call
 * @property {(env: Env) => Promise<HealthResult>} ping
 *
 * @typedef {Object} Env
 * @property {string} OCR_SPACE_API_KEY
 * @property {string} AZURE_DOC_INTELLIGENCE_KEY
 * @property {string} AZURE_DOC_INTELLIGENCE_ENDPOINT
 * @property {string} GEMINI_API_KEY
 * @property {string} HF_SPACE_BASE
 */

export {};  // make this a module
