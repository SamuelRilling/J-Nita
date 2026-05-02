import { ocrSpaceEngine3 } from "./providers/ocr-space-e3.js";

/** @type {import('./types.js').Provider[]} */
export const providers = [
  ocrSpaceEngine3
];

/** @param {string} id */
export function findProvider(id) {
  return providers.find(p => p.id === id);
}
