/** @type {Provider[]} */
export const providers = [];

/** @param {string} id */
export function findProvider(id) {
    return providers.find(p => p.id === id);
}
