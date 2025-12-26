/**
 * OxiRS WASM - JavaScript wrapper
 *
 * @module oxirs-wasm
 */

import init, { OxiRSStore, Triple, version, log } from '../pkg/oxirs_wasm.js';

let initialized = false;

/**
 * Initialize the WASM module
 * @returns {Promise<void>}
 */
export async function initialize() {
    if (!initialized) {
        await init();
        initialized = true;
    }
}

/**
 * Create a new RDF store
 * @returns {Promise<OxiRSStore>}
 */
export async function createStore() {
    await initialize();
    return new OxiRSStore();
}

/**
 * Get the version of OxiRS WASM
 * @returns {Promise<string>}
 */
export async function getVersion() {
    await initialize();
    return version();
}

// Re-export classes
export { OxiRSStore, Triple, log };
