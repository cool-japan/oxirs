/**
 * @oxirs/svelte - Svelte stores for the OxiRS WASM RDF store
 *
 * Thin TypeScript adapters wrapping the wasm-bindgen `OxiRSStore` API as
 * Svelte 4 readable/writable stores.
 *
 * @example
 * ```svelte
 * <script lang="ts">
 *   import init, { OxiRSStore } from '../../pkg/oxirs_wasm.js';
 *   import { createSparqlStore, createRdfsStore } from '@oxirs/svelte';
 *   import { writable } from 'svelte/store';
 *
 *   const storeRef = writable<OxiRSStore | null>(null);
 *   init().then(() => { storeRef.set(new OxiRSStore()); });
 *
 *   const sparql = createSparqlStore(storeRef, 'SELECT * WHERE { ?s ?p ?o }');
 *   const rdfs   = createRdfsStore(storeRef);
 * </script>
 * ```
 */
import { writable, derived, get, type Readable, type Writable } from 'svelte/store';

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

/** Binding row returned by OxiRSStore.query() */
export type SparqlBinding = Record<string, string>;

/** Anything that looks like an OxiRSStore to these stores */
export interface OxiRSStoreRef {
  query(sparql: string): SparqlBinding[];
  insert(subject: string, predicate: string, object: string): boolean;
  inferRdfs?(): { added: number };
}

// ---------------------------------------------------------------------------
// createSparqlStore
// ---------------------------------------------------------------------------

/** State shape for {@link createSparqlStore} */
export interface SparqlState {
  data: SparqlBinding[] | null;
  loading: boolean;
  error: Error | null;
}

/** Handle returned by {@link createSparqlStore} */
export interface SparqlStoreHandle {
  subscribe: Readable<SparqlState>['subscribe'];
  refetch: () => void;
}

/**
 * Creates a Svelte readable store tracking the result of a SPARQL SELECT.
 *
 * Call `refetch()` to re-execute the query at any point.  The store
 * starts in the `loading: false, data: null` state until the first fetch.
 *
 * @param storeRef - A Svelte store holding an `OxiRSStore` instance or `null`.
 * @param queryStr - A SPARQL SELECT query string.
 */
export function createSparqlStore(
  storeRef: Readable<OxiRSStoreRef | null>,
  queryStr: string,
): SparqlStoreHandle {
  const state: Writable<SparqlState> = writable({ data: null, loading: false, error: null });

  function refetch(): void {
    const s = get(storeRef);
    if (!s || !queryStr.trim()) return;
    state.update(st => ({ ...st, loading: true, error: null }));
    try {
      const data = s.query(queryStr);
      state.update(st => ({ ...st, data, loading: false }));
    } catch (e) {
      const error = e instanceof Error ? e : new Error(String(e));
      state.update(st => ({ ...st, error, loading: false }));
    }
  }

  return { subscribe: state.subscribe, refetch };
}

// ---------------------------------------------------------------------------
// createRdfsStore
// ---------------------------------------------------------------------------

/** Handle returned by {@link createRdfsStore} */
export interface RdfsStoreHandle {
  /** Number of triples added on the last inference pass. */
  added: Readable<number>;
  /** Whether inference is currently running. */
  running: Readable<boolean>;
  /** Trigger RDFS forward-chaining materialisation. */
  infer: () => void;
}

/**
 * Creates Svelte stores for RDFS forward-chaining inference.
 *
 * Call `infer()` to materialise all derivable triples.  The `added` store
 * updates with the count of newly inserted triples.  Subsequent calls are
 * idempotent once the fixed-point is reached.
 *
 * @param storeRef - A Svelte store holding an `OxiRSStore` instance or `null`.
 */
export function createRdfsStore(storeRef: Readable<OxiRSStoreRef | null>): RdfsStoreHandle {
  const addedStore: Writable<number> = writable(0);
  const runningStore: Writable<boolean> = writable(false);

  function infer(): void {
    const s = get(storeRef);
    if (!s) return;
    runningStore.set(true);
    try {
      const result = s.inferRdfs?.() ?? { added: 0 };
      addedStore.set(result.added);
    } finally {
      runningStore.set(false);
    }
  }

  return { added: addedStore, running: runningStore, infer };
}

// ---------------------------------------------------------------------------
// createTripleStore  (reactive store-backed triple collection)
// ---------------------------------------------------------------------------

/** Handle returned by {@link createTripleStore} */
export interface TripleStoreHandle {
  /** Current triple count, updated after each mutation. */
  count: Readable<number>;
  /** Insert a triple; returns `true` if the triple was novel. */
  insert: (subject: string, predicate: string, object: string) => boolean;
}

/**
 * Creates a reactive triple count store alongside an insert helper.
 *
 * @param storeRef - A Svelte store holding an `OxiRSStore` instance or `null`.
 */
export function createTripleStore(storeRef: Readable<OxiRSStoreRef | null>): TripleStoreHandle {
  const triggerStore: Writable<number> = writable(0);

  const count = derived([storeRef, triggerStore], ([$store, _trigger]) => {
    if (!$store) return 0;
    return typeof ($store as { size?: () => number }).size === 'function'
      ? (($store as { size: () => number }).size())
      : 0;
  });

  function insert(subject: string, predicate: string, object: string): boolean {
    const s = get(storeRef);
    if (!s) return false;
    const inserted = s.insert(subject, predicate, object);
    if (inserted) {
      triggerStore.update(n => n + 1);
    }
    return inserted;
  }

  return { count, insert };
}
