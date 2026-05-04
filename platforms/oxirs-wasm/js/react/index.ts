/**
 * @oxirs/react - React hooks for the OxiRS WASM RDF store
 *
 * Thin TypeScript adapters wrapping the wasm-bindgen `OxiRSStore` API.
 * Import these hooks after the WASM module has been initialised.
 *
 * @example
 * ```tsx
 * import init, { OxiRSStore } from '../../pkg/oxirs_wasm.js';
 * import { useSparqlQuery, useRdfsInference } from '@oxirs/react';
 *
 * await init();
 * const store = new OxiRSStore();
 * ```
 */
import { useEffect, useState, useCallback } from 'react';

// ---------------------------------------------------------------------------
// Minimal type aliases for the wasm-bindgen surface.
// The real types come from `../../pkg/oxirs_wasm.d.ts`; these aliases keep
// the adapters self-contained so they compile without a bundler step.
// ---------------------------------------------------------------------------

/** Binding row returned by OxiRSStore.query() */
export type SparqlBinding = Record<string, string>;

/** Anything that looks like an OxiRSStore to these hooks */
export interface OxiRSStoreRef {
  query(sparql: string): SparqlBinding[];
  insert(subject: string, predicate: string, object: string): boolean;
  inferRdfs?(): { added: number };
}

/** Options for {@link useSparqlQuery} */
export interface UseSparqlQueryOptions {
  /** If set, the query is re-executed automatically every N milliseconds. */
  refreshInterval?: number;
}

/** Return value of {@link useSparqlQuery} */
export interface UseSparqlQueryResult {
  data: SparqlBinding[] | null;
  loading: boolean;
  error: Error | null;
  refetch: () => void;
}

/** Return value of {@link useRdfsInference} */
export interface UseRdfsInferenceResult {
  /** How many triples were added on the last call. */
  added: number | null;
  running: boolean;
  run: () => void;
}

// ---------------------------------------------------------------------------
// useSparqlQuery
// ---------------------------------------------------------------------------

/**
 * Execute a SPARQL SELECT query against an OxiRS store.
 *
 * The query re-runs whenever `store` or `query` changes.  Pass
 * `options.refreshInterval` (milliseconds) for live-polling semantics.
 *
 * @param store  - An initialised `OxiRSStore` instance, or `null`.
 * @param query  - A SPARQL SELECT query string.
 * @param options - Optional configuration.
 */
export function useSparqlQuery(
  store: OxiRSStoreRef | null,
  query: string,
  options?: UseSparqlQueryOptions,
): UseSparqlQueryResult {
  const [data, setData] = useState<SparqlBinding[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const execute = useCallback(() => {
    if (!store || !query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const result = store.query(query);
      setData(result);
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
    } finally {
      setLoading(false);
    }
  }, [store, query]);

  useEffect(() => {
    execute();
    if (options?.refreshInterval != null && options.refreshInterval > 0) {
      const id = setInterval(execute, options.refreshInterval);
      return () => clearInterval(id);
    }
    return undefined;
  }, [execute, options?.refreshInterval]);

  return { data, loading, error, refetch: execute };
}

// ---------------------------------------------------------------------------
// useAddTriple
// ---------------------------------------------------------------------------

/**
 * Returns a stable callback for inserting a single triple into the store.
 *
 * The callback returns `true` if the triple was novel, `false` if it already
 * existed.
 *
 * @param store - An initialised `OxiRSStore` instance, or `null`.
 */
export function useAddTriple(
  store: OxiRSStoreRef | null,
): (subject: string, predicate: string, object: string) => boolean {
  return useCallback(
    (subject: string, predicate: string, object: string): boolean => {
      if (!store) return false;
      return store.insert(subject, predicate, object);
    },
    [store],
  );
}

// ---------------------------------------------------------------------------
// useRdfsInference
// ---------------------------------------------------------------------------

/**
 * Exposes RDFS forward-chaining inference on an OxiRS store as a React hook.
 *
 * Call `run()` to materialise all RDFS-derivable triples.  The hook tracks
 * the count of newly added triples and a `running` flag.
 *
 * Rules applied: rdfs:subClassOf/subPropertyOf transitivity, rdf:type
 * propagation (rdfs9/rdfs11), domain/range typing (rdfs2/rdfs3),
 * subPropertyOf usage propagation (rdfs7).
 *
 * @param store - An initialised `OxiRSStore` instance, or `null`.
 */
export function useRdfsInference(store: OxiRSStoreRef | null): UseRdfsInferenceResult {
  const [added, setAdded] = useState<number | null>(null);
  const [running, setRunning] = useState(false);

  const run = useCallback(() => {
    if (!store) return;
    setRunning(true);
    try {
      const result = store.inferRdfs?.() ?? { added: 0 };
      setAdded(result.added);
    } finally {
      setRunning(false);
    }
  }, [store]);

  return { added, running, run };
}
