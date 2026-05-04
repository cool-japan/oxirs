/**
 * @oxirs/vue - Vue 3 composables for the OxiRS WASM RDF store
 *
 * Thin TypeScript adapters wrapping the wasm-bindgen `OxiRSStore` API as
 * reactive Vue 3 composables.
 *
 * @example
 * ```ts
 * import init, { OxiRSStore } from '../../pkg/oxirs_wasm.js';
 * import { useSparqlQuery, useRdfsInference } from '@oxirs/vue';
 *
 * const store = shallowRef<OxiRSStore | null>(null);
 * await init();
 * store.value = new OxiRSStore();
 *
 * const { data, loading, error } = useSparqlQuery(store, computed(() => myQuery));
 * ```
 */
import { ref, watch, computed, Ref, ComputedRef } from 'vue';

// ---------------------------------------------------------------------------
// Type aliases (mirrors the wasm-bindgen surface without importing it)
// ---------------------------------------------------------------------------

/** Binding row returned by OxiRSStore.query() */
export type SparqlBinding = Record<string, string>;

/** Anything that looks like an OxiRSStore to these composables */
export interface OxiRSStoreRef {
  query(sparql: string): SparqlBinding[];
  insert(subject: string, predicate: string, object: string): boolean;
  inferRdfs?(): { added: number };
}

// ---------------------------------------------------------------------------
// useSparqlQuery
// ---------------------------------------------------------------------------

/** Return type of {@link useSparqlQuery} */
export interface UseSparqlQueryComposable {
  data: Ref<SparqlBinding[] | null>;
  loading: Ref<boolean>;
  error: Ref<Error | null>;
  refetch: () => void;
}

/**
 * Composable that executes a SPARQL SELECT query reactively.
 *
 * The query re-runs whenever `store` or `query` changes.  Both arguments
 * may be plain refs, computed refs, or getter functions.
 *
 * @param store - A ref wrapping an `OxiRSStore` instance, or `null`.
 * @param query - A ref or computed ref holding the SPARQL query string.
 */
export function useSparqlQuery(
  store: Ref<OxiRSStoreRef | null>,
  query: Ref<string> | ComputedRef<string>,
): UseSparqlQueryComposable {
  const data = ref<SparqlBinding[] | null>(null) as Ref<SparqlBinding[] | null>;
  const loading = ref(false);
  const error = ref<Error | null>(null);

  function execute(): void {
    const s = store.value;
    const q = query.value;
    if (!s || !q.trim()) return;
    loading.value = true;
    error.value = null;
    try {
      data.value = s.query(q);
    } catch (e) {
      error.value = e instanceof Error ? e : new Error(String(e));
    } finally {
      loading.value = false;
    }
  }

  watch([store, query], execute, { immediate: true });

  return { data, loading, error, refetch: execute };
}

// ---------------------------------------------------------------------------
// useAddTriple
// ---------------------------------------------------------------------------

/**
 * Composable that returns a function for inserting a single triple.
 *
 * @param store - A ref wrapping an `OxiRSStore` instance, or `null`.
 */
export function useAddTriple(
  store: Ref<OxiRSStoreRef | null>,
): (subject: string, predicate: string, object: string) => boolean {
  return (subject: string, predicate: string, object: string): boolean => {
    const s = store.value;
    if (!s) return false;
    return s.insert(subject, predicate, object);
  };
}

// ---------------------------------------------------------------------------
// useRdfsInference
// ---------------------------------------------------------------------------

/** Return type of {@link useRdfsInference} */
export interface UseRdfsInferenceComposable {
  /** Number of triples added on the last inference pass. `null` before first run. */
  added: Ref<number | null>;
  running: Ref<boolean>;
  infer: () => void;
}

/**
 * Composable exposing RDFS forward-chaining entailment on an OxiRS store.
 *
 * Call `infer()` to materialise all derivable triples.  Subsequent calls are
 * idempotent — `added` will be 0 after the fixed-point is reached.
 *
 * @param store - A ref wrapping an `OxiRSStore` instance, or `null`.
 */
export function useRdfsInference(store: Ref<OxiRSStoreRef | null>): UseRdfsInferenceComposable {
  const added = ref<number | null>(null) as Ref<number | null>;
  const running = ref(false);

  function infer(): void {
    const s = store.value;
    if (!s) return;
    running.value = true;
    try {
      const result = s.inferRdfs?.() ?? { added: 0 };
      added.value = result.added;
    } finally {
      running.value = false;
    }
  }

  return { added, running, infer };
}

// ---------------------------------------------------------------------------
// useTripleCount
// ---------------------------------------------------------------------------

/**
 * Reactive triple count derived from the store.
 * Useful for triggering UI updates after mutations.
 *
 * @param store - A ref wrapping an `OxiRSStore` instance, or `null`.
 * @param deps  - Additional refs whose changes should re-compute the count.
 */
export function useTripleCount(
  store: Ref<OxiRSStoreRef | null>,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  deps?: Ref<any>[],
): ComputedRef<number> {
  const trigger = ref(0);

  if (deps && deps.length > 0) {
    watch(deps, () => { trigger.value++; });
  }

  return computed(() => {
    // Access trigger so Vue tracks it
    void trigger.value;
    const s = store.value;
    if (!s) return 0;
    // OxiRSStore exposes size() as a plain method
    return typeof (s as { size?: () => number }).size === 'function'
      ? ((s as { size: () => number }).size())
      : 0;
  });
}
