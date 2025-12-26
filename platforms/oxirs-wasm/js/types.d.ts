/**
 * OxiRS WASM TypeScript definitions
 */

declare module 'oxirs-wasm' {
    /**
     * Initialize the WASM module
     */
    export function initialize(): Promise<void>;

    /**
     * Create a new RDF store
     */
    export function createStore(): Promise<OxiRSStore>;

    /**
     * Get the version of OxiRS WASM
     */
    export function getVersion(): Promise<string>;

    /**
     * Log a message to the browser console
     */
    export function log(message: string): void;

    /**
     * RDF Triple
     */
    export class Triple {
        constructor(subject: string, predicate: string, object: string);

        readonly subject: string;
        readonly predicate: string;
        readonly object: string;
    }

    /**
     * In-memory RDF store
     */
    export class OxiRSStore {
        constructor();

        /**
         * Load Turtle data
         * @param turtle - Turtle formatted RDF data
         * @returns Number of triples loaded
         */
        loadTurtle(turtle: string): Promise<number>;

        /**
         * Load N-Triples data
         * @param ntriples - N-Triples formatted RDF data
         * @returns Number of triples loaded
         */
        loadNTriples(ntriples: string): Promise<number>;

        /**
         * Insert a single triple
         */
        insert(subject: string, predicate: string, object: string): boolean;

        /**
         * Delete a single triple
         */
        delete(subject: string, predicate: string, object: string): boolean;

        /**
         * Check if a triple exists
         */
        contains(subject: string, predicate: string, object: string): boolean;

        /**
         * Get the number of triples
         */
        size(): number;

        /**
         * Clear all triples
         */
        clear(): void;

        /**
         * Execute a SPARQL SELECT query
         * @param sparql - SPARQL query string
         * @returns Array of binding objects
         */
        query(sparql: string): Promise<QueryResult[]>;

        /**
         * Execute a SPARQL ASK query
         * @param sparql - SPARQL query string
         * @returns Boolean result
         */
        ask(sparql: string): Promise<boolean>;

        /**
         * Execute a SPARQL CONSTRUCT query
         * @param sparql - SPARQL query string
         * @returns Array of Triple objects
         */
        construct(sparql: string): Promise<Triple[]>;

        /**
         * Export to Turtle format
         */
        toTurtle(): string;

        /**
         * Export to N-Triples format
         */
        toNTriples(): string;

        /**
         * Get all subjects
         */
        subjects(): string[];

        /**
         * Get all predicates
         */
        predicates(): string[];

        /**
         * Get all objects
         */
        objects(): string[];

        /**
         * Add a namespace prefix
         */
        addPrefix(prefix: string, uri: string): void;
    }

    /**
     * Query result binding
     */
    export interface QueryResult {
        [variable: string]: string;
    }

    /**
     * Validation report
     */
    export interface ValidationReport {
        conforms: boolean;
        results: ValidationResult[];
    }

    /**
     * Validation result
     */
    export interface ValidationResult {
        focusNode: string;
        resultPath?: string;
        value?: string;
        message: string;
        severity: 'Violation' | 'Warning' | 'Info';
    }
}
