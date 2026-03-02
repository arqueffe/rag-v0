import { useEffect, useState } from "react";
import type { BaseRetrieverInterface } from "@langchain/core/retrievers";
import type { Document } from "@langchain/core/documents";
import { chunkDocuments } from "./lib/chunking";
import { BrowserTransformersEmbeddings } from "./lib/embeddings";
import { loadPdfAsDocuments } from "./lib/pdf";
import { createRetriever, queryWithRetriever } from "./lib/retrieval";
import { AVAILABLE_VECTOR_STORES, createVectorStore } from "./lib/vectorstore";
import type {
  ChunkingConfig,
  LocalVectorStore,
  ModelDownloadProgress,
  QueryResult,
  SearchConfig,
  VectorStoreOption
} from "./types";

const embeddings = new BrowserTransformersEmbeddings();

const defaultChunkingConfig: ChunkingConfig = {
  strategy: "recursive",
  fixed: {
    chunkSize: 200,
    chunkOverlap: 20,
    separator: " "
  },
  recursive: {
    chunkSize: 500,
    chunkOverlap: 100,
    separators: ["\\n\\n", "\\n", ". ", " ", ""]
  }
};

const defaultSearchConfig: SearchConfig = {
  searchType: "mmr",
  k: 2,
  fetchK: 8,
  lambda: 0.5
};

function parseNumber(value: string, fallback: number): number {
  const num = Number(value);
  return Number.isFinite(num) ? num : fallback;
}

function getResultPageNumber(result: QueryResult): number | null {
  const rawPage = result.document.metadata?.loc?.pageNumber;
  const rawDirectPage = result.document.metadata?.page;

  if (typeof rawPage === "number" && Number.isFinite(rawPage) && rawPage >= 1) {
    return Math.floor(rawPage);
  }

  if (typeof rawPage === "string") {
    const parsed = Number(rawPage);
    if (Number.isFinite(parsed) && parsed >= 1) {
      return Math.floor(parsed);
    }
  }

  if (typeof rawDirectPage === "number" && Number.isFinite(rawDirectPage)) {
    if (rawDirectPage >= 1) {
      return Math.floor(rawDirectPage);
    }

    if (rawDirectPage >= 0) {
      return Math.floor(rawDirectPage) + 1;
    }
  }

  if (typeof rawDirectPage === "string") {
    const parsed = Number(rawDirectPage);
    if (Number.isFinite(parsed)) {
      if (parsed >= 1) {
        return Math.floor(parsed);
      }

      if (parsed >= 0) {
        return Math.floor(parsed) + 1;
      }
    }
  }

  return null;
}

function buildViewerSrc(pdfUrl: string, page: number): string {
  return `${pdfUrl}#page=${Math.max(1, page)}&zoom=page-fit`;
}

function formatMetric(value: number): string {
  return value.toFixed(3);
}

export default function App() {
  const [modelReady, setModelReady] = useState(false);
  const [downloadState, setDownloadState] = useState<ModelDownloadProgress | null>(null);
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const [isDownloading, setIsDownloading] = useState(false);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [chunks, setChunks] = useState<Document[]>([]);

  const [chunkingConfig, setChunkingConfig] = useState<ChunkingConfig>(defaultChunkingConfig);
  const [isChunking, setIsChunking] = useState(false);
  const [chunkError, setChunkError] = useState<string | null>(null);

  const [vectorStoreType, setVectorStoreType] = useState<VectorStoreOption>("memory");
  const [vectorStore, setVectorStore] = useState<LocalVectorStore | null>(null);
  const [isIndexing, setIsIndexing] = useState(false);
  const [vectorStoreError, setVectorStoreError] = useState<string | null>(null);
  const [indexingProgress, setIndexingProgress] = useState({ indexed: 0, total: 0 });

  const [searchConfig, setSearchConfig] = useState<SearchConfig>(defaultSearchConfig);
  const [retriever, setRetriever] = useState<BaseRetrieverInterface | null>(null);
  const [retrieverError, setRetrieverError] = useState<string | null>(null);

  const [query, setQuery] = useState("");
  const [results, setResults] = useState<QueryResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [pdfObjectUrl, setPdfObjectUrl] = useState<string | null>(null);
  const [viewerPage, setViewerPage] = useState(1);
  const [viewerSrc, setViewerSrc] = useState("");
  const [viewerKey, setViewerKey] = useState(0);

  const canPickPdf = modelReady;
  const canChunk = canPickPdf && documents.length > 0;
  const canIndex = chunks.length > 0;
  const canCreateRetriever = vectorStore !== null && !isIndexing;
  const canSearch = retriever !== null;

  const modelProgressPercent = (() => {
    if (!downloadState) {
      return 0;
    }
    return Math.round(Math.max(0, Math.min(1, downloadState.progress)) * 100);
  })();

  const indexingProgressPercent = (() => {
    if (!indexingProgress.total) {
      return 0;
    }

    return Math.round((indexingProgress.indexed / indexingProgress.total) * 100);
  })();

  useEffect(() => {
    if (!selectedFile) {
      setPdfObjectUrl(null);
      setViewerPage(1);
      setViewerSrc("");
      return;
    }

    const objectUrl = URL.createObjectURL(selectedFile);
    setPdfObjectUrl(objectUrl);
    setViewerPage(1);
    setViewerSrc(buildViewerSrc(objectUrl, 1));
    setViewerKey((current) => current + 1);

    return () => {
      URL.revokeObjectURL(objectUrl);
    };
  }, [selectedFile]);

  function jumpViewerToPage(page: number) {
    if (!pdfObjectUrl) {
      return;
    }

    const safePage = Math.max(1, page);
    setViewerPage(safePage);

    setViewerSrc("");
    setViewerKey((current) => current + 1);

    setTimeout(() => {
      setViewerSrc(buildViewerSrc(pdfObjectUrl, safePage));
      setViewerKey((current) => current + 1);
    }, 0);
  }

  async function handleDownloadModel() {
    setIsDownloading(true);
    setDownloadError(null);

    try {
      await embeddings.downloadModel((progress) => {
        setDownloadState(progress);
      });
      setModelReady(true);
    } catch (error) {
      setDownloadError(error instanceof Error ? error.message : "Failed to download model");
      setModelReady(false);
    } finally {
      setIsDownloading(false);
    }
  }

  async function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null;
    setSelectedFile(file);
    setDocuments([]);
    setChunks([]);
    setVectorStore(null);
    setIndexingProgress({ indexed: 0, total: 0 });
    setRetriever(null);
    setResults([]);

    if (!file) {
      return;
    }

    const loaded = await loadPdfAsDocuments(file);
    setDocuments(loaded);
  }

  async function handleChunkDocuments() {
    setIsChunking(true);
    setChunkError(null);
    setVectorStore(null);
    setIndexingProgress({ indexed: 0, total: 0 });
    setRetriever(null);
    setResults([]);

    try {
      const generated = await chunkDocuments(documents, chunkingConfig);
      setChunks(generated);
    } catch (error) {
      setChunkError(error instanceof Error ? error.message : "Failed to chunk document");
      setChunks([]);
    } finally {
      setIsChunking(false);
    }
  }

  async function handleIndexChunks() {
    setIsIndexing(true);
    setVectorStoreError(null);
    setVectorStore(null);
    setIndexingProgress({ indexed: 0, total: chunks.length });
    setRetriever(null);
    setResults([]);

    try {
      const store = await createVectorStore(vectorStoreType, chunks, embeddings, {
        batchSize: 24,
        onProgress: (progress) => {
          setIndexingProgress(progress);
        }
      });
      setVectorStore(store);
    } catch (error) {
      setVectorStoreError(error instanceof Error ? error.message : "Failed to create vector store");
      setIndexingProgress({ indexed: 0, total: 0 });
    } finally {
      setIsIndexing(false);
    }
  }

  function handleCreateRetriever() {
    setRetrieverError(null);
    setResults([]);

    if (!vectorStore) {
      setRetrieverError("Vector store is not initialized");
      return;
    }

    try {
      const created = createRetriever(vectorStore, searchConfig);
      setRetriever(created);
    } catch (error) {
      setRetrieverError(error instanceof Error ? error.message : "Failed to create retriever");
      setRetriever(null);
    }
  }

  async function handleSearch() {
    if (!retriever || !query.trim()) {
      return;
    }

    setIsSearching(true);

    try {
      const queryResults = await queryWithRetriever(retriever, query.trim());
      setResults(queryResults);
    } finally {
      setIsSearching(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="step-card step-appear">
        <h2>Download Embedding Model</h2>
        <p>Model: {embeddings.modelName}</p>
        <button disabled={isDownloading || modelReady} onClick={handleDownloadModel}>
          {modelReady ? "Model Ready" : isDownloading ? "Downloading..." : "Download Embedding"}
        </button>
        <div className="progress-wrap" aria-live="polite">
          <progress max={100} value={modelProgressPercent} />
          <span>{modelProgressPercent}%</span>
          {downloadState ? <small>Status: {downloadState.status}</small> : null}
        </div>
        {downloadError ? <p className="error">{downloadError}</p> : null}
      </section>

      {canPickPdf ? (
        <section className="step-card step-appear">
          <h2>Pick PDF</h2>
          <input type="file" accept="application/pdf" onChange={handleFileChange} />
          <p>Selected file: {selectedFile?.name ?? "None"}</p>
          <p>Pages loaded as documents: {documents.length}</p>
        </section>
      ) : null}

      {canChunk ? (
        <section className="step-card step-appear">
          <h2>Chunking Strategy</h2>
          <label>
            Strategy
            <select
              value={chunkingConfig.strategy}
              disabled={isChunking}
              onChange={(event) =>
                setChunkingConfig((current) => ({
                  ...current,
                  strategy: event.target.value as ChunkingConfig["strategy"]
                }))
              }
            >
              <option value="fixed">Fixed size</option>
              <option value="recursive">Recursive character</option>
            </select>
          </label>

          {chunkingConfig.strategy === "fixed" ? (
            <div className="inline-fields">
              <label>
                Chunk size
                <input
                  type="number"
                  min={1}
                  value={chunkingConfig.fixed.chunkSize}
                  onChange={(event) =>
                    setChunkingConfig((current) => ({
                      ...current,
                      fixed: {
                        ...current.fixed,
                        chunkSize: parseNumber(event.target.value, current.fixed.chunkSize)
                      }
                    }))
                  }
                  disabled={isChunking}
                />
              </label>
              <label>
                Chunk overlap
                <input
                  type="number"
                  min={0}
                  value={chunkingConfig.fixed.chunkOverlap}
                  onChange={(event) =>
                    setChunkingConfig((current) => ({
                      ...current,
                      fixed: {
                        ...current.fixed,
                        chunkOverlap: parseNumber(event.target.value, current.fixed.chunkOverlap)
                      }
                    }))
                  }
                  disabled={isChunking}
                />
              </label>
              <label>
                Separator
                <input
                  type="text"
                  value={chunkingConfig.fixed.separator}
                  onChange={(event) =>
                    setChunkingConfig((current) => ({
                      ...current,
                      fixed: {
                        ...current.fixed,
                        separator: event.target.value
                      }
                    }))
                  }
                  disabled={isChunking}
                />
              </label>
            </div>
          ) : (
            <div className="inline-fields">
              <label>
                Chunk size
                <input
                  type="number"
                  min={1}
                  value={chunkingConfig.recursive.chunkSize}
                  onChange={(event) =>
                    setChunkingConfig((current) => ({
                      ...current,
                      recursive: {
                        ...current.recursive,
                        chunkSize: parseNumber(event.target.value, current.recursive.chunkSize)
                      }
                    }))
                  }
                  disabled={isChunking}
                />
              </label>
              <label>
                Chunk overlap
                <input
                  type="number"
                  min={0}
                  value={chunkingConfig.recursive.chunkOverlap}
                  onChange={(event) =>
                    setChunkingConfig((current) => ({
                      ...current,
                      recursive: {
                        ...current.recursive,
                        chunkOverlap: parseNumber(event.target.value, current.recursive.chunkOverlap)
                      }
                    }))
                  }
                  disabled={isChunking}
                />
              </label>
              <label>
                Separators (comma-separated)
                <input
                  type="text"
                  value={chunkingConfig.recursive.separators.join(",")}
                  onChange={(event) =>
                    setChunkingConfig((current) => ({
                      ...current,
                      recursive: {
                        ...current.recursive,
                        separators: event.target.value.split(",")
                      }
                    }))
                  }
                  disabled={isChunking}
                />
              </label>
            </div>
          )}

          <button onClick={handleChunkDocuments} disabled={isChunking}>
            {isChunking ? "Chunking..." : "Create Chunks"}
          </button>
          <p>Chunks created: {chunks.length}</p>
          {chunkError ? <p className="error">{chunkError}</p> : null}
        </section>
      ) : null}

      {canIndex ? (
        <section className="step-card step-appear">
          <h2>Vector Store (Local Browser)</h2>
          <label>
            Vector store
            <select
              value={vectorStoreType}
              onChange={(event) => setVectorStoreType(event.target.value as VectorStoreOption)}
              disabled={isIndexing}
            >
              {AVAILABLE_VECTOR_STORES.map((store) => (
                <option value={store} key={store}>
                  {store}
                </option>
              ))}
            </select>
          </label>

          <button onClick={handleIndexChunks} disabled={isIndexing}>
            {isIndexing ? "Indexing..." : "Index Chunks"}
          </button>
          <div className="progress-wrap" aria-live="polite">
            <progress max={100} value={indexingProgressPercent} />
            <span>{indexingProgressPercent}%</span>
            <small>
              {indexingProgress.indexed}/{indexingProgress.total} chunks indexed
            </small>
          </div>
          {vectorStoreError ? <p className="error">{vectorStoreError}</p> : null}
        </section>
      ) : null}

      {canCreateRetriever ? (
        <section className="step-card step-appear">
          <h2>Retriever Search Type</h2>
          <div className="inline-fields">
            <label>
              Search type
              <select
                value={searchConfig.searchType}
                onChange={(event) =>
                  setSearchConfig((current) => ({
                    ...current,
                    searchType: event.target.value as SearchConfig["searchType"]
                  }))
                }
              >
                <option value="similarity">similarity</option>
                <option value="mmr">mmr</option>
              </select>
            </label>
            <label>
              k
              <input
                type="number"
                min={1}
                value={searchConfig.k}
                onChange={(event) =>
                  setSearchConfig((current) => ({
                    ...current,
                    k: parseNumber(event.target.value, current.k)
                  }))
                }
              />
            </label>
            {searchConfig.searchType === "mmr" ? (
              <>
                <label>
                  fetchK
                  <input
                    type="number"
                    min={1}
                    value={searchConfig.fetchK}
                    onChange={(event) =>
                      setSearchConfig((current) => ({
                        ...current,
                        fetchK: parseNumber(event.target.value, current.fetchK)
                      }))
                    }
                  />
                </label>
                <label>
                  lambda
                  <input
                    type="number"
                    min={0}
                    max={1}
                    step={0.1}
                    value={searchConfig.lambda}
                    onChange={(event) =>
                      setSearchConfig((current) => ({
                        ...current,
                        lambda: parseNumber(event.target.value, current.lambda)
                      }))
                    }
                  />
                </label>
              </>
            ) : null}
          </div>
          <button onClick={handleCreateRetriever}>Set Retriever</button>
          {retriever ? <p>Retriever ready.</p> : <p>Retriever not set.</p>}
          {retrieverError ? <p className="error">{retrieverError}</p> : null}
        </section>
      ) : null}

      {canSearch ? (
        <section className="step-card step-appear">
          <h2>Query + Results</h2>
          <div className="inline-fields">
            <input
              type="text"
              placeholder="Type your query"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              disabled={isSearching}
            />
            <button onClick={handleSearch} disabled={isSearching || !query.trim()}>
              {isSearching ? "Searching..." : "Search"}
            </button>
          </div>

          <p>Results: {results.length}</p>
          <ol className="results-list">
            {results.map((result, index) => (
              <li key={`${index}-${result.document.metadata?.chunkIndex ?? "na"}`}>
                <button
                  type="button"
                  className="result-item-button"
                  onClick={() => {
                    const pageNumber = getResultPageNumber(result);
                    if (pageNumber) {
                      jumpViewerToPage(pageNumber);
                    }
                  }}
                >
                  Go to this result page in viewer
                </button>
                <p>
                  <strong>Page:</strong> {String(result.document.metadata?.loc?.pageNumber ?? "unknown")}
                </p>
                <p>
                  <strong>Source:</strong> {String(result.document.metadata?.source ?? "unknown")}
                </p>
                <pre>{result.document.pageContent}</pre>
              </li>
            ))}
          </ol>

          <div className="pdf-viewer-wrap">
            <p>Viewer page: {viewerPage}</p>
            {pdfObjectUrl ? (
              <iframe key={viewerKey} className="pdf-viewer" src={viewerSrc} title="PDF Viewer" />
            ) : (
              <p>No PDF selected.</p>
            )}
          </div>
        </section>
      ) : null}
    </main>
  );
}
