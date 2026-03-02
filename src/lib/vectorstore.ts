import type { Document } from "@langchain/core/documents";
import type { Embeddings } from "@langchain/core/embeddings";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import type { LocalVectorStore, VectorStoreOption } from "../types";

export const AVAILABLE_VECTOR_STORES: VectorStoreOption[] = ["memory"];

export interface IndexingProgress {
  indexed: number;
  total: number;
}

interface CreateVectorStoreOptions {
  batchSize?: number;
  onProgress?: (progress: IndexingProgress) => void;
}

function nextTick(): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, 0);
  });
}

export async function createVectorStore(
  type: VectorStoreOption,
  documents: Document[],
  embeddings: Embeddings,
  options: CreateVectorStoreOptions = {}
): Promise<LocalVectorStore> {
  const total = documents.length;
  const batchSize = Math.max(1, options.batchSize ?? 24);

  if (type === "memory") {
    const store = new MemoryVectorStore(embeddings);
    options.onProgress?.({ indexed: 0, total });

    for (let start = 0; start < total; start += batchSize) {
      const batch = documents.slice(start, start + batchSize);
      await store.addDocuments(batch);

      const indexed = Math.min(start + batch.length, total);
      options.onProgress?.({ indexed, total });

      await nextTick();
    }

    return store;
  }

  throw new Error(`Unsupported vector store type: ${type}`);
}
