import type { Document } from "@langchain/core/documents";
import type { MemoryVectorStore } from "langchain/vectorstores/memory";

export type ChunkingStrategy = "fixed" | "recursive";
export type VectorStoreOption = "memory";
export type RetrieverSearchType = "similarity" | "mmr";

export interface FixedChunkingConfig {
  chunkSize: number;
  chunkOverlap: number;
  separator: string;
}

export interface RecursiveChunkingConfig {
  chunkSize: number;
  chunkOverlap: number;
  separators: string[];
}

export interface ChunkingConfig {
  strategy: ChunkingStrategy;
  fixed: FixedChunkingConfig;
  recursive: RecursiveChunkingConfig;
}

export interface SearchConfig {
  searchType: RetrieverSearchType;
  k: number;
  fetchK: number;
  lambda: number;
}

export interface ModelDownloadProgress {
  status: string;
  loaded: number;
  total: number;
  progress: number;
}

export interface QueryResult {
  document: Document;
  score?: number;
}

export type LocalVectorStore = MemoryVectorStore;
