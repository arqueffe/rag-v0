import { CharacterTextSplitter, RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import type { Document } from "@langchain/core/documents";
import type { ChunkingConfig } from "../types";

export async function chunkDocuments(documents: Document[], config: ChunkingConfig): Promise<Document[]> {
  if (config.strategy === "fixed") {
    const splitter = new CharacterTextSplitter({
      separator: config.fixed.separator,
      chunkSize: config.fixed.chunkSize,
      chunkOverlap: config.fixed.chunkOverlap
    });
    return splitter.splitDocuments(documents);
  }

  const splitter = new RecursiveCharacterTextSplitter({
    separators: config.recursive.separators,
    chunkSize: config.recursive.chunkSize,
    chunkOverlap: config.recursive.chunkOverlap
  });

  return splitter.splitDocuments(documents);
}
