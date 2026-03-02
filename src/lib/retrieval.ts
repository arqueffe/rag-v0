import type { BaseRetrieverInterface } from "@langchain/core/retrievers";
import type { QueryResult, SearchConfig } from "../types";
import type { LocalVectorStore } from "../types";

export function createRetriever(vectorStore: LocalVectorStore, config: SearchConfig): BaseRetrieverInterface {
  if (config.searchType === "mmr") {
    return vectorStore.asRetriever({
      searchType: "mmr",
      k: config.k,
      searchKwargs: {
        fetchK: config.fetchK,
        lambda: config.lambda
      }
    });
  }

  return vectorStore.asRetriever({
    searchType: "similarity",
    k: config.k
  });
}

export async function queryWithRetriever(
  retriever: BaseRetrieverInterface,
  query: string
): Promise<QueryResult[]> {
  const documents = await retriever.invoke(query);
  return documents.map((document) => ({ document }));
}
