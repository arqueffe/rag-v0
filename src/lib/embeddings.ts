import { Embeddings } from "@langchain/core/embeddings";
import { env, pipeline, type FeatureExtractionPipeline } from "@xenova/transformers";
import type { ModelDownloadProgress } from "../types";

export interface BrowserEmbeddingConfig {
  model: string;
}

const DEFAULT_MODEL = "Xenova/all-MiniLM-L6-v2";

export class BrowserTransformersEmbeddings extends Embeddings {
  private extractor: FeatureExtractionPipeline | null = null;
  private readonly model: string;

  constructor(config: Partial<BrowserEmbeddingConfig> = {}) {
    super({});
    this.model = config.model ?? DEFAULT_MODEL;
    env.allowLocalModels = false;
  }

  get modelName(): string {
    return this.model;
  }

  async downloadModel(onProgress?: (progress: ModelDownloadProgress) => void): Promise<void> {
    if (this.extractor) {
      onProgress?.({ status: "ready", loaded: 1, total: 1, progress: 1 });
      return;
    }

    this.extractor = await pipeline("feature-extraction", this.model, {
      progress_callback: (event: unknown) => {
        if (!onProgress || typeof event !== "object" || event === null) {
          return;
        }

        const e = event as {
          status?: string;
          loaded?: number;
          total?: number;
          progress?: number;
        };

        const loaded = Number.isFinite(e.loaded) ? Number(e.loaded) : 0;
        const total = Number.isFinite(e.total) ? Number(e.total) : 0;
        const inferredProgress = total > 0 ? loaded / total : 0;
        const progress = Number.isFinite(e.progress)
          ? Math.max(0, Math.min(1, Number(e.progress) / (Number(e.progress) > 1 ? 100 : 1)))
          : inferredProgress;

        onProgress({
          status: e.status ?? "downloading",
          loaded,
          total,
          progress
        });
      }
    });

    onProgress?.({ status: "ready", loaded: 1, total: 1, progress: 1 });
  }

  private async ensureExtractor(): Promise<FeatureExtractionPipeline> {
    if (!this.extractor) {
      await this.downloadModel();
    }

    return this.extractor as FeatureExtractionPipeline;
  }

  async embedDocuments(texts: string[]): Promise<number[][]> {
    const extractor = await this.ensureExtractor();
    return Promise.all(texts.map((text) => this.embedQueryWithExtractor(extractor, text)));
  }

  async embedQuery(text: string): Promise<number[]> {
    const extractor = await this.ensureExtractor();
    return this.embedQueryWithExtractor(extractor, text);
  }

  private async embedQueryWithExtractor(
    extractor: FeatureExtractionPipeline,
    text: string
  ): Promise<number[]> {
    const result = await extractor(text, { pooling: "mean", normalize: true });
    const typed = result?.data as Float32Array | number[] | undefined;

    if (!typed) {
      return [];
    }

    return Array.from(typed as ArrayLike<number>);
  }
}
