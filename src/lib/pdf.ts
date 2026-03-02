import { WebPDFLoader } from "@langchain/community/document_loaders/web/pdf";
import type { Document } from "@langchain/core/documents";
import * as pdfjs from "pdfjs-dist/build/pdf.mjs";
import pdfWorkerSrc from "pdfjs-dist/build/pdf.worker.min.mjs?url";

pdfjs.GlobalWorkerOptions.workerSrc = pdfWorkerSrc;

export async function loadPdfAsDocuments(file: File): Promise<Document[]> {
  const loader = new WebPDFLoader(file, {
    pdfjs: async () => pdfjs,
    splitPages: true
  });

  const documents = await loader.load();

  return documents.map((doc, index) => ({
    ...doc,
    metadata: {
      ...doc.metadata,
      source: file.name,
      chunkIndex: index
    }
  }));
}
