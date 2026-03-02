/// <reference types="vite/client" />

declare module "pdfjs-dist/build/pdf.mjs" {
  export const getDocument: (...args: unknown[]) => unknown;
  export const version: string;
  export const GlobalWorkerOptions: {
    workerSrc: string;
  };
}

declare module "pdfjs-dist/build/pdf.worker.min.mjs?url" {
  const src: string;
  export default src;
}
