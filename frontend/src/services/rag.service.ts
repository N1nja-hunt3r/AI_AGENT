import api from "@/services/api";
import type { AxiosResponse } from "axios";

// ── Enums ─────────────────────────────────────────────────────────────────────
export const DocumentStatus = {
  Pending: "pending",
  Processing: "processing",
  Indexed: "indexed",
  Failed: "failed",
} as const;

export type DocumentStatus = (typeof DocumentStatus)[keyof typeof DocumentStatus];

export const DocumentType = {
  PDF: "pdf",
  Text: "text",
  Markdown: "markdown",
  Word: "word",
  HTML: "html",
  CSV: "csv",
  JSON: "json",
} as const;

export type DocumentType = (typeof DocumentType)[keyof typeof DocumentType];

export const ChunkStrategy = {
  Fixed: "fixed",
  Sentence: "sentence",
  Paragraph: "paragraph",
  Semantic: "semantic",
} as const;

export type ChunkStrategy = (typeof ChunkStrategy)[keyof typeof ChunkStrategy];

// ── Types ─────────────────────────────────────────────────────────────────────
export interface Document {
  id: string;
  name: string;
  type: DocumentType;
  status: DocumentStatus;
  size: number;
  chunkCount: number;
  collectionId?: string;
  mimeType: string;
  createdAt: string;
  updatedAt: string;
  indexedAt?: string;
  metadata?: Record<string, unknown>;
}

export interface DocumentChunk {
  id: string;
  documentId: string;
  content: string;
  index: number;
  embedding?: number[];
  metadata?: Record<string, unknown>;
}

export interface RetrievedChunk {
  chunk: DocumentChunk;
  document: Pick<Document, "id" | "name" | "type">;
  similarity: number;
  relevanceScore: number;
}

export interface Collection {
  id: string;
  name: string;
  description?: string;
  documentCount: number;
  createdAt: string;
  updatedAt: string;
  metadata?: Record<string, unknown>;
}

// ── Request payloads ──────────────────────────────────────────────────────────
export interface UploadDocumentPayload {
  file: File;
  collectionId?: string;
  chunkStrategy?: ChunkStrategy;
  chunkSize?: number;
  chunkOverlap?: number;
  metadata?: Record<string, unknown>;
}

export interface QueryPayload {
  query: string;
  collectionIds?: string[];
  documentIds?: string[];
  topK?: number;
  threshold?: number;
  includeMetadata?: boolean;
}

export interface CreateCollectionPayload {
  name: string;
  description?: string;
  metadata?: Record<string, unknown>;
}

export interface ListDocumentsParams {
  collectionId?: string;
  status?: DocumentStatus;
  type?: DocumentType;
  page?: number;
  limit?: number;
}

// ── Response shapes ───────────────────────────────────────────────────────────
export interface DocumentResponse {
  document: Document;
}

export interface DocumentListResponse {
  documents: Document[];
  total: number;
  page: number;
  limit: number;
}

export interface QueryResponse {
  results: RetrievedChunk[];
  total: number;
  query: string;
  processingTimeMs: number;
}

export interface CollectionResponse {
  collection: Collection;
}

export interface CollectionListResponse {
  collections: Collection[];
  total: number;
}

export interface DeleteDocumentResponse {
  success: boolean;
  id: string;
  deletedAt: string;
}

// ── Endpoints ─────────────────────────────────────────────────────────────────
const ENDPOINTS = {
  DOCUMENTS: "/api/v1/rag/documents",
  DOCUMENT_BY_ID: (id: string) => `/api/v1/rag/documents/${id}`,
  QUERY: "/api/v1/rag/query",
  COLLECTIONS: "/api/v1/rag/collections",
  COLLECTION_BY_ID: (id: string) => `/api/v1/rag/collections/${id}`,
} as const;

// ── Service methods ───────────────────────────────────────────────────────────
const uploadDocument = async (
  payload: UploadDocumentPayload
): Promise<DocumentResponse> => {
  const formData = new FormData();
  formData.append("file", payload.file);

  if (payload.collectionId)
    formData.append("collectionId", payload.collectionId);
  if (payload.chunkStrategy)
    formData.append("chunkStrategy", payload.chunkStrategy);
  if (payload.chunkSize !== undefined)
    formData.append("chunkSize", String(payload.chunkSize));
  if (payload.chunkOverlap !== undefined)
    formData.append("chunkOverlap", String(payload.chunkOverlap));
  if (payload.metadata)
    formData.append("metadata", JSON.stringify(payload.metadata));

  const response: AxiosResponse<DocumentResponse> =
    await api.post<DocumentResponse>(ENDPOINTS.DOCUMENTS, formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
  return response.data;
};

const listDocuments = async (
  params?: ListDocumentsParams
): Promise<DocumentListResponse> => {
  const response: AxiosResponse<DocumentListResponse> =
    await api.get<DocumentListResponse>(ENDPOINTS.DOCUMENTS, { params });
  return response.data;
};

const getDocument = async (id: string): Promise<DocumentResponse> => {
  const response: AxiosResponse<DocumentResponse> =
    await api.get<DocumentResponse>(ENDPOINTS.DOCUMENT_BY_ID(id));
  return response.data;
};

const deleteDocument = async (id: string): Promise<DeleteDocumentResponse> => {
  const response: AxiosResponse<DeleteDocumentResponse> =
    await api.delete<DeleteDocumentResponse>(ENDPOINTS.DOCUMENT_BY_ID(id));
  return response.data;
};

const queryDocuments = async (
  payload: QueryPayload
): Promise<QueryResponse> => {
  const response: AxiosResponse<QueryResponse> =
    await api.post<QueryResponse>(ENDPOINTS.QUERY, payload);
  return response.data;
};

const createCollection = async (
  payload: CreateCollectionPayload
): Promise<CollectionResponse> => {
  const response: AxiosResponse<CollectionResponse> =
    await api.post<CollectionResponse>(ENDPOINTS.COLLECTIONS, payload);
  return response.data;
};

const listCollections = async (): Promise<CollectionListResponse> => {
  const response: AxiosResponse<CollectionListResponse> =
    await api.get<CollectionListResponse>(ENDPOINTS.COLLECTIONS);
  return response.data;
};

const deleteCollection = async (
  id: string
): Promise<DeleteDocumentResponse> => {
  const response: AxiosResponse<DeleteDocumentResponse> =
    await api.delete<DeleteDocumentResponse>(ENDPOINTS.COLLECTION_BY_ID(id));
  return response.data;
};

export const RagService = {
  uploadDocument,
  listDocuments,
  getDocument,
  deleteDocument,
  queryDocuments,
  createCollection,
  listCollections,
  deleteCollection,
} as const;

export default RagService;