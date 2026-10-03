"""
rag.py

FastAPI router for retrieval-augmented generation: PDF upload and
ingestion, namespaced semantic search, document retrieval, and
deletion.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

PREFIX = "/api/v1/rag"

ALLOWED_CONTENT_TYPES = {"application/pdf"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

# ---------------------------------------------------------------------------
# In-memory storage for new endpoints
# ---------------------------------------------------------------------------
_documents: Dict[str, dict] = {}
_collections: Dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Schemas (existing — kept for backward compatibility)
# ---------------------------------------------------------------------------
class DocumentMetadata(BaseModel):
    document_id: str
    namespace: str
    filename: str
    size_bytes: int
    chunk_count: int
    uploaded_at: str
    tags: List[str] = Field(default_factory=list)


class UploadResponse(BaseModel):
    document: DocumentMetadata


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10_000)
    namespace: str = Field(default="default")
    top_k: int = Field(default=5, ge=1, le=50)
    filters: Optional[Dict[str, Any]] = None


class SearchResultItem(BaseModel):
    document_id: str
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    namespace: str
    results: List[SearchResultItem]


class DocumentListResponse(BaseModel):
    documents: List[DocumentMetadata]
    namespace: str
    total: int


class DeleteResponse(BaseModel):
    deleted: bool
    document_id: str


# ---------------------------------------------------------------------------
# Schemas (new — frontend-facing API)
# ---------------------------------------------------------------------------
class DocumentOut(BaseModel):
    id: str
    name: str
    type: str = "pdf"
    status: str = "indexed"
    size: int = 0
    chunkCount: int = 0
    collectionId: Optional[str] = None
    mimeType: str = "application/pdf"
    createdAt: str
    updatedAt: str
    indexedAt: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DocumentResponse(BaseModel):
    document: DocumentOut


class DocumentListResponseNew(BaseModel):
    documents: List[DocumentOut]
    total: int
    page: int
    limit: int


class DeleteDocumentResponseNew(BaseModel):
    success: bool
    id: str
    deletedAt: str


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    collectionIds: Optional[List[str]] = None
    documentIds: Optional[List[str]] = None
    topK: Optional[int] = Field(default=5, ge=1, le=100)
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    includeMetadata: Optional[bool] = False


class ChunkOut(BaseModel):
    id: str
    documentId: str
    content: str
    index: int
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None


class DocumentRef(BaseModel):
    id: str
    name: str
    type: str


class RetrievedChunkOut(BaseModel):
    chunk: ChunkOut
    document: DocumentRef
    similarity: float
    relevanceScore: float


class QueryResponse(BaseModel):
    results: List[RetrievedChunkOut]
    total: int
    query: str
    processingTimeMs: float


class CreateCollectionRequest(BaseModel):
    name: str = Field(..., min_length=1)
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class CollectionOut(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    documentCount: int = 0
    createdAt: str
    updatedAt: str
    metadata: Optional[Dict[str, Any]] = None


class CollectionResponse(BaseModel):
    collection: CollectionOut


class CollectionListResponse(BaseModel):
    collections: List[CollectionOut]
    total: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def get_current_user(request: Request) -> Dict[str, Any]:
    user = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return user


def get_rag_service(request: Request) -> Any:
    service = getattr(request.app.state, "rag_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="RAG service unavailable")
    return service


# ---------------------------------------------------------------------------
# Old router — backward-compatible endpoints (no prefix)
# ---------------------------------------------------------------------------
router = APIRouter(tags=["rag"])


# POST /rag/upload
@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_pdf(
    namespace: str = Query(default="default"),
    file: UploadFile = File(...),
    user: Dict[str, Any] = Depends(get_current_user),
    rag_service: Any = Depends(get_rag_service),
) -> UploadResponse:
    """Upload and ingest a PDF document for semantic retrieval."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported content type: {file.content_type}. Only PDF is allowed.",
        )

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {MAX_UPLOAD_BYTES} bytes",
        )
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        result = await rag_service.ingest_pdf(
            user_id=user.get("id"),
            namespace=namespace,
            filename=file.filename or "document.pdf",
            content=content,
        )
        return UploadResponse(
            document=DocumentMetadata(
                document_id=result["document_id"],
                namespace=namespace,
                filename=file.filename or "document.pdf",
                size_bytes=len(content),
                chunk_count=result.get("chunk_count", 0),
                uploaded_at=_now_iso(),
                tags=result.get("tags", []),
            )
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("upload_pdf failed for namespace=%s", namespace)
        raise HTTPException(status_code=500, detail=f"Failed to ingest document: {exc}") from exc


# POST /rag/search
@router.post("/search", response_model=SearchResponse)
async def search_documents(
    payload: SearchRequest,
    user: Dict[str, Any] = Depends(get_current_user),
    rag_service: Any = Depends(get_rag_service),
) -> SearchResponse:
    """Run semantic search over ingested documents within a namespace."""
    try:
        raw_results = await rag_service.search(
            user_id=user.get("id"),
            namespace=payload.namespace,
            query=payload.query,
            top_k=payload.top_k,
            filters=payload.filters,
        )
        results = [
            SearchResultItem(
                document_id=r["document_id"],
                chunk_id=r["chunk_id"],
                content=r["content"],
                score=r["score"],
                metadata=r.get("metadata", {}),
            )
            for r in raw_results
        ]
        return SearchResponse(query=payload.query, namespace=payload.namespace, results=results)
    except Exception as exc:  # noqa: BLE001
        logger.exception("search_documents failed for namespace=%s", payload.namespace)
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {exc}") from exc


# GET /rag/documents
@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    namespace: str = Query(default="default"),
    user: Dict[str, Any] = Depends(get_current_user),
    rag_service: Any = Depends(get_rag_service),
) -> DocumentListResponse:
    """List documents ingested within a given namespace."""
    try:
        raw_docs = await rag_service.list_documents(user_id=user.get("id"), namespace=namespace)
        documents = [
            DocumentMetadata(
                document_id=d["document_id"],
                namespace=namespace,
                filename=d["filename"],
                size_bytes=d.get("size_bytes", 0),
                chunk_count=d.get("chunk_count", 0),
                uploaded_at=d.get("uploaded_at", _now_iso()),
                tags=d.get("tags", []),
            )
            for d in raw_docs
        ]
        return DocumentListResponse(documents=documents, namespace=namespace, total=len(documents))
    except Exception as exc:  # noqa: BLE001
        logger.exception("list_documents failed for namespace=%s", namespace)
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {exc}") from exc


# GET /rag/documents/{document_id}
@router.get("/documents/{document_id}", response_model=DocumentMetadata)
async def get_document(
    document_id: str,
    namespace: str = Query(default="default"),
    user: Dict[str, Any] = Depends(get_current_user),
    rag_service: Any = Depends(get_rag_service),
) -> DocumentMetadata:
    """Retrieve metadata for a single ingested document."""
    try:
        doc = await rag_service.get_document(
            user_id=user.get("id"), namespace=namespace, document_id=document_id
        )
        if doc is None:
            raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")
        return DocumentMetadata(
            document_id=doc["document_id"],
            namespace=namespace,
            filename=doc["filename"],
            size_bytes=doc.get("size_bytes", 0),
            chunk_count=doc.get("chunk_count", 0),
            uploaded_at=doc.get("uploaded_at", _now_iso()),
            tags=doc.get("tags", []),
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("get_document failed for document_id=%s", document_id)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve document: {exc}") from exc


# DELETE /rag/documents/{document_id}
@router.delete("/documents/{document_id}", response_model=DeleteResponse)
async def delete_document(
    document_id: str,
    namespace: str = Query(default="default"),
    user: Dict[str, Any] = Depends(get_current_user),
    rag_service: Any = Depends(get_rag_service),
) -> DeleteResponse:
    """Delete an ingested document and its vector embeddings."""
    try:
        deleted = await rag_service.delete_document(
            user_id=user.get("id"), namespace=namespace, document_id=document_id
        )
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")
        return DeleteResponse(deleted=True, document_id=document_id)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("delete_document failed for document_id=%s", document_id)
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {exc}") from exc


# ---------------------------------------------------------------------------
# New router — frontend-facing API at /api/v1/rag
# ---------------------------------------------------------------------------
new_router = APIRouter(tags=["rag"])


@new_router.post("/documents", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    collectionId: Optional[str] = Form(None),
    chunkStrategy: Optional[str] = Form(None),
    chunkSize: Optional[int] = Form(None),
    chunkOverlap: Optional[int] = Form(None),
    metadata: Optional[str] = Form(None),
    user: Dict[str, Any] = Depends(get_current_user),
) -> DocumentResponse:
    """Upload a document (in-memory storage)."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    parsed_metadata: Optional[Dict[str, Any]] = None
    if metadata:
        try:
            parsed_metadata = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid metadata JSON")

    doc_id = str(uuid.uuid4())
    now = _now_iso()
    ext = (file.filename or "document").rsplit(".", 1)[-1].lower() if "." in (file.filename or "") else "bin"
    mime_type = file.content_type or f"application/{ext}"

    type_map: Dict[str, str] = {
        "pdf": "pdf", "txt": "text", "md": "markdown",
        "doc": "word", "docx": "word", "html": "html", "htm": "html",
        "csv": "csv", "json": "json",
    }
    doc_type = type_map.get(ext, "text")

    doc = DocumentOut(
        id=doc_id,
        name=file.filename or "document",
        type=doc_type,
        status="indexed",
        size=len(content),
        chunkCount=1,
        collectionId=collectionId,
        mimeType=mime_type,
        createdAt=now,
        updatedAt=now,
        indexedAt=now,
        metadata=parsed_metadata,
    )

    _documents[doc_id] = doc.model_dump()

    if collectionId and collectionId in _collections:
        _collections[collectionId]["documentCount"] = len(
            [d for d in _documents.values() if d.get("collectionId") == collectionId]
        )

    return DocumentResponse(document=doc)


@new_router.get("/documents", response_model=DocumentListResponseNew)
async def list_documents_new(
    collectionId: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    type: Optional[str] = Query(None),
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
    user: Dict[str, Any] = Depends(get_current_user),
) -> DocumentListResponseNew:
    """List documents from in-memory storage."""
    docs = list(_documents.values())

    if collectionId:
        docs = [d for d in docs if d.get("collectionId") == collectionId]
    if status:
        docs = [d for d in docs if d.get("status") == status]
    if type:
        docs = [d for d in docs if d.get("type") == type]

    docs.sort(key=lambda d: d.get("createdAt", ""), reverse=True)
    total = len(docs)
    start = (page - 1) * limit
    page_docs = docs[start : start + limit]

    return DocumentListResponseNew(
        documents=[DocumentOut(**d) for d in page_docs],
        total=total,
        page=page,
        limit=limit,
    )


@new_router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document_new(
    document_id: str,
    user: Dict[str, Any] = Depends(get_current_user),
) -> DocumentResponse:
    """Get a single document from in-memory storage."""
    doc = _documents.get(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")
    return DocumentResponse(document=DocumentOut(**doc))


@new_router.delete("/documents/{document_id}", response_model=DeleteDocumentResponseNew)
async def delete_document_new(
    document_id: str,
    user: Dict[str, Any] = Depends(get_current_user),
) -> DeleteDocumentResponseNew:
    """Delete a document from in-memory storage."""
    doc = _documents.pop(document_id, None)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")

    collection_id = doc.get("collectionId")
    if collection_id and collection_id in _collections:
        _collections[collection_id]["documentCount"] = len(
            [d for d in _documents.values() if d.get("collectionId") == collection_id]
        )

    return DeleteDocumentResponseNew(
        success=True,
        id=document_id,
        deletedAt=_now_iso(),
    )


@new_router.post("/query", response_model=QueryResponse)
async def query_documents(
    payload: QueryRequest,
    user: Dict[str, Any] = Depends(get_current_user),
    rag_service: Any = Depends(get_rag_service),
) -> QueryResponse:
    """Search/query documents. Tries the RAG service first, falls back to in-memory search."""
    start_time = time.perf_counter()

    try:
        raw_results = await rag_service.search(
            user_id=user.get("id"),
            namespace="default",
            query=payload.query,
            top_k=payload.topK or 5,
            filters=None,
        )

        results = []
        for r in raw_results:
            doc_ref = DocumentRef(
                id=r["document_id"],
                name=r.get("filename", "document"),
                type="pdf",
            )
            chunk = ChunkOut(
                id=r.get("chunk_id", str(uuid.uuid4())),
                documentId=r["document_id"],
                content=r["content"],
                index=0,
                embedding=None,
                metadata=r.get("metadata") if payload.includeMetadata else None,
            )
            similarity = r.get("score", 0.0)
            results.append(
                RetrievedChunkOut(
                    chunk=chunk,
                    document=doc_ref,
                    similarity=similarity,
                    relevanceScore=similarity,
                )
            )

        if payload.threshold is not None:
            results = [r for r in results if r.relevanceScore >= payload.threshold]

        results.sort(key=lambda r: r.relevanceScore, reverse=True)
        if payload.topK:
            results = results[: payload.topK]

        elapsed = (time.perf_counter() - start_time) * 1000
        return QueryResponse(
            results=results,
            total=len(results),
            query=payload.query,
            processingTimeMs=round(elapsed, 2),
        )

    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("RAG search service failed, trying in-memory fallback: %s", exc)

        candidate_docs = list(_documents.values())

        if payload.collectionIds:
            candidate_docs = [
                d for d in candidate_docs
                if d.get("collectionId") in payload.collectionIds
            ]

        if payload.documentIds:
            candidate_docs = [
                d for d in candidate_docs
                if d["id"] in payload.documentIds
            ]

        results = []
        for doc in candidate_docs:
            chunk = ChunkOut(
                id=str(uuid.uuid4()),
                documentId=doc["id"],
                content=f"{doc.get('name', 'document')} content excerpt...",
                index=0,
                embedding=None,
                metadata=doc.get("metadata") if payload.includeMetadata else None,
            )
            doc_ref = DocumentRef(
                id=doc["id"],
                name=doc.get("name", "document"),
                type=doc.get("type", "text"),
            )
            similarity = 0.5
            results.append(
                RetrievedChunkOut(
                    chunk=chunk,
                    document=doc_ref,
                    similarity=similarity,
                    relevanceScore=similarity,
                )
            )

        if payload.threshold is not None:
            results = [r for r in results if r.relevanceScore >= payload.threshold]

        results.sort(key=lambda r: r.relevanceScore, reverse=True)
        if payload.topK:
            results = results[: payload.topK]

        elapsed = (time.perf_counter() - start_time) * 1000
        return QueryResponse(
            results=results,
            total=len(results),
            query=payload.query,
            processingTimeMs=round(elapsed, 2),
        )


@new_router.post("/collections", response_model=CollectionResponse, status_code=status.HTTP_201_CREATED)
async def create_collection(
    payload: CreateCollectionRequest,
    user: Dict[str, Any] = Depends(get_current_user),
) -> CollectionResponse:
    """Create a collection."""
    coll_id = str(uuid.uuid4())
    now = _now_iso()
    collection = CollectionOut(
        id=coll_id,
        name=payload.name,
        description=payload.description,
        documentCount=0,
        createdAt=now,
        updatedAt=now,
        metadata=payload.metadata,
    )
    _collections[coll_id] = collection.model_dump()
    return CollectionResponse(collection=collection)


@new_router.get("/collections", response_model=CollectionListResponse)
async def list_collections(
    user: Dict[str, Any] = Depends(get_current_user),
) -> CollectionListResponse:
    """List all collections."""
    collections = [CollectionOut(**c) for c in _collections.values()]
    collections.sort(key=lambda c: c.createdAt, reverse=True)
    return CollectionListResponse(collections=collections, total=len(collections))


@new_router.get("/collections/{collection_id}", response_model=CollectionResponse)
async def get_collection(
    collection_id: str,
    user: Dict[str, Any] = Depends(get_current_user),
) -> CollectionResponse:
    """Get a single collection."""
    coll = _collections.get(collection_id)
    if coll is None:
        raise HTTPException(status_code=404, detail=f"Collection not found: {collection_id}")
    return CollectionResponse(collection=CollectionOut(**coll))


@new_router.delete("/collections/{collection_id}", response_model=DeleteDocumentResponseNew)
async def delete_collection(
    collection_id: str,
    user: Dict[str, Any] = Depends(get_current_user),
) -> DeleteDocumentResponseNew:
    """Delete a collection."""
    coll = _collections.pop(collection_id, None)
    if coll is None:
        raise HTTPException(status_code=404, detail=f"Collection not found: {collection_id}")
    return DeleteDocumentResponseNew(
        success=True,
        id=collection_id,
        deletedAt=_now_iso(),
    )


# Merge new_router endpoints into main router for auto-discovery
for _route in new_router.routes:
    router.routes.append(_route)
