import hashlib
import json
from typing import List
from langchain.embeddings.base import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from lfx.base.vectorstores.model import LCVectorStoreComponent, check_cached_vector_store
from lfx.helpers.data import docs_to_data
from lfx.io import (
    BoolInput,
    DropdownInput,
    HandleInput,
    IntInput,
    SecretStrInput,
    StrInput,
)
from lfx.schema.data import Data


def generate_content_hash(data: dict) -> str:
    """Generate deterministic hash from all content except embeddings.
    
    This creates a content-addressable ID - same content always produces
    the same hash, enabling automatic deduplication on upsert.
    
    Args:
        data: Dictionary containing document data
        
    Returns:
        32-character hex hash suitable for use as Qdrant point ID
    """
    # Copy data and remove embedding-related fields (they're representations, not identity)
    content = {k: v for k, v in data.items() 
               if k not in ("embeddings", "embedding", "vector", "vectors", "model")}
    
    # Sort keys for deterministic serialization
    try:
        serialized = json.dumps(content, sort_keys=True, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        # Fallback: convert to string representation
        serialized = str(sorted(content.items()))
    
    # SHA256 hash, truncated to 32 chars
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()[:32]


class PrecomputedEmbeddingsFromData(Embeddings):
    """LangChain Embeddings wrapper that extracts pre-computed embeddings from Data objects."""
    
    def __init__(self, data_items: list):
        self._vectors: List[List[float]] = []
        self._texts: List[str] = []
        self._text_to_vector: dict = {}
        
        for item in data_items or []:
            if isinstance(item, Data):
                data = item.data
            elif isinstance(item, dict):
                data = item
            else:
                continue
            
            # Extract text
            text = data.get("text", "")
            if isinstance(text, list):
                # Handle list of dicts (e.g., news items)
                text = "\n".join(
                    f"- {entry.get('title', '')}: {entry.get('summary', '')}"
                    if isinstance(entry, dict) else str(entry)
                    for entry in text
                )
            elif isinstance(text, dict):
                text = str(text)
            
            # Extract embeddings
            embeddings = data.get("embeddings") or data.get("vector") or data.get("vectors")
            if embeddings and isinstance(embeddings, list):
                # Handle single embedding or list of embeddings
                if embeddings and isinstance(embeddings[0], (int, float)):
                    # Single embedding vector
                    self._vectors.append(embeddings)
                    self._texts.append(str(text))
                    if text:
                        self._text_to_vector[str(text)] = embeddings
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results = []
        for i, text in enumerate(texts):
            text_str = str(text)
            if text_str in self._text_to_vector:
                results.append(self._text_to_vector[text_str])
            elif i < len(self._vectors):
                results.append(self._vectors[i])
            elif self._vectors:
                results.append(self._vectors[0])
            else:
                results.append([])
        return results
    
    def embed_query(self, text: str) -> List[float]:
        text_str = str(text)
        if text_str in self._text_to_vector:
            return self._text_to_vector[text_str]
        return self._vectors[0] if self._vectors else []
    
    @property
    def vectors(self) -> List[List[float]]:
        return self._vectors
    
    def has_embeddings(self) -> bool:
        return len(self._vectors) > 0


class QdrantVectorStoreComponent(LCVectorStoreComponent):
    display_name = "Qdrant"
    description = "Qdrant Vector Store with search capabilities"
    icon = "Qdrant"

    inputs = [
        StrInput(name="collection_name", display_name="Collection Name", required=True),
        StrInput(name="host", display_name="Host", value="localhost", advanced=True),
        IntInput(name="port", display_name="Port", value=6333, advanced=True),
        IntInput(name="grpc_port", display_name="gRPC Port", value=6334, advanced=True),
        BoolInput(
            name="https",
            display_name="Use HTTPS",
            value=False,
            info="Use HTTPS for connection. Disable for local Qdrant instances.",
            advanced=True,
        ),
        SecretStrInput(name="api_key", display_name="Qdrant API Key", advanced=True),
        StrInput(name="prefix", display_name="Prefix", advanced=True),
        IntInput(name="timeout", display_name="Timeout", advanced=True),
        StrInput(name="path", display_name="Path", advanced=True),
        StrInput(name="url", display_name="URL", advanced=True),
        DropdownInput(
            name="distance_func",
            display_name="Distance Function",
            options=["Cosine", "Euclidean", "Dot Product"],
            value="Cosine",
            advanced=True,
        ),
        StrInput(name="content_payload_key", display_name="Content Payload Key", value="page_content", advanced=True),
        StrInput(name="metadata_payload_key", display_name="Metadata Payload Key", value="metadata", advanced=True),
        *LCVectorStoreComponent.inputs,
        HandleInput(name="embedding", display_name="Embedding", input_types=["Embeddings"]),
        IntInput(
            name="number_of_results",
            display_name="Number of Results",
            info="Number of results to return.",
            value=4,
            advanced=True,
        ),
    ]

    def _get_client(self) -> QdrantClient:
        """Create a QdrantClient with the configured settings."""
        use_https = getattr(self, "https", False)
        
        server_kwargs = {
            "host": self.host or "localhost",
            "port": int(self.port) if self.port else 6333,
            "grpc_port": int(self.grpc_port) if self.grpc_port else 6334,
            "https": use_https,
            "api_key": self.api_key if self.api_key else None,
            "prefix": self.prefix if self.prefix else None,
            "timeout": int(self.timeout) if self.timeout else None,
            "path": self.path if self.path else None,
        }

        # Only use url if explicitly provided, otherwise use host/port
        if self.url:
            server_kwargs["url"] = self.url
            server_kwargs.pop("host", None)
            server_kwargs.pop("port", None)
            server_kwargs.pop("grpc_port", None)

        # Filter out None values but KEEP False values (important for https=False)
        server_kwargs = {k: v for k, v in server_kwargs.items() if v is not None}
        
        return QdrantClient(**server_kwargs)

    @check_cached_vector_store
    def build_vector_store(self) -> QdrantVectorStore:
        # Convert DataFrame to Data if needed using parent's method
        self.ingest_data = self._prepare_ingest_data()

        # Extract texts, metadatas, and generate content-based IDs for deduplication
        texts = []
        metadatas = []
        content_ids = []
        
        for _input in self.ingest_data or []:
            if isinstance(_input, Data):
                doc = _input.to_lc_document()
                data_dict = _input.data if hasattr(_input, 'data') else {}
            elif isinstance(_input, dict):
                doc_text = _input.get("text", _input.get("page_content", str(_input)))
                doc = type('Doc', (), {'page_content': doc_text, 'metadata': _input})()
                data_dict = _input
            else:
                continue
            
            texts.append(doc.page_content)
            metadatas.append(doc.metadata if hasattr(doc, 'metadata') else {})
            
            # Generate content hash for deduplication
            # Use the full data dict (which includes all fields except embeddings)
            content_id = generate_content_hash(data_dict)
            content_ids.append(content_id)

        # If no valid embedding model, try to extract pre-computed embeddings from data
        embedding_model = self.embedding
        if not isinstance(embedding_model, Embeddings):
            # Try to create embeddings from data items that contain pre-computed vectors
            extracted = PrecomputedEmbeddingsFromData(self.ingest_data)
            if extracted.has_embeddings():
                embedding_model = extracted
            else:
                msg = "Invalid embedding object. Either connect an Embeddings model or provide Data with pre-computed 'embeddings' field."
                raise TypeError(msg)

        # Build connection kwargs (don't pass client directly - it can't be deepcopied)
        use_https = getattr(self, "https", False)
        connection_kwargs = {
            "host": self.host or "localhost",
            "port": int(self.port) if self.port else 6333,
            "grpc_port": int(self.grpc_port) if self.grpc_port else 6334,
            "https": use_https,
            "api_key": self.api_key if self.api_key else None,
            "prefix": self.prefix if self.prefix else None,
            "timeout": int(self.timeout) if self.timeout else None,
            "path": self.path if self.path else None,
        }
        
        # Use url if provided
        if self.url:
            connection_kwargs["url"] = self.url
            connection_kwargs.pop("host", None)
            connection_kwargs.pop("port", None)
            connection_kwargs.pop("grpc_port", None)
        
        # Filter out None values but KEEP False values
        connection_kwargs = {k: v for k, v in connection_kwargs.items() if v is not None}
        
        if texts:
            # Use from_texts with content-based IDs for automatic deduplication
            # Same content = same ID = upsert instead of duplicate
            vector_store = QdrantVectorStore.from_texts(
                texts=texts,
                embedding=embedding_model,
                metadatas=metadatas,
                ids=content_ids,  # Content-based IDs for deduplication
                collection_name=self.collection_name,
                content_payload_key=self.content_payload_key,
                metadata_payload_key=self.metadata_payload_key,
                **connection_kwargs,
            )
        else:
            # Connect to existing collection - need a client for this
            client = self._get_client()
            vector_store = QdrantVectorStore(
                client=client,
                collection_name=self.collection_name,
                embedding=embedding_model,
                content_payload_key=self.content_payload_key,
                metadata_payload_key=self.metadata_payload_key,
            )

        return vector_store

    def search_documents(self) -> list[Data]:
        vector_store = self.build_vector_store()

        if self.search_query and isinstance(self.search_query, str) and self.search_query.strip():
            try:
                docs = vector_store.similarity_search(
                    query=self.search_query,
                    k=self.number_of_results,
                )

                data = docs_to_data(docs)
                self.status = data
                return data
            except Exception as e:
                error_msg = str(e)
                if "doesn't exist" in error_msg or "not found" in error_msg.lower():
                    # Collection doesn't exist yet - return empty results
                    self.status = f"Collection '{self.collection_name}' not found. Please ingest documents first."
                    return []
                raise
        return []
