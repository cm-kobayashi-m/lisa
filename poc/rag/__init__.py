"""
LISA RAG (Retrieval-Augmented Generation) モジュール

Amazon S3 Vectors（プレビュー版）を使用したベクトル検索機能を提供し、
Gemini APIと連携してRAGベースの分析を実現します。

主要コンポーネント:
- S3VectorStore: ベクトルストア（S3 Vectors統合）
- GeminiEmbeddings: Gemini埋め込みAPI
- RAGRetriever: RAG検索機能
- EnhancedRAGSearch: CRAG統合検索
- RAGFusion: 複数クエリ統合検索

例外クラス:
- GeminiQuotaError: APIクォータエラー
- GeminiNetworkError: ネットワークエラー
"""

__version__ = "0.1.0"

# コアコンポーネント
from .vector_store import S3VectorStore, Document
from .embeddings import GeminiEmbeddings
from .rag_retriever import RAGRetriever
from .enhanced_rag_search import create_enhanced_rag_search, EnhancedRAGConfig
from .rag_fusion import rag_fusion_search, reciprocal_rank_fusion, apply_hybrid_scoring
from .corrective_rag import InternalCRAG, create_internal_crag
from .knowledge_refiner import KnowledgeRefiner
from .evaluator import RelevanceEvaluator, RelevanceLevel, EvaluationResult, DocumentTypeAwareEvaluator, create_evaluator
from .document_classifier import DocumentClassifier

# 例外クラス
from .exceptions import GeminiQuotaError, GeminiNetworkError, is_quota_error, is_network_error

__all__ = [
    # コアコンポーネント
    "S3VectorStore",
    "Document",
    "GeminiEmbeddings",
    "RAGRetriever",

    # 高度な検索機能
    "create_enhanced_rag_search",
    "EnhancedRAGConfig",
    "rag_fusion_search",
    "reciprocal_rank_fusion",
    "apply_hybrid_scoring",

    # CRAG & Knowledge Refinement
    "InternalCRAG",
    "create_internal_crag",
    "RelevanceLevel",
    "RelevanceEvaluator",
    "EvaluationResult",
    "DocumentTypeAwareEvaluator",
    "create_evaluator",
    "KnowledgeRefiner",

    # ユーティリティ
    "DocumentClassifier",

    # 例外
    "GeminiQuotaError",
    "GeminiNetworkError",
    "is_quota_error",
    "is_network_error",
]
