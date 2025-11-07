"""
共通ユーティリティモジュール

各種共通機能を提供します。

主要モジュール:
- gemini_client: Gemini APIクライアント管理
- llm_response: LLMレスポンス処理
- note_helpers: リフレクションノート生成用ヘルパー
- thought_analyzer: 思考プロセス分析（統一版）
"""
from .gemini_client import initialize_gemini_client, get_api_key
from .llm_response import extract_content
from .note_helpers import (
    generate_project_keywords,
    generate_multiple_queries,
    generate_project_summary,
    generate_similar_project_query,
    normalize_tag,
    filter_tags,
    sanitize_project_name,
    validate_tags_schema
)
from .thought_analyzer import (
    analyze_reflection_note_thought_process,
    analyze_document_generation_thought_process,
    save_thought_analysis
)

__all__ = [
    # Gemini クライアント
    'initialize_gemini_client',
    'get_api_key',

    # LLM レスポンス処理
    'extract_content',

    # ノート生成ヘルパー
    'generate_project_keywords',
    'generate_multiple_queries',
    'generate_project_summary',
    'generate_similar_project_query',
    'normalize_tag',
    'filter_tags',
    'sanitize_project_name',
    'validate_tags_schema',

    # 思考プロセス分析
    'analyze_reflection_note_thought_process',
    'analyze_document_generation_thought_process',
    'save_thought_analysis',
]
