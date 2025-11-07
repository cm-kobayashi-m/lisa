"""
思考プロセス分析機能（統一版）

thought_process_analyzer.pyとdocument_thought_analyzer.pyを統合し、
リフレクションノート生成とドキュメント生成の両方の思考プロセス分析を提供します。
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from google import genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from rag.exceptions import GeminiQuotaError, is_quota_error


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(GeminiQuotaError)
)
def analyze_reflection_note_thought_process(
    client: genai.Client,
    project_name: str,
    reflection_note: str,
    rag_context: Optional[str] = None,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    リフレクションノート生成の思考プロセスを分析

    Args:
        client: Gemini APIクライアント
        project_name: プロジェクト名
        reflection_note: 生成されたリフレクションノート
        rag_context: 使用されたRAGコンテキスト（オプション）
        model_name: 使用するモデル名（デフォルト: 環境変数から取得）

    Returns:
        思考プロセスの分析結果（JSON形式）
    """

    if model_name is None:
        model_name = os.getenv('THOUGHT_ANALYSIS_MODEL', os.getenv('GEMINI_MODEL', 'gemini-2.5-pro '))

    # コンテキストの要約（トークン削減のため）
    context_summary = ""
    if rag_context:
        context_lines = rag_context.split('\n')[:20]
        context_summary = '\n'.join(context_lines)
        if len(context_lines) < len(rag_context.split('\n')):
            context_summary += "\n...(以下省略)"

    analysis_prompt = f"""
あなたが以下のリフレクションノートを生成した際の、詳細な思考プロセスと判断根拠を教えてください。
特に「どの情報源のどの部分」から「何を読み取り」「どのような判断」をしたのか、具体的に説明してください。

## 生成されたリフレクションノート
{reflection_note}

## 使用されたコンテキスト（要約）
{context_summary if context_summary else "なし"}

## 質問事項
以下の観点について、できる限り具体的かつ詳細に、JSONフォーマットで回答してください：

1. **reasoning_process**: 思考の流れと根拠
   - どの情報から何を読み取ったか
   - どのような判断基準で重要度を評価したか
   - なぜその表現・構成を選択したか

2. **information_sources**: 使用した情報源の詳細
   - 各セクションで参照した具体的な情報源
   - それぞれの情報の信頼性評価
   - 情報間の矛盾や補完関係

3. **key_insights**: 重要な洞察
   - プロジェクトの核心的な特徴
   - 気づいたパターンや傾向
   - 予想される課題や注意点

4. **confidence_assessment**: 確信度評価
   - 各セクションの確信度（高/中/低）
   - 不確実性が高い箇所とその理由
   - 追加で欲しい情報

5. **alternative_interpretations**: 代替解釈
   - 別の解釈や視点の可能性
   - 判断に迷った箇所とその理由

**出力形式**: 必ず有効なJSONで、上記5つのキーを含めてください。
"""

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=analysis_prompt,
            config=genai.types.GenerateContentConfig(
                max_output_tokens=8192,
                temperature=0.3,
                response_mime_type="application/json"
            )
        )

        analysis_text = response.text.strip()

        try:
            analysis_data = json.loads(analysis_text)
        except json.JSONDecodeError as e:
            print(f"[WARN] JSON解析エラー、再試行します: {e}")
            analysis_data = {
                "reasoning_process": analysis_text,
                "information_sources": "解析失敗",
                "key_insights": [],
                "confidence_assessment": {},
                "alternative_interpretations": []
            }

        analysis_data["project_name"] = project_name
        analysis_data["analyzed_at"] = datetime.now().isoformat()
        analysis_data["model_used"] = model_name

        return analysis_data

    except Exception as e:
        if is_quota_error(e):
            raise GeminiQuotaError(str(e))
        else:
            print(f"[ERROR] 思考プロセス分析に失敗: {e}")
            return {
                "project_name": project_name,
                "analyzed_at": datetime.now().isoformat(),
                "error": str(e),
                "reasoning_process": "分析失敗",
                "information_sources": [],
                "key_insights": [],
                "confidence_assessment": {},
                "alternative_interpretations": []
            }


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(GeminiQuotaError)
)
def analyze_document_generation_thought_process(
    client: genai.Client,
    document_type: str,
    source_document: str,
    generated_document: str,
    rag_results: Optional[str] = None,
    project_context: Optional[Dict[str, Any]] = None,
    additional_prompt: Optional[str] = None,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    ドキュメント生成の思考プロセスを分析

    Args:
        client: Gemini APIクライアント
        document_type: ドキュメント種別（hearing-sheet, proposal）
        source_document: 入力ドキュメント（リフレクションノートやヒアリングシート）
        generated_document: 生成されたドキュメント
        rag_results: 使用されたRAG検索結果（オプション）
        project_context: プロジェクトコンテキスト
        additional_prompt: 追加指示
        model_name: 使用するモデル名

    Returns:
        思考プロセスの分析結果（JSON形式）
    """

    if model_name is None:
        model_name = os.getenv('THOUGHT_ANALYSIS_MODEL', os.getenv('GEMINI_MODEL', 'gemini-2.5-pro '))

    # ドキュメント種別の日本語名
    doc_type_ja = {
        'hearing-sheet': 'ヒアリングシート',
        'proposal': '提案書'
    }.get(document_type, document_type)

    # コンテキストの要約
    source_summary = source_document[:3000] if len(source_document) > 3000 else source_document
    generated_summary = generated_document[:3000] if len(generated_document) > 3000 else generated_document

    rag_summary = ""
    if rag_results:
        rag_lines = rag_results.split('\n')[:30]
        rag_summary = '\n'.join(rag_lines)
        if len(rag_lines) < len(rag_results.split('\n')):
            rag_summary += "\n...(以下省略)"

    analysis_prompt = f"""
あなたが以下の{doc_type_ja}を生成した際の、詳細な思考プロセスと判断根拠を教えてください。

## 入力ドキュメント（要約）
{source_summary}

## 生成された{doc_type_ja}（要約）
{generated_summary}

## RAG検索結果（要約）
{rag_summary if rag_summary else "なし"}

## プロジェクトコンテキスト
{json.dumps(project_context, ensure_ascii=False, indent=2) if project_context else "なし"}

## 追加指示
{additional_prompt if additional_prompt else "なし"}

## 質問事項
以下の観点について、JSONフォーマットで詳細に回答してください：

1. **content_selection**: コンテンツ選択の理由
   - 入力から何を抽出したか
   - RAG結果からどの情報を採用したか
   - 情報の優先順位付け基準

2. **structure_design**: 構成設計の判断
   - {doc_type_ja}の構成をどう決定したか
   - セクション分けの根拠
   - 情報の配置順序の理由

3. **tone_and_style**: 表現スタイルの選択
   - 文体・トーンの選択理由
   - 対象読者の想定
   - 専門用語レベルの調整

4. **rag_integration**: RAG情報の統合
   - RAG結果の活用方法
   - 入力ドキュメントとRAG情報のバランス
   - 矛盾する情報の処理

5. **quality_considerations**: 品質への配慮
   - 確信度が低い箇所
   - 改善の余地がある点
   - 追加で欲しい情報

**出力形式**: 必ず有効なJSONで、上記5つのキーを含めてください。
"""

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=analysis_prompt,
            config=genai.types.GenerateContentConfig(
                max_output_tokens=8192,
                temperature=0.3,
                response_mime_type="application/json"
            )
        )

        analysis_text = response.text.strip()

        try:
            analysis_data = json.loads(analysis_text)
        except json.JSONDecodeError as e:
            print(f"[WARN] JSON解析エラー: {e}")
            analysis_data = {
                "content_selection": analysis_text,
                "structure_design": "解析失敗",
                "tone_and_style": {},
                "rag_integration": {},
                "quality_considerations": []
            }

        analysis_data["document_type"] = document_type
        analysis_data["analyzed_at"] = datetime.now().isoformat()
        analysis_data["model_used"] = model_name

        return analysis_data

    except Exception as e:
        if is_quota_error(e):
            raise GeminiQuotaError(str(e))
        else:
            print(f"[ERROR] ドキュメント思考プロセス分析に失敗: {e}")
            return {
                "document_type": document_type,
                "analyzed_at": datetime.now().isoformat(),
                "error": str(e),
                "content_selection": "分析失敗",
                "structure_design": {},
                "tone_and_style": {},
                "rag_integration": {},
                "quality_considerations": []
            }


def save_thought_analysis(
    analysis_data: Dict[str, Any],
    output_path: Path,
    document_name: str
) -> Tuple[bool, str]:
    """
    思考プロセス分析結果を保存

    Args:
        analysis_data: 分析データ
        output_path: 出力ディレクトリ
        document_name: ドキュメント名（ファイル名のベース）

    Returns:
        (成功フラグ, 保存パス)
    """
    try:
        output_path.mkdir(parents=True, exist_ok=True)

        file_path = output_path / f"{document_name}_thought_process.json"

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, ensure_ascii=False, indent=2)

        print(f"[INFO] 思考プロセス分析を保存: {file_path}")
        return True, str(file_path)

    except Exception as e:
        print(f"[ERROR] 思考プロセス分析の保存に失敗: {e}")
        return False, ""
