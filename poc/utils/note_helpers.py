"""
リフレクションノート生成用のヘルパー関数

generate_note.pyから抽出した補助関数群
"""
import os
import re
import unicodedata
from typing import List, Tuple, Dict, Any
from google import genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from rag.exceptions import GeminiQuotaError, is_quota_error


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(GeminiQuotaError)
)
def generate_project_keywords(client: genai.Client, project_name: str) -> str:
    """プロジェクト名から検索用キーワードを生成

    Args:
        client: Gemini APIクライアント
        project_name: プロジェクト名

    Returns:
        生成されたキーワード（スペース区切り）
    """
    prompt = f"""プロジェクト名から、そのプロジェクトに関連しそうなキーワードを抽出してください。

プロジェクト名: {project_name}

以下の観点でキーワードを生成してください：
- 業界/ドメイン（金融、EC、製造、物流、小売等）
- 技術/ツール（AI、データ分析、API、クラウド等）
- 課題/目的（業務改善、自動化、統合、最適化等）
- プロジェクトタイプ（システム開発、データ基盤、分析、導入等）

出力形式: スペース区切りのキーワード（説明不要、日本語可）
例: EC データ分析 売上予測 機械学習 AWS"""

    model_name = os.getenv('GEMINI_MODEL', 'gemini-2.5-pro ')

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=genai.types.GenerateContentConfig(max_output_tokens=8192,temperature=0.3)
        )
        keywords = response.text.strip()
        # 改行やタブをスペースに変換
        keywords = ' '.join(keywords.split())
        print(f"[INFO] 生成されたキーワード: {keywords}")
        return keywords
    except Exception as e:
        if is_quota_error(e):
            raise GeminiQuotaError(str(e))
        else:
            print(f"[WARN] キーワード生成に失敗しました: {e}")
            # フォールバック: プロジェクト名をそのまま返す
            return project_name


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(GeminiQuotaError)
)
def generate_multiple_queries(
    client: genai.Client,
    project_name: str,
    base_context: str = "",
    num_queries: int = None
) -> List[str]:
    """RAG-Fusion用：同じ意図を持つ複数の異なるクエリを生成

    同じ情報を探すが、異なる表現・観点のクエリを生成することで、
    検索のカバレッジを向上させる。

    Args:
        client: Gemini APIクライアント
        project_name: プロジェクト名
        base_context: プロジェクトの基本情報（オプション）
        num_queries: 生成するクエリ数（Noneの場合は環境変数から取得）

    Returns:
        生成されたクエリのリスト
    """
    if num_queries is None:
        num_queries = int(os.getenv('RAG_FUSION_NUM_QUERIES', '3'))

    prompt = f"""プロジェクト名「{project_name}」に関連する情報を検索するため、
異なる観点から{num_queries}つの検索クエリを生成してください。

【プロジェクト情報】
{base_context if base_context else "（基本情報なし）"}

【要件】
- 同じ情報を探すが、異なる表現・観点のクエリにする
- 以下の観点で多様化する：
  1. 業界用語 vs 一般用語
  2. 技術スタック vs ビジネス価値
  3. 課題領域 vs 解決手段
  4. プロジェクト規模 vs 実装詳細
- 各クエリは200文字以内
- 検索に適した具体的なキーワードを含める

【出力形式】
クエリ1: （検索クエリ）
クエリ2: （検索クエリ）
クエリ3: （検索クエリ）

説明文は不要、各行1つのクエリのみを出力してください。"""

    model_name = os.getenv('GEMINI_MODEL', 'gemini-2.5-pro ')

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=genai.types.GenerateContentConfig(max_output_tokens=8192,temperature=0.3)
        )

        # クエリを抽出
        queries = []
        for line in response.text.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            # "クエリN:" のプレフィックスを除去
            if ':' in line:
                query = line.split(':', 1)[1].strip()
            else:
                query = line

            # 長すぎるクエリは短縮
            if len(query) > 100:
                query = query[:100]

            queries.append(query)

        # 目標数に達していない場合はプロジェクト名を追加
        while len(queries) < num_queries:
            queries.append(project_name)

        queries = queries[:num_queries]
        print(f"[INFO] RAG-Fusion: {len(queries)}個のクエリを生成")
        for i, q in enumerate(queries, 1):
            print(f"  クエリ{i}: {q[:60]}...")

        return queries

    except Exception as e:
        if is_quota_error(e):
            raise GeminiQuotaError(str(e))
        else:
            print(f"[WARN] 複数クエリ生成に失敗、フォールバック: {e}")
            # フォールバック: プロジェクト名を複数返す
            return [project_name] * num_queries


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(GeminiQuotaError)
)
def generate_project_summary(client: genai.Client, project_name: str, current_results: List[Tuple]) -> str:
    """現在のプロジェクトの情報から概要を生成

    Args:
        client: Gemini APIクライアント
        project_name: プロジェクト名
        current_results: 現在のプロジェクトの検索結果

    Returns:
        プロジェクト概要
    """
    # 上位3件の情報を使用
    context = ""
    if current_results:
        # 簡易的なコンテキスト作成（上位3件まで）
        for i, (doc, distance) in enumerate(current_results[:3]):
            if i >= 3:
                break
            context += f"[文書{i+1}]\n{doc.text[:500]}\n\n"

    prompt = f"""以下のプロジェクト情報から、このプロジェクトの特徴・概要を抽出してください。

プロジェクト名: {project_name}

プロジェクト情報:
{context if context else "（情報なし）"}

以下の観点で100文字程度で要約してください：
- 業界/分野
- 主要な課題/目的
- 使用技術
- プロジェクト規模

出力形式: 簡潔な文章で要約（箇条書き不要）"""

    model_name = os.getenv('GEMINI_MODEL', 'gemini-2.5-pro ')

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=genai.types.GenerateContentConfig(max_output_tokens=8192,temperature=0.3)
        )
        summary = response.text.strip()
        print(f"[INFO] 生成されたプロジェクト概要: {summary[:100]}...")
        return summary
    except Exception as e:
        if is_quota_error(e):
            raise GeminiQuotaError(str(e))
        else:
            print(f"[WARN] プロジェクト概要生成に失敗しました: {e}")
            # フォールバック: プロジェクト名をそのまま返す
            return project_name


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(GeminiQuotaError)
)
def generate_similar_project_query(client: genai.Client, project_summary: str, project_name: str) -> str:
    """プロジェクト概要から類似プロジェクト検索用クエリを生成

    Args:
        client: Gemini APIクライアント
        project_summary: プロジェクト概要
        project_name: プロジェクト名（除外用）

    Returns:
        類似プロジェクト検索用クエリ
    """
    prompt = f"""以下のプロジェクト概要から、類似プロジェクトを検索するためのクエリを生成してください。

プロジェクト名: {project_name}
プロジェクト概要: {project_summary}

【要件】
- プロジェクト名自体は含めない（類似プロジェクトを探すため）
- 業界、技術、課題、規模などの特徴を含める
- 100文字以内で簡潔に

【出力形式】
検索クエリのみ（説明不要）"""

    model_name = os.getenv('GEMINI_MODEL', 'gemini-2.5-pro ')

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=genai.types.GenerateContentConfig(max_output_tokens=8192,temperature=0.3)
        )
        query = response.text.strip()
        print(f"[INFO] 生成された類似プロジェクト検索クエリ: {query}")
        return query
    except Exception as e:
        if is_quota_error(e):
            raise GeminiQuotaError(str(e))
        else:
            print(f"[WARN] 類似プロジェクトクエリ生成に失敗しました: {e}")
            # フォールバック: プロジェクト概要をそのまま返す
            return project_summary


def normalize_tag(tag: str) -> str:
    """タグを正規化

    Args:
        tag: 元のタグ

    Returns:
        正規化されたタグ
    """
    # Unicode正規化（全角/半角統一）
    tag = unicodedata.normalize('NFKC', tag)
    # 前後の空白削除
    tag = tag.strip()
    # 大文字小文字統一（英語のみ小文字化）
    tag = tag.lower()
    return tag


def filter_tags(tags: List[str]) -> List[str]:
    """タグをフィルタリング

    - 重複削除（正規化後）
    - 空文字削除
    - ノイズタグ削除

    Args:
        tags: タグリスト

    Returns:
        フィルタリング済みタグリスト
    """
    # ノイズタグ（除外するタグ）
    noise_tags = {
        'その他', 'なし', '不明', 'n/a', 'na', 'none', 'null',
        'プロジェクト', 'システム', '案件', '業務'
    }

    filtered = []
    seen = set()

    for tag in tags:
        # 正規化
        normalized = normalize_tag(tag)

        # 空文字チェック
        if not normalized:
            continue

        # 重複チェック
        if normalized in seen:
            continue

        # ノイズチェック
        if normalized in noise_tags:
            continue

        # 長すぎるタグをスキップ（50文字以上）
        if len(normalized) > 50:
            continue

        filtered.append(tag)  # 元のタグを保持
        seen.add(normalized)

    return filtered


def sanitize_project_name(project_name: str) -> str:
    """プロジェクト名をファイル名として安全な形式に変換

    Args:
        project_name: プロジェクト名

    Returns:
        サニタイズされたプロジェクト名
    """
    # ファイルシステムで使えない文字を除去
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', project_name)
    # 連続するアンダースコアを1つに
    sanitized = re.sub(r'_+', '_', sanitized)
    # 前後の空白とアンダースコアを削除
    sanitized = sanitized.strip('_ ')
    return sanitized


def validate_tags_schema(data: Any) -> Dict:
    """タグデータのスキーマ検証

    Args:
        data: 検証するデータ

    Returns:
        検証済みタグデータ（修正済み）

    Raises:
        ValueError: スキーマ不正の場合
    """
    if not isinstance(data, dict):
        raise ValueError(f"タグデータはdictである必要があります。実際の型: {type(data)}")

    # 必須フィールド
    required_fields = ['project_name', 'generated_at', 'tags']
    for field in required_fields:
        if field not in data:
            raise ValueError(f"必須フィールドが不足: {field}")

    # tags フィールドの検証と修正
    if not isinstance(data['tags'], dict):
        raise ValueError(f"tags フィールドはdictである必要があります。実際の型: {type(data['tags'])}")

    # 各タグカテゴリの検証
    expected_categories = [
        'industry_domain', 'technology_stack', 'business_objective',
        'project_scale', 'implementation_phase', 'challenges_risks'
    ]

    for category in expected_categories:
        if category not in data['tags']:
            print(f"[WARN] タグカテゴリが不足: {category} - 空リストで補完")
            data['tags'][category] = []
        elif not isinstance(data['tags'][category], list):
            print(f"[WARN] {category} がリストではありません - 空リストに修正")
            data['tags'][category] = []

    return data
