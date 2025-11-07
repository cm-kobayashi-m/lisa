"""
LLMレスポンス処理ユーティリティ

4つのファイルで重複していたLLMレスポンス抽出ロジックを統一化します。
Gemini 2.0などでcontentがリスト形式で返される場合にも対応します。
"""
from typing import Any


def extract_content(response: Any) -> str:
    """
    LLMレスポンスからコンテンツを抽出

    様々な形式のLLMレスポンス（文字列、dict、list、bytes、オブジェクト）から
    テキストコンテンツを安全に抽出します。

    Args:
        response: LLMからのレスポンス（様々な形式に対応）

    Returns:
        抽出されたテキストコンテンツ（空の場合は空文字列）

    Examples:
        >>> extract_content("Hello")
        'Hello'
        >>> extract_content({"text": "World"})
        'World'
        >>> extract_content({"parts": [{"text": "Foo"}]})
        'Foo'
    """
    if response is None:
        return ""

    # 公式SDK系: response.text がある場合は最短経路
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    # contentが無ければ response 自体を中身とみなす
    content = getattr(response, "content", response)
    if content is None:
        return ""

    # 文字列
    if isinstance(content, str):
        return content.strip()

    # バイト列
    if isinstance(content, (bytes, bytearray)):
        try:
            return content.decode("utf-8", errors="ignore").strip()
        except Exception:
            return ""

    # dict（Gemini系: {'parts': [...]} / {'text': '...'} など）
    if isinstance(content, dict):
        if "text" in content and isinstance(content["text"], str):
            return content["text"].strip()
        parts = content.get("parts")
        if isinstance(parts, list):
            content = parts  # 下のlist処理へ
        else:
            return str(content).strip()

    # list（LangChainのAIMessage.contentがリスト化されるケース等）
    if isinstance(content, list):
        texts = []
        for part in content:
            if part is None:
                continue
            if isinstance(part, str):
                t = part
            elif isinstance(part, dict):
                t = part.get("text")
            else:
                t = getattr(part, "text", None)
            if isinstance(t, str) and t:
                texts.append(t)
        return "".join(texts).strip()

    # フォールバック
    try:
        return str(content).strip()
    except Exception:
        return ""
