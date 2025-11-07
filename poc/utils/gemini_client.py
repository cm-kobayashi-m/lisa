"""
Gemini APIクライアント管理ユーティリティ

3つのファイルで重複していたクライアント初期化ロジックを統一化します。
"""
import os
import sys
from google import genai


def initialize_gemini_client() -> genai.Client:
    """
    Gemini APIクライアントを初期化

    環境変数からAPIキーを取得し、Geminiクライアントを初期化します。
    APIキーが設定されていない場合はエラーメッセージを表示して終了します。

    Returns:
        初期化されたGemini APIクライアント

    Raises:
        SystemExit: APIキーが設定されていない場合

    Example:
        >>> client = initialize_gemini_client()
        >>> # クライアントを使用してAPI呼び出し
    """
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("[ERROR] GEMINI_API_KEY が環境変数に設定されていません。")
        sys.exit(1)

    return genai.Client(api_key=api_key)


def get_api_key() -> str:
    """
    環境変数からGemini APIキーを取得

    Returns:
        APIキー文字列

    Raises:
        ValueError: APIキーが設定されていない場合

    Example:
        >>> api_key = get_api_key()
    """
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY が環境変数に設定されていません")
    return api_key
