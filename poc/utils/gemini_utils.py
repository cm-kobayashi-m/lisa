"""
Gemini API関連の共通ユーティリティ関数
"""

import os
import sys
import json
import re
from typing import Optional, Dict, Any
from google import genai


class GeminiQuotaError(Exception):
    """Gemini APIのクォータ制限エラー"""
    pass


def is_quota_error(exception: Exception) -> bool:
    """
    クォータエラーかどうかを判定

    Args:
        exception: 判定する例外

    Returns:
        クォータエラーの場合True
    """
    error_msg = str(exception).lower()
    quota_keywords = [
        '429',
        '503',
        'quota',
        'rate limit',
        'resource exhausted',
        'too many requests',
        'overloaded',
        'unavailable'
    ]
    return any(keyword in error_msg for keyword in quota_keywords)


def initialize_gemini_client() -> genai.Client:
    """
    Gemini APIクライアントを初期化

    Returns:
        初期化されたGemini APIクライアント

    Raises:
        SystemExit: GEMINI_API_KEYが設定されていない場合
    """
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("[ERROR] GEMINI_API_KEY が環境変数に設定されていません。")
        sys.exit(1)

    return genai.Client(api_key=api_key)


def extract_json_from_response(response_text: str) -> Optional[Dict[str, Any]]:
    """
    LLMレスポンスからJSONを抽出・修復

    コードブロック、不完全なJSON、途中で切れたJSONなどに対応

    Args:
        response_text: LLMからのレスポンステキスト

    Returns:
        パースされたJSON辞書、失敗時はNone
    """
    response_text = response_text.strip()

    # コードブロックを除去
    if response_text.startswith('```'):
        code_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response_text, re.DOTALL)
        if code_block_match:
            response_text = code_block_match.group(1).strip()

    # 直接パースを試みる
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # JSONオブジェクトを抽出
    json_str = _extract_json_object(response_text)

    if not json_str:
        return None

    # 再度パースを試みる
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # 不完全なJSONを修復
    json_str = _repair_incomplete_json(json_str)

    # 修復後にパース
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def _extract_json_object(text: str) -> Optional[str]:
    """
    テキストから最初のJSONオブジェクトを抽出

    Args:
        text: 抽出元のテキスト

    Returns:
        抽出されたJSON文字列、見つからない場合はNone
    """
    start = text.find('{')
    if start == -1:
        return None

    bracket_count = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        char = text[i]

        if escape_next:
            escape_next = False
            continue

        if char == '\\':
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if not in_string:
            if char == '{':
                bracket_count += 1
            elif char == '}':
                bracket_count -= 1
                if bracket_count == 0:
                    return text[start:i+1]

    # 不完全なJSONの場合、最後まで取得
    return text[start:]


def _repair_incomplete_json(json_str: str) -> str:
    """
    不完全なJSON文字列を修復

    Args:
        json_str: 修復するJSON文字列

    Returns:
        修復されたJSON文字列
    """
    if json_str.rstrip().endswith('}'):
        return json_str

    # 文字列が途中で切れている場合
    if '"' in json_str and json_str.count('"') % 2 == 1:
        json_str += '"'

    # 配列が閉じていない場合
    if '[' in json_str:
        bracket_diff = json_str.count('[') - json_str.count(']')
        json_str += ']' * bracket_diff

    # オブジェクトが閉じていない場合
    brace_diff = json_str.count('{') - json_str.count('}')
    json_str += '}' * brace_diff

    return json_str
