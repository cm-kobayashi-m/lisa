"""
Gemini API用の共通例外クラスとユーティリティ関数

7つのファイルで重複していたエラーハンドリングを統一化します。
"""


class GeminiQuotaError(Exception):
    """Gemini APIのクォータ制限エラー"""
    pass


class GeminiNetworkError(Exception):
    """Gemini APIのネットワーク接続エラー"""
    pass


def is_quota_error(exception: Exception) -> bool:
    """
    クォータエラーかどうかを判定

    429、503、overloadedエラーを含むクォータ関連エラーを検出します。

    Args:
        exception: 検査する例外オブジェクト

    Returns:
        クォータエラーの場合True
    """
    error_msg = str(exception)
    return (
        '429' in error_msg
        or '503' in error_msg
        or 'quota' in error_msg.lower()
        or 'overloaded' in error_msg.lower()
        or 'UNAVAILABLE' in error_msg
        or 'rate limit' in error_msg.lower()
    )


def is_network_error(exception: Exception) -> bool:
    """
    ネットワークエラーかどうかを判定

    ConnectError, TimeoutException, DNS エラーなどを検出します。

    Args:
        exception: 検査する例外オブジェクト

    Returns:
        ネットワークエラーの場合True
    """
    # httpx の ConnectError, TimeoutException を検出
    exception_type = type(exception).__name__
    if exception_type in ('ConnectError', 'TimeoutException'):
        return True

    error_msg = str(exception).lower()
    network_keywords = [
        'nodename nor servname provided',
        'connection refused',
        'connection reset',
        'connection timeout',
        'name resolution',
        'dns',
        'network unreachable',
        'timeout',
        'connect',
    ]
    return any(keyword in error_msg for keyword in network_keywords)
