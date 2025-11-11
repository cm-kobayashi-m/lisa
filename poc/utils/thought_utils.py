"""
思考プロセス分析の共通ユーティリティ関数
"""

import json
from typing import Dict, Any, Optional


def build_yaml_frontmatter(thought_process: Dict[str, Any], json_file_path: Optional[str] = None) -> str:
    """
    YAMLフロントマターを生成

    Args:
        thought_process: 思考プロセスの分析結果
        json_file_path: 対応するJSONファイルパス

    Returns:
        YAMLフロントマター文字列
    """
    frontmatter = ["---"]

    # 基本メタデータ
    if 'project_name' in thought_process:
        frontmatter.append(f"project_name: {json.dumps(thought_process.get('project_name', 'unknown'))}")

    if 'document_type' in thought_process:
        frontmatter.append(f"document_type: {json.dumps(thought_process.get('document_type', 'unknown'))}")

    frontmatter.append(f"generated_at: {json.dumps(thought_process.get('generated_at', 'N/A'))}")
    frontmatter.append(f"model: {json.dumps(thought_process.get('model', 'N/A'))}")

    # ノート/ドキュメントファイルパス
    if 'note_file' in thought_process:
        frontmatter.append(f"note_file: {json.dumps(thought_process['note_file'])}")
    if 'document_file' in thought_process:
        frontmatter.append(f"document_file: {json.dumps(thought_process['document_file'])}")

    # JSONファイルパス
    if json_file_path:
        frontmatter.append(f"json_file: {json.dumps(str(json_file_path))}")

    # エラー情報（存在する場合）
    if 'error' in thought_process:
        frontmatter.append(f"has_error: true")

    frontmatter.append("---")
    frontmatter.append("")  # 空行

    return '\n'.join(frontmatter)


def enhance_markdown_with_metadata(
    markdown_content: str,
    thought_process: Dict[str, Any],
    json_file_path: Optional[str] = None,
    toc_sections: Optional[list[str]] = None
) -> str:
    """
    Markdown本文にメタデータ（YAMLフロントマター、目次）を追加

    Args:
        markdown_content: Markdown本文
        thought_process: 思考プロセスの分析結果
        json_file_path: 対応するJSONファイルパス
        toc_sections: 目次に含めるセクション名のリスト（Noneの場合はデフォルト）

    Returns:
        メタデータ付きMarkdown文字列
    """
    # YAMLフロントマターを追加
    frontmatter = build_yaml_frontmatter(thought_process, json_file_path)

    # 目次を生成
    if toc_sections is None:
        # デフォルトの目次（汎用的）
        toc_sections = [
            "思考プロセス",
            "重要な判断",
            "情報源の活用",
            "品質評価",
            "メタ振り返り"
        ]

    toc_lines = ["## 目次", ""]
    for section in toc_sections:
        # Markdown アンカー形式に変換（簡易版）
        anchor = section.lower().replace(" ", "-")
        toc_lines.append(f"- [{section}](#{anchor})")
    toc_lines.append("")
    toc = '\n'.join(toc_lines)

    # 組み立て
    enhanced = frontmatter + '\n' + toc + '\n' + markdown_content

    return enhanced
