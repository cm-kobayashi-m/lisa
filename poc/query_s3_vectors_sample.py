#!/usr/bin/env python3
"""
S3 Vectorsクエリサンプルスクリプト

使用方法:
    # デフォルトクエリで検索
    python3 query_s3_vectors_sample.py

    # カスタムクエリで検索
    python3 query_s3_vectors_sample.py --query "LISAのアーキテクチャについて教えて"

    # 件数を指定
    python3 query_s3_vectors_sample.py --query "RAGエンジン" --k 10

    # プロジェクトでフィルタ
    python3 query_s3_vectors_sample.py --query "提案書" --project "LISAのPoCテスト"

    # ドキュメント種別でフィルタ
    python3 query_s3_vectors_sample.py --query "会議" --doc-type "meeting_minutes"
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent))

# RAGモジュールのインポート
from rag.vector_store import S3VectorStore, Document
from rag.embeddings import GeminiEmbeddings

# 環境変数読み込み
load_dotenv()

# Rich Console初期化
console = Console()


def format_metadata(metadata: Dict) -> str:
    """メタデータを整形して表示用文字列に変換"""

    # 表示順序を定義
    display_order = [
        "project_name",
        "file_name",
        "title",
        "document_type",
        "document_type_confidence",
        "heading_level",
        "section_path",
        "chunk_position",
        "relative_position",
        "information_density",
        "importance",
        "has_table",
        "has_list",
        "has_code",
        "has_numbers",
        "speakers",
        "temporal_index",
        "pages",
        "element_types",
        "created_at",
        "modified_at",
        "prev_section",
        "next_section",
        "chunk_index",
        "data_source",
        "source_id"
    ]

    result = []

    # 定義された順序で表示
    for key in display_order:
        if key in metadata and metadata[key] is not None:
            value = metadata[key]

            # 値の整形
            if isinstance(value, float):
                value = f"{value:.2f}"
            elif isinstance(value, list):
                if len(value) > 0:
                    value = ", ".join(str(v) for v in value)
                else:
                    continue  # 空のリストは表示しない
            elif isinstance(value, bool):
                value = "✓" if value else "✗"
            elif key in ["created_at", "modified_at"]:
                # ISO形式の日時を読みやすい形式に
                try:
                    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                    value = dt.strftime("%Y-%m-%d %H:%M")
                except:
                    pass

            # キー名を読みやすく変換
            display_key = key.replace("_", " ").title()
            result.append(f"  {display_key}: {value}")

    # その他の未定義キーも表示
    for key, value in metadata.items():
        if key not in display_order and value is not None:
            display_key = key.replace("_", " ").title()
            result.append(f"  {display_key}: {value}")

    return "\n".join(result)


def display_results(results: List[Tuple[Document, float]], query_text: str):
    """検索結果を美しく表示"""

    console.print(f"\n[bold cyan]検索クエリ:[/bold cyan] {query_text}")
    console.print(f"[bold green]検索結果:[/bold green] {len(results)}件\n")

    if not results:
        console.print("[yellow]該当するドキュメントが見つかりませんでした。[/yellow]")
        return

    for idx, (doc, score) in enumerate(results, 1):
        # スコアに基づく色分け
        if score > 0.8:
            score_color = "green"
        elif score > 0.6:
            score_color = "yellow"
        else:
            score_color = "red"

        # タイトル作成
        title = doc.metadata.get("title", f"Document {idx}")
        project = doc.metadata.get("project_name", "不明")
        file_name = doc.metadata.get("file_name", "不明")

        # パネルのタイトル
        panel_title = f"[{score_color}]#{idx}[/{score_color}] {title} | スコア: [{score_color}]{score:.3f}[/{score_color}]"

        # テキストプレビュー（最初の300文字）
        text_preview = doc.text[:300] + "..." if len(doc.text) > 300 else doc.text

        # コンテンツ作成
        content = f"""[bold]プロジェクト:[/bold] {project}
[bold]ファイル:[/bold] {file_name}

[bold]テキスト:[/bold]
{text_preview}

[bold]メタデータ:[/bold]
{format_metadata(doc.metadata)}"""

        # パネルで表示
        panel = Panel(
            content,
            title=panel_title,
            border_style=score_color,
            expand=True
        )
        console.print(panel)
        console.print()


def create_filter_dict(project: Optional[str], doc_type: Optional[str]) -> Optional[Dict]:
    """フィルタ条件を作成"""

    filters = []

    if project:
        filters.append({"project_name": {"$eq": project}})

    if doc_type:
        filters.append({"document_type": {"$eq": doc_type}})

    if len(filters) == 0:
        return None
    elif len(filters) == 1:
        return filters[0]
    else:
        return {"$and": filters}


def main():
    """メイン処理"""

    # コマンドライン引数
    parser = argparse.ArgumentParser(description="S3 Vectorsクエリサンプル")
    parser.add_argument(
        "--query",
        type=str,
        default="アーキテクチャについて教えてください",
        help="検索クエリテキスト"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="取得する結果の数（デフォルト: 5）"
    )
    parser.add_argument(
        "--project",
        type=str,
        help="プロジェクト名でフィルタ"
    )
    parser.add_argument(
        "--doc-type",
        type=str,
        choices=["technical_document", "meeting_minutes", "proposal", "RFP", "other"],
        help="ドキュメント種別でフィルタ"
    )
    parser.add_argument(
        "--show-stats",
        action="store_true",
        help="統計情報を表示"
    )

    args = parser.parse_args()

    console.print("[bold magenta]S3 Vectors クエリサンプル[/bold magenta]")
    console.print("=" * 60)

    # 環境変数チェック
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        console.print("[red]エラー: GEMINI_API_KEY が設定されていません[/red]")
        sys.exit(1)

    # 設定値の読み込み
    vector_bucket_name = os.getenv("VECTOR_BUCKET_NAME", "lisa-poc-vectors")
    vector_index_name = os.getenv("VECTOR_INDEX_NAME", "project-documents")
    aws_region = os.getenv("AWS_REGION", "us-west-2")
    embedding_model = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    dimension = int(os.getenv("DIMENSION", 1536))

    console.print("[dim]設定:[/dim]")
    console.print(f"[dim]  バケット: {vector_bucket_name}[/dim]")
    console.print(f"[dim]  インデックス: {vector_index_name}[/dim]")
    console.print(f"[dim]  リージョン: {aws_region}[/dim]")
    console.print(f"[dim]  埋め込みモデル: {embedding_model}[/dim]")
    console.print(f"[dim]  次元数: {dimension}[/dim]\n")

    try:
        # Embeddings初期化
        console.print("[cyan]1. 埋め込みモデル初期化中...[/cyan]")
        embeddings = GeminiEmbeddings(
            api_key=api_key,
            model_name=embedding_model,
            dimension=dimension
        )

        # Vector Store初期化
        console.print("[cyan]2. S3 Vector Store初期化中...[/cyan]")
        vector_store = S3VectorStore(
            vector_bucket_name=vector_bucket_name,
            index_name=vector_index_name,
            dimension=dimension,
            distance_metric="cosine",
            region_name=aws_region,
            create_if_not_exists=False  # 既存のインデックスを使用
        )

        # クエリベクトル生成
        console.print(f"[cyan]3. クエリをベクトル化中...[/cyan]")
        query_vector = embeddings.embed_text(args.query)

        # フィルタ条件作成
        filter_dict = create_filter_dict(args.project, args.doc_type)
        if filter_dict:
            console.print(f"[cyan]4. フィルタ条件:[/cyan] {json.dumps(filter_dict, ensure_ascii=False)}")

        # 類似検索実行
        console.print(f"[cyan]5. 類似検索実行中 (k={args.k})...[/cyan]")
        results = vector_store.similarity_search(
            query_vector=query_vector,
            k=args.k,
            filter_dict=filter_dict
        )

        # 結果表示
        display_results(results, args.query)

        # 統計情報表示
        if args.show_stats and results:
            console.print("\n[bold]統計情報:[/bold]")

            # スコア統計
            scores = [score for _, score in results]
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            min_score = min(scores)

            # プロジェクト別集計
            project_counts = {}
            doc_type_counts = {}
            importance_counts = {"high": 0, "medium": 0, "low": 0}

            for doc, _ in results:
                # プロジェクト
                proj = doc.metadata.get("project_name", "不明")
                project_counts[proj] = project_counts.get(proj, 0) + 1

                # ドキュメント種別
                dtype = doc.metadata.get("document_type", "不明")
                doc_type_counts[dtype] = doc_type_counts.get(dtype, 0) + 1

                # 重要度
                imp = doc.metadata.get("importance", "不明")
                if imp in importance_counts:
                    importance_counts[imp] += 1

            # テーブル作成
            table = Table(title="検索結果統計")
            table.add_column("項目", style="cyan")
            table.add_column("値", style="green")

            table.add_row("平均スコア", f"{avg_score:.3f}")
            table.add_row("最高スコア", f"{max_score:.3f}")
            table.add_row("最低スコア", f"{min_score:.3f}")
            table.add_row("", "")

            for proj, count in project_counts.items():
                table.add_row(f"プロジェクト: {proj}", str(count))

            table.add_row("", "")
            for dtype, count in doc_type_counts.items():
                table.add_row(f"種別: {dtype}", str(count))

            table.add_row("", "")
            for imp, count in importance_counts.items():
                table.add_row(f"重要度: {imp}", str(count))

            console.print(table)

    except Exception as e:
        console.print(f"[red]エラーが発生しました: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)

    console.print("\n[green]✅ クエリが正常に完了しました[/green]")


if __name__ == "__main__":
    main()
