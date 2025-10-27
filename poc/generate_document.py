#!/usr/bin/env python3
"""
ドキュメント生成統合スクリプト

リフレクションノートまたはヒアリングシートから、
各種ドキュメント（ヒアリングシート、提案書など）を自動生成します。

使用例:
    # ヒアリングシート生成
    python generate_document.py hearing-sheet \
        --input reflection_note.md \
        --output hearing_sheet.md

    # 提案書生成
    python generate_document.py proposal \
        --input hearing_sheet.md \
        --output proposal.md

    # リフレクションノートから直接提案書生成
    python generate_document.py proposal \
        --input reflection_note.md \
        --output proposal.md
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# 親ディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

from rag.vector_store import S3VectorStore
from rag.embeddings import GeminiEmbeddings
from generators.hearing_sheet_generator import HearingSheetGenerator
from generators.proposal_generator import ProposalGenerator


def load_source_document(input_path: Optional[str] = None) -> str:
    """
    ソースドキュメントを読み込み

    Args:
        input_path: ファイルパス（Noneの場合は標準入力から読み込み）

    Returns:
        ソースドキュメントのテキスト
    """
    if input_path:
        with open(input_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        print("[INFO] 標準入力からソースドキュメントを読み込み中... (Ctrl+D で終了)")
        return sys.stdin.read()


def save_document(content: str, output_path: Optional[str] = None) -> None:
    """
    ドキュメントを保存

    Args:
        content: ドキュメントの内容
        output_path: 出力ファイルパス（Noneの場合は標準出力）
    """
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"\n[SUCCESS] ドキュメントを保存しました: {output_path}")
    else:
        print("\n" + "=" * 60)
        print("生成されたドキュメント")
        print("=" * 60 + "\n")
        print(content)


def generate_hearing_sheet(args, vector_store, embeddings):
    """ヒアリングシート生成"""
    print("[INFO] ヒアリングシート生成モード")

    # 生成器初期化
    print("\n[3/4] ヒアリングシート生成器を初期化中...")
    generator = HearingSheetGenerator(
        vector_store=vector_store,
        embeddings=embeddings,
        template_path=args.template
    )

    # プロジェクト情報のオーバーライド
    project_context = {}
    if args.project_name:
        project_context["project_name"] = args.project_name
    if args.customer_name:
        project_context["customer_name"] = args.customer_name
    if args.industry:
        project_context["industry"] = args.industry
    if args.scale:
        project_context["scale"] = args.scale
    if args.target_date:
        project_context["target_date"] = args.target_date

    # ヒアリングシート生成
    print("\n[4/4] ヒアリングシートを生成中...")
    return generator.generate(
        reflection_note=args.source_document,
        project_context=project_context if project_context else None,
        search_k=args.search_k
    )


def generate_proposal(args, vector_store, embeddings):
    """提案書生成"""
    print("[INFO] 提案書生成モード")

    # 生成器初期化
    print("\n[3/4] 提案書生成器を初期化中...")
    generator = ProposalGenerator(
        vector_store=vector_store,
        embeddings=embeddings,
        template_path=args.template
    )

    # プロジェクト情報のオーバーライド
    project_context = {}
    if args.project_name:
        project_context["project_name"] = args.project_name
    if args.customer_name:
        project_context["customer_name"] = args.customer_name

    # 提案書生成
    print("\n[4/4] 提案書を生成中...")
    return generator.generate(
        source_document=args.source_document,
        project_context=project_context if project_context else None,
        search_k=args.search_k
    )


def main():
    parser = argparse.ArgumentParser(
        description="リフレクションノート/ヒアリングシートから各種ドキュメントを生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # ヒアリングシート生成
  %(prog)s hearing-sheet --input reflection_note.md --output hearing_sheet.md

  # 提案書生成（ヒアリングシートから）
  %(prog)s proposal --input hearing_sheet.md --output proposal.md

  # 提案書生成（リフレクションノートから直接）
  %(prog)s proposal --input reflection_note.md --output proposal.md

  # プロジェクト情報を指定
  %(prog)s proposal --input reflection_note.md \\
      --project-name "ECサイトリニューアル" \\
      --customer-name "株式会社サンプル商事"
        """
    )

    # サブコマンド
    subparsers = parser.add_subparsers(
        dest='document_type',
        help='生成するドキュメントの種類',
        required=True
    )

    # ヒアリングシートサブコマンド
    hearing_parser = subparsers.add_parser(
        'hearing-sheet',
        help='ヒアリングシート生成',
        aliases=['hs']
    )

    # 提案書サブコマンド
    proposal_parser = subparsers.add_parser(
        'proposal',
        help='提案書生成',
        aliases=['prop']
    )

    # 共通オプション
    for subparser in [hearing_parser, proposal_parser]:
        # 入出力オプション
        subparser.add_argument(
            '--input', '-i',
            type=str,
            help='ソースドキュメントのファイルパス（省略時は標準入力）'
        )
        subparser.add_argument(
            '--output', '-o',
            type=str,
            help='出力ファイルパス（省略時は標準出力）'
        )

        # プロジェクト情報オプション
        subparser.add_argument(
            '--project-name',
            type=str,
            help='案件名（自動抽出される値を上書き）'
        )
        subparser.add_argument(
            '--customer-name',
            type=str,
            help='顧客名（自動抽出される値を上書き）'
        )

        # RAG検索オプション
        subparser.add_argument(
            '--search-k',
            type=int,
            default=5,
            help='類似案件の検索件数（デフォルト: 5）'
        )

        # テンプレートオプション
        subparser.add_argument(
            '--template',
            type=str,
            help='カスタムテンプレートファイルパス'
        )

    # ヒアリングシート固有のオプション
    hearing_parser.add_argument(
        '--industry',
        type=str,
        help='業界'
    )
    hearing_parser.add_argument(
        '--scale',
        type=str,
        help='案件規模'
    )
    hearing_parser.add_argument(
        '--target-date',
        type=str,
        help='希望導入時期'
    )

    args = parser.parse_args()

    # 環境変数チェック
    if not os.getenv("GEMINI_API_KEY"):
        print("[ERROR] GEMINI_API_KEY が設定されていません", file=sys.stderr)
        sys.exit(1)

    # S3 Vector Store設定（環境変数から取得）
    vector_bucket_name = os.getenv("VECTOR_BUCKET_NAME", "lisa-poc-vectors")
    vector_index_name = os.getenv("VECTOR_INDEX_NAME", "project-documents")
    aws_region = os.getenv("AWS_REGION", "us-west-2")
    dimension = int(os.getenv("DIMENSION", 1536))

    try:
        # 1. ソースドキュメント読み込み
        print("[1/4] ソースドキュメントを読み込み中...")
        source_document = load_source_document(args.input)

        if not source_document.strip():
            print("[ERROR] ソースドキュメントが空です", file=sys.stderr)
            sys.exit(1)

        print(f"[INFO] {len(source_document)} 文字のドキュメントを読み込みました")

        # ソースドキュメントをargsに追加（生成関数で使用）
        args.source_document = source_document

        # 2. RAGシステム初期化
        print("\n[2/4] RAGシステムを初期化中...")

        embeddings = GeminiEmbeddings(
            api_key=os.getenv("GEMINI_API_KEY"),
            model_name=os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
        )

        vector_store = S3VectorStore(
            vector_bucket_name=vector_bucket_name,
            index_name=vector_index_name,
            dimension=dimension,
            distance_metric="cosine",
            region_name=aws_region,
            create_if_not_exists=False
        )

        # 3-4. ドキュメント生成（サブコマンドに応じて分岐）
        if args.document_type in ['hearing-sheet', 'hs']:
            document = generate_hearing_sheet(args, vector_store, embeddings)
        elif args.document_type in ['proposal', 'prop']:
            document = generate_proposal(args, vector_store, embeddings)
        else:
            print(f"[ERROR] 未知のドキュメントタイプ: {args.document_type}", file=sys.stderr)
            sys.exit(1)

        # 5. 結果を保存
        save_document(document, args.output)

        print("\n[SUCCESS] ドキュメント生成が完了しました！")

    except FileNotFoundError as e:
        print(f"[ERROR] ファイルが見つかりません: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] エラーが発生しました: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
