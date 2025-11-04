#!/usr/bin/env python3
"""
ドキュメント生成統合スクリプト

リフレクションノートまたはヒアリングシートから、
各種ドキュメント（ヒアリングシート、提案書など）を自動生成します。

出力ファイルは入力ファイルと同じディレクトリに以下の2つが作成されます：
- タイムスタンプ付きファイル（例: 20251031_123456_hearing_sheet.md）
- 最新版ファイル（例: hearing_sheet_latest.md）

使用例:
    # ヒアリングシート生成（入力ファイルと同じディレクトリに自動保存）
    python generate_document.py hearing-sheet \
        --input reflection_note.md

    # 提案書生成
    python generate_document.py proposal \
        --input hearing_sheet.md

    # 標準入力から読み込み
    python generate_document.py proposal --input -
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# 親ディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

from rag.vector_store import S3VectorStore
from rag.embeddings import GeminiEmbeddings
from generators.hearing_sheet_generator import HearingSheetGenerator
from generators.proposal_generator import ProposalGenerator

# CRAG機能のインポート（オプション）
from rag.enhanced_rag_search import (
        create_enhanced_rag_search,
        EnhancedRAGConfig
    )


def load_source_document(input_path: str) -> str:
    """
    ソースドキュメントを読み込み

    Args:
        input_path: ファイルパス（"-"の場合は標準入力から読み込み）

    Returns:
        ソースドキュメントのテキスト
    """
    if input_path == "-":
        print("[INFO] 標準入力からソースドキュメントを読み込み中... (Ctrl+D で終了)")
        return sys.stdin.read()
    else:
        with open(input_path, 'r', encoding='utf-8') as f:
            return f.read()


def generate_output_path(input_path: str, document_type: str) -> tuple[Path, str]:
    """
    入力ファイルパスからドキュメント種別に応じた出力パスを生成

    Args:
        input_path: 入力ファイルパス
        document_type: 'hearing-sheet' または 'proposal'

    Returns:
        (タイムスタンプ付きファイルパス, _latestファイル名) のタプル
    """
    input_file = Path(input_path) if input_path != "-" else Path.cwd()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if document_type in ['hearing-sheet', 'hs']:
        filename = f"{timestamp}_hearing_sheet.md"
        latest_name = "hearing_sheet_latest.md"
    elif document_type in ['proposal', 'prop']:
        filename = f"{timestamp}_proposal.md"
        latest_name = "proposal_latest.md"
    else:
        raise ValueError(f"Unknown document type: {document_type}")

    return input_file.parent / filename, latest_name


def save_document(content: str, output_path: str, latest_name: str) -> None:
    """
    ドキュメントを保存（日付付き + _latest の2つ）

    Args:
        content: ドキュメントの内容
        output_path: タイムスタンプ付き出力ファイルパス
        latest_name: _latestファイルの名前（例: "hearing_sheet_latest.md"）
    """
    # 日付付きファイルに保存
    output_file = Path(output_path)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"\n[SUCCESS] ドキュメントを保存しました: {output_path}")

    # _latest.mdファイルも作成（種別ベースの固定名）
    latest_file = output_file.parent / latest_name
    with open(latest_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"[SUCCESS] 最新版も保存しました: {latest_file}")


def generate_hearing_sheet(args, vector_store, embeddings, enable_crag=False):
    """ヒアリングシート生成（Query Translation対応版）"""
    print("[INFO] ヒアリングシート生成モード")

    # CRAG機能の状態を表示
    if enable_crag:
        print("[INFO] CRAG機能: 有効")
    else:
        print("[INFO] CRAG機能: 無効")

    # 追加プロンプト情報の表示
    if hasattr(args, 'additional_prompt') and args.additional_prompt:
        print(f"[INFO] 追加指示: {args.additional_prompt}")

    # 生成器初期化
    print("\n[3/4] ヒアリングシート生成器を初期化中...")
    generator = HearingSheetGenerator(
        vector_store=vector_store,
        embeddings=embeddings,
        template_path=args.template,
        enable_crag=enable_crag  # CRAG有効化フラグを渡す
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

    # ヒアリングシート生成（追加プロンプトを渡す）
    print("\n[4/4] ヒアリングシートを生成中...")
    return generator.generate(
        reflection_note=args.source_document,
        project_context=project_context if project_context else None,
        search_k=args.search_k,
        additional_prompt=getattr(args, 'additional_prompt', None)  # 追加プロンプトを渡す
    )


def generate_proposal(args, vector_store, embeddings, enable_crag=False):
    """提案書生成（Query Translation対応版）"""
    print("[INFO] 提案書生成モード")

    # CRAG機能の状態を表示
    if enable_crag:
        print("[INFO] CRAG機能: 有効")
    else:
        print("[INFO] CRAG機能: 無効")

    # 追加プロンプト情報の表示
    if hasattr(args, 'additional_prompt') and args.additional_prompt:
        print(f"[INFO] 追加指示: {args.additional_prompt}")

    # 生成器初期化
    print("\n[3/4] 提案書生成器を初期化中...")
    generator = ProposalGenerator(
        vector_store=vector_store,
        embeddings=embeddings,
        template_path=args.template,
        enable_crag=enable_crag  # CRAG有効化フラグを渡す
    )

    # プロジェクト情報のオーバーライド
    project_context = {}
    if args.project_name:
        project_context["project_name"] = args.project_name
    if args.customer_name:
        project_context["customer_name"] = args.customer_name

    # 提案書生成（追加プロンプトを渡す）
    print("\n[4/4] 提案書を生成中...")
    return generator.generate(
        source_document=args.source_document,
        project_context=project_context if project_context else None,
        search_k=args.search_k,
        additional_prompt=getattr(args, 'additional_prompt', None)  # 追加プロンプトを渡す
    )


def main():
    parser = argparse.ArgumentParser(
        description="リフレクションノート/ヒアリングシートから各種ドキュメントを生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # ヒアリングシート生成（入力ファイルと同じディレクトリに自動保存）
  %(prog)s hearing-sheet --input reflection_note.md

  # 提案書生成
  %(prog)s proposal --input hearing_sheet.md

  # 標準入力から読み込み
  cat reflection_note.md | %(prog)s proposal --input -

  # プロジェクト情報を指定
  %(prog)s proposal --input reflection_note.md \\
      --project-name "ECサイトリニューアル" \\
      --customer-name "株式会社サンプル商事"

出力ファイル:
  入力ファイルと同じディレクトリに以下の2つが作成されます：
  - タイムスタンプ付きファイル: {yyyymmdd_HHMMSS}_{type}.md
  - 最新版ファイル: {type}_latest.md
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
            required=True,
            help='ソースドキュメントのファイルパス（必須、"-"で標準入力）'
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

        # Query Translation用の追加プロンプト
        subparser.add_argument(
            '--additional-prompt',
            type=str,
            help='追加の指示やコンテキスト（例: "ヤーマン案件を参考に、期限が厳しいので精度重視で"）'
        )

        # CRAG機能オプション
        subparser.add_argument(
            '--enable-crag',
            action='store_true',
            help='CRAG機能を有効にする（関連性評価とKnowledge Refinement）'
        )
        subparser.add_argument(
            '--disable-crag',
            action='store_true',
            help='CRAG機能を無効にする'
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

    # CRAG有効/無効の決定
    enable_crag = False
    if hasattr(args, 'enable_crag') and args.enable_crag:
        enable_crag = True
    elif hasattr(args, 'disable_crag') and args.disable_crag:
        enable_crag = False
    else:
        # 環境変数から取得
        enable_crag = os.getenv('ENABLE_CRAG', 'false').lower() == 'true'

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
            model_name=os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
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
            document = generate_hearing_sheet(args, vector_store, embeddings, enable_crag)
        elif args.document_type in ['proposal', 'prop']:
            document = generate_proposal(args, vector_store, embeddings, enable_crag)
        else:
            print(f"[ERROR] 未知のドキュメントタイプ: {args.document_type}", file=sys.stderr)
            sys.exit(1)

        # 5. 出力パスを生成
        output_path, latest_name = generate_output_path(args.input, args.document_type)

        # 6. 結果を保存
        save_document(document, str(output_path), latest_name)

        print("\n[SUCCESS] ドキュメント生成が完了しました！")

    except FileNotFoundError as e:
        print(f"[ERROR] ファイルが見つかりません: {e}", file=sys.stderr)
        print(f"[INFO] カレントディレクトリ: {os.getcwd()}", file=sys.stderr)
        sys.exit(1)
    except PermissionError as e:
        print(f"[ERROR] ファイルへのアクセス権限がありません: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] エラーが発生しました: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
