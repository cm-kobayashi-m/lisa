#!/usr/bin/env python3
"""
S3 Vectorsのバケットとインデックスを削除するスクリプト

使用方法:
    # インデックスのみ削除
    python3 delete_s3_vectors.py --index-only

    # バケットとインデックスを削除
    python3 delete_s3_vectors.py

    # 特定のバケット/インデックスを削除
    python3 delete_s3_vectors.py --bucket my-bucket --index my-index

    # Dry-runモード（削除せず確認のみ）
    python3 delete_s3_vectors.py --dry-run

警告: このスクリプトはデータを永久に削除します。実行前に必ずバックアップを取ってください。
"""
import os
import sys
import argparse
import logging
from dotenv import load_dotenv
from rag.vector_store import S3VectorStore

# 環境変数読み込み
load_dotenv()

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def delete_s3_vectors(
    bucket_name: str,
    index_name: str,
    region: str,
    index_only: bool = False,
    dry_run: bool = False
):
    """S3 Vectorsのリソースを削除"""

    if dry_run:
        print("\n" + "=" * 60)
        print("DRY-RUN モード: 実際の削除は行いません")
        print("=" * 60)
        print(f"削除対象:")
        print(f"  - バケット: {bucket_name}")
        print(f"  - インデックス: {index_name}")
        print(f"  - リージョン: {region}")
        print(f"  - インデックスのみ: {index_only}")
        print()

        if not index_only:
            print("⚠️ 警告: バケット削除により、すべてのデータが永久に失われます")

        print("\n実際に削除する場合は --dry-run オプションを外して実行してください")
        return

    # 確認プロンプト
    print("\n" + "=" * 60)
    print("⚠️ 警告: データの永久削除")
    print("=" * 60)
    print(f"削除対象:")
    print(f"  - バケット: {bucket_name}")
    print(f"  - インデックス: {index_name}")
    print(f"  - リージョン: {region}")
    print(f"  - インデックスのみ: {index_only}")
    print()

    if not index_only:
        print("🚨 バケット削除により、すべてのデータが永久に失われます 🚨")

    confirmation = input("\n本当に削除しますか？ 'yes' と入力してください: ")
    if confirmation.lower() != 'yes':
        print("削除をキャンセルしました")
        return

    try:
        # S3VectorStoreインスタンスを作成（自動作成は無効）
        print(f"\nS3 Vectorsクライアントを初期化中...")
        vector_store = S3VectorStore(
            vector_bucket_name=bucket_name,
            index_name=index_name,
            region_name=region,
            create_if_not_exists=False  # 自動作成を無効化
        )

        if index_only:
            # インデックスのみ削除
            print(f"\nインデックス '{index_name}' を削除中...")
            vector_store.delete_index()
            print(f"✅ インデックス '{index_name}' を削除しました")

        else:
            # バケットとインデックスを削除
            print(f"\nバケット '{bucket_name}' とインデックス '{index_name}' を削除中...")
            vector_store.delete_bucket()  # このメソッドは内部でインデックスも削除
            print(f"✅ バケット '{bucket_name}' とインデックス '{index_name}' を削除しました")

    except Exception as e:
        logger.error(f"削除中にエラーが発生しました: {e}")
        print(f"\n❌ エラー: {e}")
        print("\n考えられる原因:")
        print("  1. リソースが既に削除されている")
        print("  2. 権限が不足している")
        print("  3. ネットワークエラー")
        print("\n詳細はログを確認してください")
        sys.exit(1)

    print("\n削除が完了しました")


def main():
    parser = argparse.ArgumentParser(
        description='S3 Vectorsのバケットとインデックスを削除'
    )
    parser.add_argument(
        '--bucket',
        type=str,
        default=os.getenv('VECTOR_BUCKET_NAME', 'lisa-poc-vectors'),
        help='S3 Vectorsバケット名（デフォルト: lisa-poc-vectors）'
    )
    parser.add_argument(
        '--index',
        type=str,
        default=os.getenv('VECTOR_INDEX_NAME', 'project-documents'),
        help='インデックス名（デフォルト: project-documents）'
    )
    parser.add_argument(
        '--region',
        type=str,
        default=os.getenv('AWS_REGION', 'us-west-2'),
        help='AWSリージョン（デフォルト: us-west-2）'
    )
    parser.add_argument(
        '--index-only',
        action='store_true',
        help='インデックスのみ削除（バケットは残す）'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='削除対象を表示するが実行しない'
    )

    args = parser.parse_args()

    delete_s3_vectors(
        bucket_name=args.bucket,
        index_name=args.index,
        region=args.region,
        index_only=args.index_only,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()