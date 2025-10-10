#!/usr/bin/env python3
"""
RAG統合機能のテストスクリプト

generate_note.pyのRAG機能が正しく動作するかをテストします。
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 環境変数読み込み
load_dotenv()


def test_rag_retriever():
    """RAGRetrieverクラスの動作テスト"""
    print("=" * 60)
    print("RAGRetrieverクラスのテスト")
    print("=" * 60)

    try:
        from rag.rag_retriever import RAGRetriever
        from rag.vector_store import S3VectorStore
        from rag.embeddings import GeminiEmbeddings

        # 初期化
        print("[TEST] RAGコンポーネントの初期化...")
        embeddings = GeminiEmbeddings(api_key=os.getenv('GEMINI_API_KEY'))

        vector_store = S3VectorStore(
            vector_bucket_name=os.getenv('VECTOR_BUCKET_NAME', 'lisa-poc-vectors'),
            index_name=os.getenv('VECTOR_INDEX_NAME', 'project-documents'),
            dimension=768,
            region_name=os.getenv('AWS_REGION', 'us-west-2'),
            create_if_not_exists=False
        )

        retriever = RAGRetriever(vector_store, embeddings)
        print("[TEST] ✓ 初期化成功")

        # 類似検索テスト
        print("\n[TEST] 類似度検索のテスト...")
        test_query = "プロジェクト計画 要件定義"
        results = retriever.search_similar_documents(
            query=test_query,
            k=3
        )

        if results:
            print(f"[TEST] ✓ {len(results)}件の結果を取得")
            for i, (doc, score) in enumerate(results, 1):
                print(f"  {i}. 類似度: {(1+score)/2*100:.1f}%")
                print(f"     プロジェクト: {doc.metadata.get('project_name', '不明')}")
                print(f"     ファイル: {doc.metadata.get('file_name', '不明')}")
        else:
            print("[TEST] 検索結果なし（データが存在しない可能性があります）")

        # フォーマットテスト
        print("\n[TEST] 結果フォーマットのテスト...")
        if results:
            formatted = retriever.format_context_for_prompt(results, max_chars=1000)
            print(f"[TEST] ✓ フォーマット済みテキスト（{len(formatted)}文字）")
            print(formatted[:500] + "..." if len(formatted) > 500 else formatted)

        print("\n[TEST] RAGRetrieverテスト完了")
        return True

    except Exception as e:
        print(f"[ERROR] RAGRetrieverテストでエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_analyze_file_with_rag():
    """analyze_file_with_gemini関数のRAG統合テスト"""
    print("\n" + "=" * 60)
    print("analyze_file_with_geminiのRAG統合テスト")
    print("=" * 60)

    try:
        from generate_note import analyze_file_with_gemini, initialize_gemini_client

        # Geminiクライアント初期化
        print("[TEST] Geminiクライアントの初期化...")
        client = initialize_gemini_client()
        print("[TEST] ✓ Geminiクライアント初期化成功")

        # テキストでの分析（RAG有効）
        print("\n[TEST] RAG有効でテキスト分析...")
        test_text = """
        プロジェクト概要：
        このプロジェクトは、データ統合基盤の構築を目的としています。
        SalesforceとBigQueryを連携させ、リアルタイムでデータ分析を可能にします。
        """

        result = analyze_file_with_gemini(
            client=client,
            file_path=None,
            file_name="test_document.txt",
            mime_type=None,
            text_content=test_text,
            use_rag=True,
            project_name="テストプロジェクト"
        )

        if result:
            print("[TEST] ✓ 分析成功")
            print(f"[TEST] 分析結果（最初の500文字）:")
            print(result[:500] + "..." if len(result) > 500 else result)
        else:
            print("[TEST] 分析結果が空です")

        print("\n[TEST] analyze_file_with_geminiテスト完了")
        return True

    except Exception as e:
        print(f"[ERROR] analyze_file_with_geminiテストでエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generate_final_reflection_with_rag():
    """generate_final_reflection_note関数のRAG統合テスト"""
    print("\n" + "=" * 60)
    print("generate_final_reflection_noteのRAG統合テスト")
    print("=" * 60)

    try:
        from generate_note import generate_final_reflection_note, initialize_gemini_client

        # Geminiクライアント初期化
        print("[TEST] Geminiクライアントの初期化...")
        client = initialize_gemini_client()
        print("[TEST] ✓ Geminiクライアント初期化成功")

        # テスト用のファイルサマリ
        test_summaries = [
            {
                'file_name': 'requirements.txt',
                'analysis': 'このドキュメントはシステム要件を定義しています。主な機能として、ユーザー認証、データ同期、レポート生成が含まれます。'
            },
            {
                'file_name': 'architecture.md',
                'analysis': 'システムアーキテクチャはマイクロサービス構成で、API Gateway、認証サービス、データ処理サービスから構成されます。'
            }
        ]

        print("\n[TEST] RAG有効でリフレクションノート生成...")
        result, summaries_text = generate_final_reflection_note(
            client=client,
            project_name="テストプロジェクト",
            file_summaries=test_summaries,
            use_rag=True
        )

        if result:
            print("[TEST] ✓ リフレクションノート生成成功")
            print(f"[TEST] ノート長さ: {len(result)}文字")

            # RAGコンテキストが含まれているか確認
            if "RAG" in result or "関連情報" in result or "類似プロジェクト" in summaries_text:
                print("[TEST] ✓ RAGコンテキストが含まれています")
            else:
                print("[TEST] △ RAGコンテキストが見つかりません（データがない可能性）")
        else:
            print("[TEST] リフレクションノートが空です")

        print("\n[TEST] generate_final_reflection_noteテスト完了")
        return True

    except Exception as e:
        print(f"[ERROR] generate_final_reflection_noteテストでエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_environment():
    """環境設定の確認"""
    print("=" * 60)
    print("環境設定の確認")
    print("=" * 60)

    required_vars = [
        'GEMINI_API_KEY',
        'VECTOR_BUCKET_NAME',
        'VECTOR_INDEX_NAME',
        'AWS_REGION'
    ]

    optional_vars = [
        'USE_RAG',
        'AWS_PROFILE',
        'GEMINI_MODEL'
    ]

    all_ok = True

    print("必須環境変数:")
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # APIキーは一部マスク
            if 'KEY' in var or 'SECRET' in var:
                display = value[:8] + "..." if len(value) > 8 else "***"
            else:
                display = value
            print(f"  ✓ {var}: {display}")
        else:
            print(f"  ✗ {var}: 未設定")
            all_ok = False

    print("\nオプション環境変数:")
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            print(f"  ✓ {var}: {value}")
        else:
            print(f"  - {var}: 未設定（デフォルト値使用）")

    print("\nRAG機能状態:")
    use_rag = os.getenv('USE_RAG', 'true').lower() == 'true'
    print(f"  RAG機能: {'有効' if use_rag else '無効'}")

    return all_ok


def main():
    """メインテスト実行"""
    print("RAG統合機能テスト")
    print("=" * 60)
    print()

    # 環境確認
    if not check_environment():
        print("\n[ERROR] 必須環境変数が設定されていません。.envファイルを確認してください。")
        sys.exit(1)

    # 各テスト実行
    results = []

    # RAGRetrieverテスト
    print("\n" + "-" * 60)
    result = test_rag_retriever()
    results.append(("RAGRetriever", result))

    # analyze_file_with_geminiテスト
    print("\n" + "-" * 60)
    result = test_analyze_file_with_rag()
    results.append(("analyze_file_with_gemini", result))

    # generate_final_reflection_noteテスト
    print("\n" + "-" * 60)
    result = test_generate_final_reflection_with_rag()
    results.append(("generate_final_reflection_note", result))

    # 結果サマリ
    print("\n" + "=" * 60)
    print("テスト結果サマリ")
    print("=" * 60)

    for test_name, success in results:
        status = "✓ 成功" if success else "✗ 失敗"
        print(f"{test_name}: {status}")

    all_success = all(r[1] for r in results)

    if all_success:
        print("\n🎉 すべてのテストに成功しました！")
        print("RAG統合機能は正常に動作しています。")
    else:
        print("\n⚠️ 一部のテストが失敗しました。")
        print("エラーログを確認して問題を修正してください。")

    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())