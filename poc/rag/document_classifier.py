#!/usr/bin/env python3
"""
ドキュメント分類器

LLM（Gemini）を使用してドキュメントの種別を自動判定します。
業務フロー文書で定義されたカテゴリに基づいて分類を行います。
"""
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# 親ディレクトリをパスに追加（project_configをインポートするため）
sys.path.insert(0, str(Path(__file__).parent.parent))

from project_config import ProjectConfig
from utils.llm_response import extract_content as _extract_content


class DocumentClassifier:
    """LLMを使用してドキュメント種別を自動判定するクラス"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        project_config: Optional[ProjectConfig] = None
    ):
        """
        Args:
            api_key: Gemini APIキー（Noneの場合は環境変数から取得）
            model_name: 使用するモデル名（デフォルト: gemini-2.0-flash-exp）
            project_config: プロジェクト設定（Noneの場合は新規作成）
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY が設定されていません")

        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
        self.config = project_config or ProjectConfig()

        # カテゴリ情報を取得
        self.categories = self.config.get_classification_categories()
        self.category_descriptions = self.config.get_all_category_descriptions()

        if not self.categories:
            raise ValueError("カテゴリが設定されていません。project_config.yamlを確認してください。")

        # LLMモデル初期化
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=0.1,  # 分類タスクなので低温度で一貫性重視
            max_output_tokens=8192,  # カテゴリ名のみなので少なめ
        )

    def _build_classification_prompt(self, text_sample: str, file_name: str) -> str:
        """
        分類用のプロンプトを構築

        Args:
            text_sample: ドキュメントのテキストサンプル
            file_name: ファイル名

        Returns:
            LLMに送信するプロンプト
        """
        # カテゴリリストと説明を整形
        categories_text = ""
        for i, category in enumerate(self.categories, 1):
            desc = self.category_descriptions.get(category, "")
            categories_text += f"{i}. **{category}**: {desc}\n"

        prompt = f"""あなたはドキュメント分類の専門家です。
以下のドキュメントの種別を判定してください。

# 判定ルール
- 以下のカテゴリリストから**最も適切な1つ**を選択してください
- カテゴリ名のみを出力してください（説明や追加テキストは不要）
- どのカテゴリにも該当しない場合は「その他」を選択してください

# カテゴリリスト
{categories_text}

# ファイル名
{file_name}

# ドキュメント内容（先頭部分）
{text_sample}

# 判定結果（カテゴリ名のみ）:
"""
        return prompt

    def classify_document(
        self,
        text_sample: str,
        file_name: str,
        max_sample_length: int = 2000
    ) -> Tuple[str, float]:
        """
        ドキュメントを分類

        Args:
            text_sample: ドキュメントのテキスト（先頭部分で十分）
            file_name: ファイル名（分類のヒントとして使用）
            max_sample_length: サンプルテキストの最大長（デフォルト: 2000文字）

        Returns:
            (カテゴリ名, 信頼度) のタプル
            信頼度は現在1.0固定（将来的にLLMの応答から抽出可能）
        """
        # テキストサンプルを制限
        if len(text_sample) > max_sample_length:
            text_sample = text_sample[:max_sample_length] + "\n...(以下省略)"

        # プロンプト構築
        prompt = self._build_classification_prompt(text_sample, file_name)

        try:
            # LLMで分類
            message = HumanMessage(content=prompt)
            response = self.llm.invoke([message])

            # カテゴリ名を抽出（前後の空白を削除）
            category = _extract_content(response)

            # カテゴリが有効かチェック
            if category not in self.categories:
                # 部分一致で探す
                matched = False
                for valid_category in self.categories:
                    if valid_category in category or category in valid_category:
                        category = valid_category
                        matched = True
                        break

                if not matched:
                    print(f"    [WARN] 無効なカテゴリ '{category}' が返されました。'その他'に分類します。")
                    category = "その他"

            # ToDo: 信頼度は現在1.0固定（将来的に改善可能）
            confidence = 1.0

            return category, confidence

        except Exception as e:
            print(f"    [ERROR] ドキュメント分類エラー: {e}")
            # エラー時は「その他」に分類
            return "その他", 0.0

    def classify_from_elements(
        self,
        elements: list,
        file_name: str,
        max_elements: int = 10
    ) -> Tuple[str, float]:
        """
        unstructuredのElementsリストから分類

        Args:
            elements: unstructuredのElementsリスト
            file_name: ファイル名
            max_elements: サンプル抽出する最大Element数（デフォルト: 10）

        Returns:
            (カテゴリ名, 信頼度) のタプル
        """
        # Elementsからテキストを抽出
        text_parts = []
        for element in elements[:max_elements]:
            text = getattr(element, "text", "") or ""
            if text:
                text_parts.append(text)

        if not text_parts:
            print("    [WARN] テキストが抽出できませんでした。'その他'に分類します。")
            return "その他", 0.0

        # テキストサンプルを結合
        text_sample = "\n\n".join(text_parts)

        # 分類実行
        return self.classify_document(text_sample, file_name)

    def get_category_info(self, category: str) -> Dict[str, str]:
        """
        カテゴリの情報を取得

        Args:
            category: カテゴリ名

        Returns:
            カテゴリ情報の辞書（name, description）
        """
        return {
            "name": category,
            "description": self.category_descriptions.get(category, "")
        }


# テスト用のメイン関数
if __name__ == "__main__":
    import sys

    # 環境変数チェック
    if not os.getenv("GEMINI_API_KEY"):
        print("[ERROR] GEMINI_API_KEY が設定されていません")
        sys.exit(1)

    # 分類器初期化
    print("=== DocumentClassifier テスト ===\n")
    print("分類器を初期化中...")
    classifier = DocumentClassifier()
    print(f"モデル: {classifier.model_name}")
    print(f"カテゴリ数: {len(classifier.categories)}")
    print()

    # テストケース
    test_cases = [
        {
            "file_name": "顧客ヒアリングシート_2025年版.docx",
            "text": """
# ヒアリングシート

## 基本情報
顧客名:
担当者:
連絡先:

## システム要件
現在の課題:
希望する機能:
予算規模:

## スケジュール
希望導入時期:
            """
        },
        {
            "file_name": "提案書_MUJI_データ統合基盤.pdf",
            "text": """
株式会社良品計画 御中

データ統合基盤構築のご提案

1. 提案概要
貴社の課題である複数システムのデータサイロ化を解決するため、
統合データ基盤の構築を提案いたします。

2. ソリューション概要
- BigQueryを中核としたDWH構築
- dltによる自動ETLパイプライン
- リアルタイムダッシュボード

3. 実施体制
PM: 1名
SA: 2名
開発: 3名

4. スケジュール
要件定義: 1ヶ月
設計・開発: 3ヶ月
テスト・導入: 1ヶ月

5. 概算費用
合計: 1,200万円
            """
        },
        {
            "file_name": "基本設計書_顧客管理システム.docx",
            "text": """
顧客管理システム 基本設計書

1. システム概要
1.1 目的
顧客情報を一元管理し、営業活動の効率化を図る

2. システム構成
2.1 アーキテクチャ
- フロントエンド: React
- バックエンド: FastAPI
- データベース: PostgreSQL

3. 機能一覧
3.1 顧客管理機能
- 顧客情報登録
- 顧客情報検索
- 顧客情報更新

4. データモデル
4.1 顧客テーブル
- customer_id (PK)
- company_name
- contact_person
            """
        },
        {
            "file_name": "議事録_2025-01-15_週次定例MTG.md",
            "text": """
# 週次定例MTG 議事録

日時: 2025年1月15日 10:00-11:00
参加者: 山田(PM), 佐藤(SA), 鈴木(開発)

## 議題
1. 進捗報告
2. 課題の共有
3. 来週の予定

## 決定事項
- 基本設計書のレビュー完了
- 詳細設計を来週から開始
- テスト環境構築を並行で進める

## ToDo
- [山田] クライアントへ進捗報告
- [佐藤] 詳細設計書のドラフト作成
- [鈴木] テスト環境の要件整理
            """
        }
    ]

    # 各テストケースを分類
    for i, test_case in enumerate(test_cases, 1):
        print(f"--- テストケース {i} ---")
        print(f"ファイル名: {test_case['file_name']}")

        category, confidence = classifier.classify_document(
            text_sample=test_case["text"],
            file_name=test_case["file_name"]
        )

        print(f"判定結果: {category}")
        print(f"信頼度: {confidence}")

        # カテゴリ情報表示
        info = classifier.get_category_info(category)
        print(f"説明: {info['description']}")
        print()
