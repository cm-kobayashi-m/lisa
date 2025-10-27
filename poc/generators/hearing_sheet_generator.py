#!/usr/bin/env python3
"""
ヒアリングシート生成器

リフレクションノートから、過去の類似案件を参照しつつ
ヒアリングシートを自動生成します。

アプローチ: ハイブリッド型（テンプレート + RAG）
1. テンプレートベースの構造生成
2. RAGで類似案件のヒアリングシートを検索
3. LLMで統合・カスタマイズ
"""
import os
import sys
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from google import genai

# 親ディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.vector_store import S3VectorStore, Document
from rag.embeddings import GeminiEmbeddings
from rag.rag_retriever import RAGRetriever
from rag.rag_fusion import rag_fusion_search


class HearingSheetGenerator:
    """リフレクションノート → ヒアリングシート生成"""

    def __init__(
        self,
        vector_store: S3VectorStore,
        embeddings: GeminiEmbeddings,
        llm: Optional[ChatGoogleGenerativeAI] = None,
        template_path: Optional[str] = None
    ):
        """
        Args:
            vector_store: S3VectorStoreインスタンス
            embeddings: GeminiEmbeddingsインスタンス
            llm: LLMインスタンス（Noneの場合は自動生成）
            template_path: テンプレートファイルパス（Noneの場合はデフォルト使用）
        """
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.retriever = RAGRetriever(vector_store, embeddings)

        # API KEY取得
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY が設定されていません")

        # LLM初期化
        if llm:
            self.llm = llm
        else:
            model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.3,  # 適度な創造性
                max_output_tokens=8192
            )

        # Gemini APIクライアント初期化（RAG-Fusion用）
        self.gemini_client = genai.Client(api_key=api_key)

        # テンプレート読み込み
        if template_path:
            self.template_path = Path(template_path)
        else:
            self.template_path = Path(__file__).parent / "templates" / "hearing_sheet.md"

        self.template = self._load_template()

    def _load_template(self) -> str:
        """テンプレートファイルを読み込み"""
        if not self.template_path.exists():
            raise FileNotFoundError(f"テンプレートが見つかりません: {self.template_path}")

        with open(self.template_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _extract_project_info(self, reflection_note: str) -> Dict[str, str]:
        """
        リフレクションノートから基本情報を抽出

        Args:
            reflection_note: リフレクションノート

        Returns:
            抽出された情報の辞書
        """
        extraction_prompt = f"""
以下のリフレクションノートから、案件の基本情報を抽出してください。

# リフレクションノート
{reflection_note}

# 抽出する情報
以下のJSON形式で出力してください。情報が見つからない場合は空文字列""を設定してください。

{{
  "project_name": "案件名",
  "customer_name": "顧客名",
  "industry": "業界",
  "scale": "案件規模（例: 小規模/中規模/大規模、または人月数）",
  "target_date": "希望導入時期",
  "background": "背景・課題の要約（3-5行程度）"
}}

JSON形式のみを出力してください（説明や追加テキストは不要）。
"""
        try:
            response = self.llm.invoke(extraction_prompt)
            import json
            # レスポンスからJSONを抽出（```json ``` の除去）
            content = response.content.strip()
            if content.startswith("```"):
                # コードブロックを除去
                lines = content.split("\n")
                content = "\n".join(lines[1:-1])

            project_info = json.loads(content)
            return project_info
        except Exception as e:
            print(f"    [WARN] 基本情報抽出エラー: {e}")
            # フォールバック: 空の辞書
            return {
                "project_name": "",
                "customer_name": "",
                "industry": "",
                "scale": "",
                "target_date": "",
                "background": ""
            }

    def _search_similar_hearing_sheets(
        self,
        reflection_note: str,
        project_name: str = "",
        k: int = 5
    ) -> List[Tuple[Document, float]]:
        """
        類似案件のヒアリングシートを検索

        Args:
            reflection_note: リフレクションノート
            project_name: プロジェクト名（RAG-Fusion用）
            k: 検索する件数

        Returns:
            検索結果のリスト
        """
        # RAG-Fusion有効化フラグ
        use_rag_fusion = os.getenv('USE_RAG_FUSION', 'true').lower() == 'true'

        if use_rag_fusion and project_name:
            print(f"    [RAG-Fusion] ヒアリングシート検索中（k={k}）...")

            # RAG-Fusionで検索
            results = rag_fusion_search(
                client=self.gemini_client,
                retriever=self.retriever,
                project_name=project_name,
                base_query=f"ヒアリングシート {reflection_note[:300]}",
                k=k,
                num_queries=int(os.getenv('RAG_FUSION_NUM_QUERIES', '3')),
                min_score=float(os.getenv('RAG_ONLY_MODE_MIN_SCORE', '0.3')),
                apply_time_weighting=True
            )

            print(f"    [RAG-Fusion] {len(results)}件のヒアリングシートを発見")
        else:
            print(f"    [従来検索] ヒアリングシート検索中（k={k}）...")

            # 従来の検索
            query = reflection_note[:500]

            # カテゴリフィルタ付き検索
            results = self.retriever.search_by_category(
                query=query,
                category="ヒアリングシート",
                k=k
            )

            print(f"    [従来検索] {len(results)}件のヒアリングシートを発見")

        return results

    def _format_similar_cases(
        self,
        results: List[Tuple[Document, float]]
    ) -> str:
        """
        類似案件をフォーマット

        Args:
            results: 検索結果

        Returns:
            フォーマット済みテキスト
        """
        if not results:
            return """### 類似案件からの参考情報

**参考案件データなし**

現時点で類似案件のヒアリングシートデータが登録されていません。
以下の点に注意してヒアリングを進めてください：

- 業界特有の商習慣や技術的制約を確認
- 顧客の期待値と予算のバランスを早期に把握
- リスク要因の洗い出しを徹底的に実施
- 過去の類似プロジェクトの知見を社内で共有

**※ 本ヒアリングシート完了後は、今後の参考のためにVector DBへの登録を推奨します**
"""

        formatted = "### 類似案件からの参考情報\n\n"

        for i, (doc, score) in enumerate(results, 1):
            similarity = max(0.0, min(1.0, 1.0 - score))  # cosine距離→類似度
            project = doc.metadata.get("project_name", "不明")
            file_name = doc.metadata.get("file_name", "不明")

            formatted += f"#### {i}. {project}（類似度: {similarity*100:.1f}%）\n"
            formatted += f"- ファイル: {file_name}\n"

            # 内容の抜粋（長すぎる場合は省略）
            content = doc.text[:300]
            if len(doc.text) > 300:
                content += "..."

            formatted += f"```\n{content}\n```\n\n"

        return formatted

    def _assess_risks(self, reflection_note: str) -> str:
        """
        リフレクションノートからリスク評価

        Args:
            reflection_note: リフレクションノート

        Returns:
            リスク評価テキスト
        """
        risk_prompt = f"""
以下のリフレクションノートから、プロジェクトのリスクを評価してください。

# リフレクションノート
{reflection_note}

# 評価基準（業務フロー文書より）
以下の項目について、リスクの有無を判定してください：

1. 体制規模: 12人月を超えるか？
2. 契約形態: 請負契約が要求されているか？
3. 技術的実現性: 未経験の技術や不透明な要件があるか？
4. 運用要件: 24/365運用が必要か？
5. システム連携: 既存システムとの複雑な連携が必要か？
6. 顧客体制: 意思決定プロセスが不明確か？

# 出力形式
各項目について、以下の形式で出力してください：

- **項目名**: リスク有無（⚠️高リスク / ⚡中リスク / ✅低リスク）
  - 理由: （簡潔に）

最後に総合評価を1-2行で記載してください。
"""
        try:
            response = self.llm.invoke(risk_prompt)
            return response.content.strip()
        except Exception as e:
            print(f"    [WARN] リスク評価エラー: {e}")
            return "（リスク評価を実行できませんでした）"

    def _generate_additional_questions(
        self,
        reflection_note: str,
        similar_cases: List[Tuple[Document, float]]
    ) -> str:
        """
        追加質問項目を生成

        Args:
            reflection_note: リフレクションノート
            similar_cases: 類似案件

        Returns:
            追加質問項目
        """
        # 類似案件からヒントを抽出
        similar_hints = ""
        if similar_cases:
            similar_hints = "# 類似案件で確認されていた項目\n\n"
            for doc, _ in similar_cases[:3]:
                similar_hints += f"- {doc.metadata.get('project_name', '不明')}: {doc.text[:200]}...\n\n"

        question_prompt = f"""
以下のリフレクションノートから、提案に向けて追加で確認が必要な事項を洗い出してください。

# リフレクションノート
{reflection_note}

{similar_hints}

# 出力形式
重要度順に3-5個の質問項目を箇条書きで出力してください。
各質問は具体的で、回答によって提案内容が変わるものにしてください。

例:
- データ量の具体的な規模は？（行数、容量、増加率）
- 既存システムのAPI仕様は公開されているか？
- セキュリティ監査の基準は？（ISO27001等）
"""
        try:
            response = self.llm.invoke(question_prompt)
            return response.content.strip()
        except Exception as e:
            print(f"    [WARN] 追加質問生成エラー: {e}")
            return "（追加質問を生成できませんでした）"

    def generate(
        self,
        reflection_note: str,
        project_context: Optional[Dict[str, str]] = None,
        search_k: int = 5
    ) -> str:
        """
        ヒアリングシートを生成

        Args:
            reflection_note: リフレクションノート
            project_context: 追加のプロジェクト情報（辞書）
            search_k: RAG検索件数

        Returns:
            生成されたヒアリングシート（Markdown形式）
        """
        print("=" * 60)
        print("ヒアリングシート生成開始")
        print("=" * 60)

        # 1. 基本情報抽出
        print("\n[1/5] リフレクションノートから基本情報を抽出中...")
        project_info = self._extract_project_info(reflection_note)

        # project_contextでオーバーライド
        if project_context:
            project_info.update(project_context)

        # 2. 類似案件検索
        print("\n[2/5] 類似案件のヒアリングシートを検索中...")
        similar_cases = self._search_similar_hearing_sheets(
            reflection_note,
            project_name=project_info.get("project_name", ""),
            k=search_k
        )
        similar_cases_text = self._format_similar_cases(similar_cases)

        # 3. リスク評価
        print("\n[3/5] プロジェクトリスクを評価中...")
        risk_assessment = self._assess_risks(reflection_note)

        # 4. 追加質問生成
        print("\n[4/5] 追加確認事項を生成中...")
        additional_questions = self._generate_additional_questions(
            reflection_note, similar_cases
        )

        # 5. テンプレート埋め込み
        print("\n[5/5] ヒアリングシートを生成中...")
        hearing_sheet = self.template.format(
            project_name=project_info.get("project_name", ""),
            customer_name=project_info.get("customer_name", ""),
            industry=project_info.get("industry", ""),
            scale=project_info.get("scale", ""),
            target_date=project_info.get("target_date", ""),
            background=project_info.get("background", ""),
            risk_assessment=risk_assessment,
            similar_cases=similar_cases_text,
            additional_questions=additional_questions,
            notes="",
            created_date=datetime.now().strftime("%Y年%m月%d日"),
            creator="LISA AI",
            updated_date=datetime.now().strftime("%Y年%m月%d日")
        )

        print("\n" + "=" * 60)
        print("ヒアリングシート生成完了")
        print("=" * 60)

        return hearing_sheet


# テスト用のメイン関数
if __name__ == "__main__":
    # サンプルのリフレクションノート
    sample_reflection_note = """
# リフレクションノート

## 案件概要
- 顧客: 株式会社サンプル商事
- 業界: 小売業
- 案件名: ECサイトリニューアル

## 背景
現在のECサイトは10年前のシステムで、スマホ対応が不十分。
顧客体験の向上とコンバージョン率改善が急務。

## 主な要件
- モバイルファースト対応
- 既存の基幹システム（AS400）との連携
- 在庫情報のリアルタイム表示
- 会員データ移行（約50万件）

## 懸念事項
- 基幹システムのAPI仕様が不明
- 本番稼働は3ヶ月後を希望（短納期）
- セキュリティ監査が必須
"""

    print("=== HearingSheetGenerator テスト ===\n")

    # 環境変数チェック
    if not os.getenv("GEMINI_API_KEY"):
        print("[ERROR] GEMINI_API_KEY が設定されていません")
        sys.exit(1)

    # 注意: 実際のテストにはS3VectorStoreとGeminiEmbeddingsの初期化が必要
    print("[INFO] 実際のテストには、S3VectorStoreとEmbeddingsの設定が必要です")
    print("[INFO] 以下はダミーの出力例です\n")

    # ダミー出力
    print("--- 生成されるヒアリングシートのイメージ ---")
    print(sample_reflection_note)
