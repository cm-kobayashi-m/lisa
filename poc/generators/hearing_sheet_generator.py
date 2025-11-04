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
from rag.rag_fusion import rag_fusion_search, apply_hybrid_scoring

# CRAG機能のインポート
from rag.enhanced_rag_search import create_enhanced_rag_search, EnhancedRAGConfig


def _extract_content(response) -> str:
    """
    LLMレスポンスからコンテンツを抽出

    Gemini 2.0などでcontentがリスト形式で返される場合に対応

    Args:
        response: LLMからのレスポンス

    Returns:
        抽出されたテキストコンテンツ
    """
    if response is None:
        return ""

    # 公式SDK系: response.text がある場合は最短経路
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    # contentが無ければ response 自体を中身とみなす
    content = getattr(response, "content", response)
    if content is None:
        return ""

    # 文字列
    if isinstance(content, str):
        return content.strip()

    # バイト列
    if isinstance(content, (bytes, bytearray)):
        try:
            return content.decode("utf-8", errors="ignore").strip()
        except Exception:
            return ""

    # dict（Gemini系: {'parts': [...]} / {'text': '...'} など）
    if isinstance(content, dict):
        if "text" in content and isinstance(content["text"], str):
            return content["text"].strip()
        parts = content.get("parts")
        if isinstance(parts, list):
            content = parts  # 下のlist処理へ
        else:
            return str(content).strip()

    # list（LangChainのAIMessage.contentがリスト化されるケース等）
    if isinstance(content, list):
        texts = []
        for part in content:
            if part is None:
                continue
            if isinstance(part, str):
                t = part
            elif isinstance(part, dict):
                t = part.get("text")
            else:
                t = getattr(part, "text", None)
            if isinstance(t, str) and t:
                texts.append(t)
        return "".join(texts).strip()

    # フォールバック
    try:
        return str(content).strip()
    except Exception:
        return ""


class HearingSheetGenerator:
    """リフレクションノート → ヒアリングシート生成"""

    def __init__(
        self,
        vector_store: S3VectorStore,
        embeddings: GeminiEmbeddings,
        llm: Optional[ChatGoogleGenerativeAI] = None,
        template_path: Optional[str] = None,
        enable_crag: bool = False
    ):
        """
        Args:
            vector_store: S3VectorStoreインスタンス
            embeddings: GeminiEmbeddingsインスタンス
            llm: LLMインスタンス（Noneの場合は自動生成）
            template_path: テンプレートファイルパス（Noneの場合はデフォルト使用）
            enable_crag: CRAG機能を有効にするか
        """
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.retriever = RAGRetriever(vector_store, embeddings)
        self.enable_crag = enable_crag

        # CRAG機能の初期化
        if self.enable_crag:
            config = EnhancedRAGConfig(
                use_crag=True,
                use_knowledge_refinement=True,
                min_score=float(os.getenv('RAG_MIN_SCORE', '0.3'))
            )
            # Geminiクライアントは後で初期化するのでここではNoneを渡す
            self.enhanced_search = None
            self.crag_config = config
        else:
            self.enhanced_search = None
            self.crag_config = None

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

        # CRAG enhanced_searchを今初期化
        if self.enable_crag:
            self.enhanced_search = create_enhanced_rag_search(
                self.retriever,
                self.embeddings,
                self.gemini_client,
                self.crag_config
            )

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

    def _load_specialist_persona(self) -> str:
        """
        スペシャリストペルソナプロンプトを読み込み

        specialist_persona_prompt_latest.mdを読み込んで返す。
        ファイルが見つからない場合はデフォルトメッセージを返す。

        Returns:
            スペシャリストペルソナプロンプト
        """
        persona_file = Path(__file__).parent.parent / "outputs" / "specialist_persona_prompt_latest.md"

        try:
            with open(persona_file, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"    [WARN] ペルソナファイルが見つかりません: {persona_file}")
            # デフォルトメッセージ
            return "あなたは、経験豊富なPM/SA/営業のスペシャリストです。"

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

# リフレクションノート(案件情報管理シート)
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
            content = _extract_content(response)
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
        k: int = 5,
        additional_prompt: Optional[str] = None  # Query Translation用
    ) -> List[Tuple[Document, float]]:
        """
        類似案件のヒアリングシートを検索（Query Translation対応版）

        Args:
            reflection_note: リフレクションノート
            project_name: プロジェクト名（RAG-Fusion用）
            k: 検索する件数
            additional_prompt: 追加の指示（Query Translation用）

        Returns:
            検索結果のリスト
        """
        # Query Translation実行
        if additional_prompt:
            from generators.query_translator import translate_query_with_context, apply_query_filters

            print(f"    [Query Translation] 追加コンテキストを考慮したクエリ生成中...")
            translated = translate_query_with_context(
                client=self.gemini_client,
                source_document=reflection_note,
                additional_prompt=additional_prompt,
                num_queries=3
            )

            # 翻訳されたクエリとフィルタを使用
            base_query = translated["primary_query"]
            search_queries = [base_query] + translated.get("alternative_queries", [])

            # 参考プロジェクトがある場合は優先的に検索
            if translated.get("reference_projects"):
                print(f"    [Query Translation] 参考プロジェクト: {', '.join(translated['reference_projects'])}")
                # 参考プロジェクト名を検索対象に追加（プロジェクト名として扱う）
                # NOTE: 現在の実装では特定プロジェクトのフィルタリングは
                # RAG-Fusion内部で処理されるため、project_nameを上書き
                if translated["reference_projects"]:
                    # 最初の参考プロジェクトを優先
                    project_name = translated["reference_projects"][0] or project_name

            print(f"    [Query Translation] 検索クエリ: {base_query[:50]}...")
        else:
            # 従来通りの処理
            base_query = (f"これからヒアリングシートを作成します。案件情報は次の通り。 {reflection_note}")
            search_queries = None

        # CRAGが有効な場合はCRAGを使用
        if self.enable_crag and self.enhanced_search:
            print(f"    [CRAG] 関連性評価付きでヒアリングシート検索中（k={k}）...")

            # CRAGで拡張検索を実行
            crag_results = self.enhanced_search.search_with_enhancements(
                query=base_query,
                project_name=project_name or "",
                k_current=k,
                k_similar=k
            )

            # 結果を統合（現在のプロジェクトと類似プロジェクト）
            results = []
            for doc, dist in crag_results.get("current_project_results", []):
                results.append((doc, dist))
            for doc, dist in crag_results.get("similar_project_results", []):
                results.append((doc, dist))

            # 精製されたドキュメントがあれば優先
            if crag_results.get("refined_documents"):
                print(f"    [CRAG] {len(crag_results['refined_documents'])}件の精製済みドキュメント")

            print(f"    [CRAG] 関連性レベル: {crag_results.get('relevance_level', 'unknown')}")
            print(f"    [CRAG] {len(results)}件のヒアリングシートを発見")

        # RAG-Fusion有効化フラグ
        elif os.getenv('USE_RAG_FUSION', 'true').lower() == 'true' and project_name:
            print(f"    [RAG-Fusion] ヒアリングシート検索中（k={k}）...")

            # 元のproject_nameを退避（Query Translationで上書きされるため）
            orig_project = project_name

            # k分割（現在プロジェクト:類似プロジェクト = ratio:1-ratio）
            ratio = float(os.getenv('RAG_FUSION_CURRENT_RATIO', '0.5'))
            k_current = max(1, int(k * ratio))
            k_similar = max(0, k - k_current)

            # 現在のプロジェクトから検索
            current_results = rag_fusion_search(
                client=self.gemini_client,
                retriever=self.retriever,
                project_name=project_name,
                base_query=base_query,  # Query Translation済みのクエリを使用
                k=k_current,
                num_queries=int(os.getenv('RAG_FUSION_NUM_QUERIES', '3')),
                min_score=float(os.getenv('RAG_ONLY_MODE_MIN_SCORE', '0.3')),
                apply_time_weighting=True
            )

            # 他プロジェクトの類似案件から検索
            similar_results = self.retriever.get_cross_project_insights(
                query=base_query,
                exclude_project=orig_project,
                k=k_similar
            )

            # マージして再スコアリング
            all_results = list(current_results) + list(similar_results)

            scoring_method = os.getenv('RAG_SCORING_METHOD', 'hybrid')
            time_weight = float(os.getenv('RAG_TIME_WEIGHT', '0.2'))
            decay_days = int(os.getenv('RAG_DECAY_DAYS', '90'))

            all_results = apply_hybrid_scoring(
                all_results,
                scoring_method,
                time_weight,
                decay_days
            )

            results = all_results[:k]

            print(f"    [RAG-Fusion] {len(current_results)}件(現在) + {len(similar_results)}件(類似) = 計{len(results)}件のヒアリングシートを発見")
        else:
            print(f"    [従来検索] ヒアリングシート検索中（k={k}）...")

            # 従来の検索（Query Translationの結果を使用）
            query = base_query if additional_prompt else reflection_note

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

# リフレクションノート(案件情報管理シート)
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
            return _extract_content(response)
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

# リフレクションノート(案件情報管理シート)
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
            return _extract_content(response)
        except Exception as e:
            print(f"    [WARN] 追加質問生成エラー: {e}")
            return "（追加質問を生成できませんでした）"

    def _generate_hearing_sheet_with_llm(
        self,
        project_info: Dict[str, str],
        risk_assessment: str,
        similar_cases_text: str,
        additional_questions: str,
        reflection_note: str
    ) -> str:
        """
        LLMを使ってヒアリングシート全体を生成

        Args:
            project_info: プロジェクト基本情報
            risk_assessment: リスク評価結果
            similar_cases_text: 類似案件情報
            additional_questions: 追加質問項目
            reflection_note: 元のリフレクションノート

        Returns:
            生成されたヒアリングシート（Markdown形式）
        """
        # テンプレートを読み込んで構造の参考にする
        template_structure = """
# ヒアリングシート

## 1. 案件基本情報
- 案件名
- 顧客名
- 業界
- 案件規模
- 希望導入時期

## 2. 背景・課題
- 現状の課題
- 期待される効果
- 優先順位

## 3. リスク評価
- 体制・規模のリスク
- 技術的リスク
- スケジュールリスク
- 顧客体制のリスク

## 4. 類似案件からの参考情報
- 過去の類似案件での知見
- 注意すべきポイント

## 5. 追加確認事項
- ヒアリングで確認すべき項目
- 提案に向けて必要な情報

## 6. 備考
- その他特記事項
"""

        # スペシャリストペルソナを読み込み
        specialist_persona = self._load_specialist_persona()

        generation_prompt = f"""## あなたの役割

{specialist_persona}

## タスク

以下の情報を元に、顧客とのヒアリングに使用するヒアリングシートを作成してください。

# 入力情報

## リフレクションノート（案件情報管理シート）
{reflection_note}...

## プロジェクト基本情報
- 案件名: {project_info.get("project_name", "未設定")}
- 顧客名: {project_info.get("customer_name", "未設定")}
- 業界: {project_info.get("industry", "未設定")}
- 案件規模: {project_info.get("scale", "未設定")}
- 希望導入時期: {project_info.get("target_date", "未設定")}
- 背景: {project_info.get("background", "未設定")}

## リスク評価
{risk_assessment}

## 類似案件からの参考情報
{similar_cases_text}

## 追加確認事項（自動抽出）
{additional_questions}

# 出力形式

以下の構造に従って、プロフェッショナルなヒアリングシートを作成してください：

{template_structure}

# 作成上の注意点

1. **実践的な内容**: 顧客とのヒアリングで実際に使える具体的な質問や確認事項を記載
2. **リスクベース**: 特定されたリスクに対して、どのような情報を確認すべきか明記
3. **類似案件の活用**: 過去の類似案件から学んだ教訓を反映
4. **優先順位**: 重要度の高い確認事項を強調
5. **Markdown形式**: 見出し、箇条書き、表などを適切に使用

# 出力

Markdown形式で完全なヒアリングシートを生成してください。
余計な説明は不要です。ヒアリングシートの内容のみを出力してください。
"""

        try:
            response = self.llm.invoke(generation_prompt)
            hearing_sheet = _extract_content(response)

            # メタデータを追加
            current_date = datetime.now().strftime("%Y年%m月%d日")
            footer = f"""

---

**作成日**: {current_date}
**作成者**: LISA AI
**更新日**: {current_date}

*このヒアリングシートは、AIが過去の案件情報とリフレクションノートを分析して自動生成しました。*
*実際のヒアリングでは、顧客の状況に応じて柔軟に対応してください。*
"""
            hearing_sheet += footer

            return hearing_sheet

        except Exception as e:
            print(f"    [ERROR] ヒアリングシート生成エラー: {e}")
            # フォールバック: 基本的な情報だけを含むシート
            return f"""# ヒアリングシート

## エラー

ヒアリングシートの自動生成に失敗しました: {e}

## 基本情報

- 案件名: {project_info.get("project_name", "未設定")}
- 顧客名: {project_info.get("customer_name", "未設定")}
- 業界: {project_info.get("industry", "未設定")}

## リスク評価

{risk_assessment}

## 追加確認事項

{additional_questions}
"""

    def generate(
        self,
        reflection_note: str,
        project_context: Optional[Dict[str, str]] = None,
        search_k: int = 5,
        additional_prompt: Optional[str] = None  # Query Translation用
    ) -> str:
        """
        ヒアリングシートを生成（Query Translation対応版）

        Args:
            reflection_note: リフレクションノート
            project_context: 追加のプロジェクト情報（辞書）
            search_k: RAG検索件数
            additional_prompt: 追加の指示（Query Translation用）

        Returns:
            生成されたヒアリングシート（Markdown形式）
        """
        print("=" * 60)
        print("ヒアリングシート生成開始")
        if additional_prompt:
            print(f"[追加指示] {additional_prompt}")
        print("=" * 60)

        # 1. 基本情報抽出
        print("\n[1/5] リフレクションノートから基本情報を抽出中...")
        project_info = self._extract_project_info(reflection_note)

        # project_contextでオーバーライド
        if project_context:
            project_info.update(project_context)

        # 2. 類似案件検索（Query Translation対応）
        print("\n[2/5] 類似案件のヒアリングシートを検索中...")
        similar_cases = self._search_similar_hearing_sheets(
            reflection_note,
            project_name=project_info.get("project_name", ""),
            k=search_k,
            additional_prompt=additional_prompt  # Query Translationを使用
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

        # 5. LLMでヒアリングシート全体を生成
        print("\n[5/5] LLMでヒアリングシート全体を生成中...")
        hearing_sheet = self._generate_hearing_sheet_with_llm(
            project_info=project_info,
            risk_assessment=risk_assessment,
            similar_cases_text=similar_cases_text,
            additional_questions=additional_questions,
            reflection_note=reflection_note
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
