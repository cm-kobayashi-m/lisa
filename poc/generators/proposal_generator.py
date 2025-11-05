#!/usr/bin/env python3
"""
提案書生成器

ヒアリングシートまたはリフレクションノートから、
過去の類似案件を参照しつつ提案書を自動生成します。

アプローチ: ハイブリッド型（テンプレート + RAG）
1. テンプレートベースの構造生成
2. RAGで類似案件の提案書を検索
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

# CRAG機能のインポート（オプション）
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


class ProposalGenerator:
    """ヒアリングシート/リフレクションノート → 提案書生成"""

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
                temperature=0.5,  # 提案書は創造性が必要
                max_output_tokens=32768
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
            self.template_path = Path(__file__).parent / "templates" / "proposal.md"

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

    def _extract_project_info(self, source_document: str) -> Dict[str, str]:
        """
        ソースドキュメントから基本情報を抽出

        Args:
            source_document: ヒアリングシートまたはリフレクションノート

        Returns:
            抽出された情報の辞書
        """
        extraction_prompt = f"""
以下のドキュメントから、提案書作成に必要な基本情報を抽出してください。

# ソースドキュメント
{source_document}

# 抽出する情報
以下のJSON形式で出力してください。情報が見つからない場合は空文字列""を設定してください。

{{
  "project_name": "案件名",
  "customer_name": "顧客名",
  "industry": "業界",
  "current_situation": "現状の把握（3-5行）",
  "challenges": "課題の整理（箇条書き3-5項目）",
  "impact": "課題による影響（2-3行）"
}}

JSON形式のみを出力してください（説明や追加テキストは不要）。
"""
        try:
            response = self.llm.invoke(extraction_prompt)
            import json
            content = _extract_content(response)
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1])

            project_info = json.loads(content)
            return project_info
        except Exception as e:
            print(f"    [WARN] 基本情報抽出エラー: {e}")
            return {
                "project_name": "",
                "customer_name": "",
                "industry": "",
                "current_situation": "",
                "challenges": "",
                "impact": ""
            }

    def _search_similar_proposals(
        self,
        reflection_note: str,
        project_name: str = "",
        k: int = 5,
        additional_prompt: Optional[str] = None  # Query Translation用
    ) -> List[Tuple[Document, float]]:
        """
        類似案件の提案書を検索（Query Translation対応版）

        Args:
            source_document: ヒアリングシートまたはリフレクションノート
            project_name: プロジェクト名（RAG-Fusion用）
            k: 検索する件数
            additional_prompt: 追加の指示（Query Translation用）

        Returns:
            検索結果のリスト
        """
        # specialist_persona_prompt_latest.mdのパス
        persona_file = Path(__file__).parent / "../outputs" / "work_flow.md"
        # ファイル読み込み
        with open(persona_file, "r", encoding="utf-8") as f:
            your_system_prompt = f.read()

        # ToDo: ベースプロンプト（ユーザー or 組織ごとに異なるようになる）
        base_prompt = f"""
        ## 役割
        あなたはSIerのデータ事業本部のメンバーです。
        データ事業本部では「データ分析」の環境構築支援を提供している事業部です。
        データの集約から加工、可視化、分析等、ビッグデータを扱う環境構築を支援しています。「データ活用で企業のビジネスを促進すること」 がわたしたちのミッションとなります。
        DX(デジタルトランスフォーメーション)というワードを聞く機会も増えた現在、データ分析はとても大切な領域となっています。一方で、

        - データ分析/機械学習分析をしたいけれど、どこから始めれば良いのかわからない
        - データ分析環境を構築したいけれど、ノウハウがない
        - データを収集するためのツールを探している
        - データ分析基盤を構築したのだけれど、運用が難しい

        など、お客様の抱える課題は様々です。そんなお客様の抱える課題、お悩みを解決するため様々な観点から支援をするのがデータ事業本部のお仕事です。

        ## ペルソナ
        あなたは、特に**技術的な実現可能性の評価と工数見積もりを担当する、経験豊富なソリューションアーキテクト(SA)またはプロジェクトマネージャー(PM)**です。

        ## 目的
        このヒアリングシートは、**初回ヒアリングでプロジェクトの目的や大枠のスコープについて合意が取れた後**に、**精度の高い工数見積もりと技術提案を行うこと**を目的とした「**技術詳細ヒアリングシート**」です。

        このヒアリングは、主に開発担当者（PM/SA）が顧客の技術担当者や情報システム部門と行うことを想定しています。単に質問を羅列するのではなく、**「なぜその質問が必要なのか（確認の意図）」**も明確にしてください。

        ## 指示
        1.  後続のリフレクションノートを最優先のインプットとして、**この特定の案件に最適化された**ヒアリングシートを作成してください。
        2.  ヒアリング項目には、【案件コンテキスト】で与えられた具体的な製品名（例：xxxx）を積極的に使用し、「我々はこの案件を深く理解している」という姿勢を顧客に示してください。
        3.  もしコンテキスト情報が不足している場合（例：yyyyが「不明」の場合）は、そのカテゴリについて一般的な用語（例：「現在、データの加工・変換はどのようなツールやスクリプトで行っていますか？」）で質問を補完してください。
        4.  各質問には、顧客が回答をイメージしやすくなるよう、括弧書きで複数の選択肢や例（例：...）を補足してください。
        5.  各質問カテゴリの冒頭には、**「（この質問の意図：...）」**という形式で、なぜその情報を知る必要があるのかを簡潔に記述してください。

        ## 出力フォーマット（このフォーマットを厳守すること）

        | カテゴリ | 確認項目 | 質問・確認方法 | （顧客からの回答記入欄） |
        | :--- | :--- | :--- | :--- |
        | (カテゴリ名) | (確認する具体的な項目) | (顧客に提示する具体的な質問や依頼) | (空欄) |

        ---

        ## 出力フォーマットの具体例（この例を完全に模倣すること）

        | カテゴリ | 確認項目 | 質問・確認方法 | （顧客からの回答記入欄） |
        | :--- | :--- | :--- | :--- |
        | **データソース** | 接続対象システム一覧 | 現在、分析基盤に接続しているすべてのデータソース（システム、サービス、DB名など）をリストアップしてください。 | |
        | | データ種類 | 各データソースから取得しているデータの概要を教えてください。（例：出退勤ログ、ストレスチェック結果、販売実績データなど） | |
        | | 接続方式 | 各データソースへの接続方式を教えてください。（例：API、DB接続(ODBC/JDBC)、SFTPファイル転送、手動アップロードなど） | |
        | | データ形式 | 連携されるファイルの形式は何ですか？（例：CSV, JSON, Parquet, Excelなど） | |
        | | ファイル構成・サイズ | ファイル連携の場合、その構成と平均的なサイズを教えてください。（例：「数KBのファイルが毎時100個」「1GBのファイルが日次で1個」など） | |
        | | 更新頻度 | 各データソースのデータ更新頻度を教えてください。（例：リアルタイム、15分ごと、日次、月次など） | |

        ---

        ## チェックリスト生成の対象カテゴリ
        上記の具体例を参考に、以下のすべてのカテゴリについて、表形式でチェックリストを生成してください。

        1.  **データソース**
        2.  **データ連携・収集 (Ingestion)**
        3.  **データ加工・変換 (Transformation)**
        4.  **データ蓄積・保管 (Storage)**
        5.  **データ活用・可視化 (BIツール)**
        6.  **データ活用・可視化 (カスタムアプリケーション)**
        7.  **非機能要件（性能・セキュリティ・運用）**

        ## 追加指示
        {additional_prompt}

        ## 本ドキュメントを作成するフェーズ
        以下の添付の業務フローの中のヒアリングシートになります。

        # 業務フロー
        {your_system_prompt}
        
        """

        base_query = f"これから提案書を作成します。{base_prompt} 案件情報は次の通り。 {reflection_note}"

        # CRAGが有効な場合はCRAGを使用
        if self.enable_crag and self.enhanced_search:
            print(f"    [CRAG] 関連性評価付きで提案書検索中（k={k}）...")

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
            print(f"    [CRAG] {len(results)}件の提案書を発見")

        # RAG-Fusion有効化フラグ
        elif os.getenv('USE_RAG_FUSION', 'true').lower() == 'true' and project_name:
            print(f"    [RAG-Fusion] 提案書検索中（k={k}）...")

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

            print(f"    [RAG-Fusion] {len(current_results)}件(現在) + {len(similar_results)}件(類似) = 計{len(results)}件の提案書を発見")
        else:
            print(f"    [従来検索] 提案書検索中（k={k}）...")

            results = self.retriever.search_by_category(
                query=base_query,
                category="提案書",
                k=k
            )

            print(f"    [従来検索] {len(results)}件の提案書を発見")

        return results

    def _format_similar_cases(
        self,
        results: List[Tuple[Document, float]]
    ) -> str:
        """類似案件をフォーマット"""
        if not results:
            return """### 過去の類似案件実績

**参考案件データなし**

現時点で類似案件の提案書データが登録されていません。
本提案は、ヒアリングシート/リフレクションノートの情報と、
業界ベストプラクティスに基づいて作成しています。

**※ 本提案書完了後は、今後の参考のためにVector DBへの登録を推奨します**
"""

        formatted = "### 過去の類似案件実績\n\n"
        formatted += "以下の類似案件を参考に提案内容を作成しました。\n\n"

        for i, (doc, score) in enumerate(results, 1):
            similarity = max(0.0, min(1.0, 1.0 - score))
            project = doc.metadata.get("project_name", "不明")
            file_name = doc.metadata.get("file_name", "不明")

            formatted += f"#### {i}. {project}（類似度: {similarity*100:.1f}%）\n"
            formatted += f"- ファイル: {file_name}\n"

            content = doc.text[:300]
            if len(doc.text) > 300:
                content += "..."

            formatted += f"```\n{content}\n```\n\n"

        return formatted

    def _generate_solution(
        self,
        source_document: str,
        project_info: Dict[str, str],
        similar_cases: List[Tuple[Document, float]]
    ) -> Dict[str, str]:
        """
        ソリューション提案を生成

        Args:
            source_document: ソースドキュメント
            project_info: プロジェクト基本情報
            similar_cases: 類似案件

        Returns:
            ソリューション関連の情報
        """
        # 類似案件からヒントを抽出
        similar_hints = ""
        if similar_cases:
            similar_hints = "\n\n# 類似案件で提案された内容\n\n"
            for doc, _ in similar_cases[:3]:
                similar_hints += f"- {doc.metadata.get('project_name', '不明')}: {doc.text[:200]}...\n\n"

        solution_prompt = f"""
以下の情報から、顧客の課題を解決するソリューションを提案してください。

# 案件情報
{source_document}

# 課題
{project_info.get('challenges', '')}

{similar_hints}

# 出力形式
以下のJSON形式で出力してください：

{{
  "executive_summary": "エグゼクティブサマリー（提案の要点を3-5行で）",
  "solution_overview": "ソリューション概要（提案するシステム・サービスの全体像）",
  "proposed_features": "提案する機能（箇条書きで主要機能を5-7個）",
  "architecture": "システムアーキテクチャ（簡潔な説明、3-5行）",
  "technology_stack": "技術スタック（使用する主要技術を箇条書き）",
  "expected_benefits": "期待される効果（定量的・定性的効果を箇条書き）",
  "roi_analysis": "ROI分析（投資対効果の概算、2-3行）"
}}

JSON形式のみを出力してください。
"""
        try:
            response = self.llm.invoke(solution_prompt)
            import json
            content = _extract_content(response)
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1])

            solution = json.loads(content)
            return solution
        except Exception as e:
            print(f"    [WARN] ソリューション生成エラー: {e}")
            return {
                "executive_summary": "",
                "solution_overview": "",
                "proposed_features": "",
                "architecture": "",
                "technology_stack": "",
                "expected_benefits": "",
                "roi_analysis": ""
            }

    def _generate_project_plan(
        self,
        source_document: str,
        project_info: Dict[str, str]
    ) -> Dict[str, str]:
        """プロジェクト計画を生成"""
        plan_prompt = f"""
以下の案件情報から、プロジェクトの実施体制とスケジュールを提案してください。

# 案件情報
{source_document}

# 基本情報
- 案件名: {project_info.get('project_name', '')}
- 業界: {project_info.get('industry', '')}

# 出力形式
以下のJSON形式で出力してください：

{{
  "project_structure": "プロジェクト体制（役割と人数を箇条書き、例: PM 1名、開発 3名）",
  "schedule": "実施スケジュール（フェーズごとの期間を箇条書き、例: 要件定義 1ヶ月、設計 2ヶ月）",
  "milestones": "主要マイルストーン（重要な節目を箇条書き、例: 要件確定、基本設計完了）"
}}

JSON形式のみを出力してください。
"""
        try:
            response = self.llm.invoke(plan_prompt)
            import json
            content = _extract_content(response)
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1])

            plan = json.loads(content)
            return plan
        except Exception as e:
            print(f"    [WARN] プロジェクト計画生成エラー: {e}")
            return {
                "project_structure": "",
                "schedule": "",
                "milestones": ""
            }

    def _generate_cost_estimate(
        self,
        source_document: str,
        project_info: Dict[str, str]
    ) -> Dict[str, str]:
        """概算費用を生成"""
        cost_prompt = f"""
以下の案件情報から、概算費用を提案してください。

# 案件情報
{source_document}

# 出力形式
以下のJSON形式で出力してください：

{{
  "cost_breakdown": "費用内訳（開発費、インフラ費、運用費などを箇条書き）",
  "payment_terms": "支払条件（例: 契約時30%、中間30%、完了時40%）"
}}

※ 具体的な金額が不明な場合は「要ヒアリング」「別途見積」などと記載してください。

JSON形式のみを出力してください。
"""
        try:
            response = self.llm.invoke(cost_prompt)
            import json
            content = _extract_content(response)
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1])

            cost = json.loads(content)
            return cost
        except Exception as e:
            print(f"    [WARN] 費用見積生成エラー: {e}")
            return {
                "cost_breakdown": "",
                "payment_terms": ""
            }

    def _generate_risks_and_next_steps(
        self,
        source_document: str
    ) -> Dict[str, str]:
        """リスクと次のステップを生成"""
        risk_prompt = f"""
以下の案件情報から、想定されるリスクと次のステップを提案してください。

# 案件情報
{source_document}

# 出力形式
以下のJSON形式で出力してください：

{{
  "risks": "想定されるリスク（技術的、スケジュール的、体制的リスクを箇条書き）",
  "risk_mitigation": "リスク軽減策（各リスクに対する対策を箇条書き）",
  "next_steps": "提案後のプロセス（提案承認後の進め方を箇条書き）",
  "qa_items": "Q&A・追加ヒアリング（確認が必要な事項を箇条書き）"
}}

JSON形式のみを出力してください。
"""
        try:
            response = self.llm.invoke(risk_prompt)
            import json
            content = _extract_content(response)
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1])

            risks = json.loads(content)
            return risks
        except Exception as e:
            print(f"    [WARN] リスク・次ステップ生成エラー: {e}")
            return {
                "risks": "",
                "risk_mitigation": "",
                "next_steps": "",
                "qa_items": ""
            }

    def _format_value(self, value):
        """
        値を文字列形式にフォーマット
        リストの場合は改行で結合、それ以外はそのまま文字列化

        Args:
            value: フォーマットする値（文字列、リスト、または任意の型）

        Returns:
            フォーマットされた文字列
        """
        if isinstance(value, list):
            # リストの場合は改行で結合
            return "\n  ".join(str(item) for item in value)
        elif value is None:
            return "未設定"
        else:
            return str(value)

    def _generate_proposal_with_llm(
        self,
        project_info: Dict[str, str],
        similar_cases_text: str,
        solution: Dict[str, str],
        plan: Dict[str, str],
        cost: Dict[str, str],
        risks: Dict[str, str],
        source_document: str
    ) -> str:
        """
        LLMを使って提案書全体を生成

        Args:
            project_info: プロジェクト基本情報
            similar_cases_text: 類似案件情報
            solution: ソリューション情報
            plan: プロジェクト計画情報
            cost: 費用見積情報
            risks: リスク情報
            source_document: 元のソースドキュメント

        Returns:
            生成された提案書（Markdown形式）
        """
        # テンプレート構造
        template_structure = """
# 提案書

## 1. エグゼクティブサマリー
- 提案の要点
- 期待される効果
- 投資対効果

## 2. 現状分析
- お客様の現状
- 課題の整理
- 課題による影響

## 3. 提案するソリューション
- ソリューション概要
- 主要機能
- システムアーキテクチャ
- 技術スタック

## 4. 期待される効果
- 定量的効果
- 定性的効果
- ROI分析

## 5. 実施体制とスケジュール
- プロジェクト体制
- 実施スケジュール
- 主要マイルストーン

## 6. 概算費用
- 費用内訳
- 支払条件

## 7. リスクと対策
- 想定されるリスク
- リスク軽減策

## 8. 類似案件実績
- 過去の類似案件
- 成功事例

## 9. 次のステップ
- 提案後のプロセス
- Q&A・追加ヒアリング事項

## 10. 付録
- 用語集
- 参考資料
"""

        # スペシャリストペルソナを読み込み
        specialist_persona = self._load_specialist_persona()

        generation_prompt = f"""## あなたの役割

{specialist_persona}

## タスク

以下の情報を元に、顧客に提出するプロフェッショナルな提案書を作成してください。

# 入力情報

## ソースドキュメント（ヒアリングシート/リフレクションノート）
{source_document}...

## プロジェクト基本情報
- 案件名: {project_info.get("project_name", "未設定")}
- 顧客名: {project_info.get("customer_name", "未設定")}
- 業界: {project_info.get("industry", "未設定")}
- 現状: {project_info.get("current_situation", "未設定")}
- 課題: {self._format_value(project_info.get("challenges", "未設定"))}
- 影響: {project_info.get("impact", "未設定")}

## ソリューション提案
- エグゼクティブサマリー: {self._format_value(solution.get("executive_summary", "未設定"))}
- ソリューション概要: {self._format_value(solution.get("solution_overview", "未設定"))}
- 提案機能: {self._format_value(solution.get("proposed_features", "未設定"))}
- アーキテクチャ: {self._format_value(solution.get("architecture", "未設定"))}
- 技術スタック: {self._format_value(solution.get("technology_stack", "未設定"))}
- 期待される効果: {self._format_value(solution.get("expected_benefits", "未設定"))}
- ROI分析: {self._format_value(solution.get("roi_analysis", "未設定"))}

## プロジェクト計画
- プロジェクト体制: {self._format_value(plan.get("project_structure", "未設定"))}
- スケジュール: {self._format_value(plan.get("schedule", "未設定"))}
- マイルストーン: {self._format_value(plan.get("milestones", "未設定"))}

## 費用見積
- 費用内訳: {self._format_value(cost.get("cost_breakdown", "未設定"))}
- 支払条件: {self._format_value(cost.get("payment_terms", "未設定"))}

## リスクと対策
- 想定リスク: {self._format_value(risks.get("risks", "未設定"))}
- リスク軽減策: {self._format_value(risks.get("risk_mitigation", "未設定"))}
- 次のステップ: {self._format_value(risks.get("next_steps", "未設定"))}
- Q&A項目: {self._format_value(risks.get("qa_items", "未設定"))}

## 類似案件実績
{similar_cases_text}

# 出力形式

以下の構造に従って、プロフェッショナルな提案書を作成してください：

{template_structure}

# 作成上の注意点

1. **顧客目線**: 顧客のビジネス課題を理解し、それに対する解決策を明確に提示
2. **具体性**: 曖昧な表現を避け、定量的な数値や具体的な実装方法を記載
3. **説得力**: 類似案件の実績やROI分析を活用して、提案の妥当性を示す
4. **リスク対策**: リスクを隠さず、対策を明確にすることで信頼性を高める
5. **読みやすさ**: 見出し、箇条書き、表などを適切に使用して構造化
6. **Markdown形式**: 完全なMarkdown形式で出力

# 出力

Markdown形式で完全な提案書を生成してください。
余計な説明は不要です。提案書の内容のみを出力してください。
"""

        try:
            response = self.llm.invoke(generation_prompt)
            proposal = _extract_content(response)

            # メタデータを追加
            current_date = datetime.now().strftime("%Y年%m月%d日")
            footer = f"""

---

**作成日**: {current_date}
**作成者**: LISA AI
**更新日**: {current_date}
**承認者**: （承認者名）
**有効期限**: 提案日より30日間

*この提案書は、AIが過去の案件情報とヒアリング結果を分析して自動生成しました。*
*実際の提案では、顧客の状況に応じて内容を調整してください。*
"""
            proposal += footer

            return proposal

        except Exception as e:
            print(f"    [ERROR] 提案書生成エラー: {e}")
            # フォールバック
            return f"""# 提案書

## エラー

提案書の自動生成に失敗しました: {e}

## 基本情報

- 案件名: {project_info.get("project_name", "未設定")}
- 顧客名: {project_info.get("customer_name", "未設定")}
- 業界: {project_info.get("industry", "未設定")}

## ソリューション概要

{solution.get("solution_overview", "未設定")}

## リスクと対策

{risks.get("risks", "未設定")}
"""

    def generate(
        self,
        source_document: str,
        project_context: Optional[Dict[str, str]] = None,
        search_k: int = 5,
        additional_prompt: Optional[str] = None  # Query Translation用
    ) -> str:
        """
        提案書を生成（Query Translation対応版）

        Args:
            source_document: ヒアリングシートまたはリフレクションノート
            project_context: 追加のプロジェクト情報（辞書）
            search_k: RAG検索件数
            additional_prompt: 追加の指示（Query Translation用）

        Returns:
            生成された提案書（Markdown形式）
        """
        print("=" * 60)
        print("提案書生成開始")
        if additional_prompt:
            print(f"[追加指示] {additional_prompt}")
        print("=" * 60)

        # 1. 基本情報抽出
        print("\n[1/7] ソースドキュメントから基本情報を抽出中...")
        project_info = self._extract_project_info(source_document)

        if project_context:
            project_info.update(project_context)

        # 2. 類似案件検索（Query Translation対応）
        print("\n[2/7] 類似案件の提案書を検索中...")
        similar_cases = self._search_similar_proposals(
            source_document,
            project_name=project_info.get("project_name", ""),
            k=search_k,
            additional_prompt=additional_prompt  # Query Translationを使用
        )
        similar_cases_text = self._format_similar_cases(similar_cases)

        # 3. ソリューション生成
        print("\n[3/7] ソリューションを生成中...")
        solution = self._generate_solution(source_document, project_info, similar_cases)

        # 4. プロジェクト計画生成
        print("\n[4/7] プロジェクト計画を生成中...")
        plan = self._generate_project_plan(source_document, project_info)

        # 5. 費用見積生成
        print("\n[5/7] 概算費用を生成中...")
        cost = self._generate_cost_estimate(source_document, project_info)

        # 6. リスク・次ステップ生成
        print("\n[6/7] リスクと次のステップを生成中...")
        risks = self._generate_risks_and_next_steps(source_document)

        # LLMで提案書全体を生成
        print("\n[7/7] LLMで提案書全体を生成中...")
        proposal = self._generate_proposal_with_llm(
            project_info=project_info,
            similar_cases_text=similar_cases_text,
            solution=solution,
            plan=plan,
            cost=cost,
            risks=risks,
            source_document=source_document
        )

        print("\n" + "=" * 60)
        print("提案書生成完了")
        print("=" * 60)

        return proposal
