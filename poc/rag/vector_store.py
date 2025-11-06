"""
Amazon S3 Vectors を使用したベクトルストア実装

S3 Vectors (Preview) を使用してベクトルデータの保存と検索を行います。
コスト効率を重視し、サーバーレスアーキテクチャで動作します。
"""
import os
import json
import logging
import time
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import boto3
from botocore.exceptions import ClientError, BotoCoreError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

# ロギング設定
logger = logging.getLogger(__name__)


@dataclass
class Document:
    """ドキュメントを表すデータクラス"""
    key: str  # ユニークなID
    text: str  # ドキュメントのテキスト内容
    metadata: Dict[str, Any]  # メタデータ（プロジェクト名、ファイル名など）
    vector: Optional[List[float]] = None  # ベクトル表現


class S3VectorStore:
    """
    Amazon S3 Vectors を使用したベクトルストア

    S3 Vectorsの特徴:
    - S3ネイティブのベクトル検索
    - サーバーレスで超低コスト
    - サブ秒レスポンス
    - 最大50Mベクトル/インデックス
    """

    def __init__(
        self,
        vector_bucket_name: str,
        index_name: str,
        dimension: int = 1536,  # Gemini埋め込みのデフォルト次元
        distance_metric: str = "cosine",
        region_name: str = "us-west-2",
        create_if_not_exists: bool = True
    ):
        """
        S3 Vectorsストアを初期化

        Args:
            vector_bucket_name: S3ベクトルバケット名
            index_name: インデックス名
            dimension: ベクトルの次元数（Gemini: 1536）
            distance_metric: 距離メトリック（cosine/euclidean）
            region_name: AWSリージョン（Preview対応: us-west-2, us-east-1等）
            create_if_not_exists: 存在しない場合に自動作成するか
        """
        self.vector_bucket_name = vector_bucket_name
        self.index_name = index_name
        self.dimension = dimension
        self.distance_metric = distance_metric
        self.region_name = region_name

        # S3 Vectorsクライアントを初期化
        try:
            # AWS_PROFILEが設定されている場合はそれを使用
            profile_name = os.getenv('AWS_PROFILE')
            if profile_name:
                session = boto3.Session(profile_name=profile_name, region_name=region_name)
                self.client = session.client("s3vectors")
                logger.info(f"S3 Vectorsクライアント初期化（プロファイル: {profile_name}, リージョン: {region_name}）")
            else:
                self.client = boto3.client(
                    "s3vectors",
                    region_name=region_name
                )
                logger.info(f"S3 Vectorsクライアント初期化（リージョン: {region_name}）")
        except Exception as e:
            logger.error(f"S3 Vectorsクライアントの初期化に失敗: {e}")
            raise

        # バケットとインデックスの初期化
        if create_if_not_exists:
            self._initialize_storage()

    def _is_throttling_error(self, exception: Exception) -> bool:
        """スロットリングエラーかどうかを判定"""
        if isinstance(exception, ClientError):
            error_code = exception.response.get('Error', {}).get('Code', '')
            return error_code in ['TooManyRequestsException', 'ThrottlingException', 'RequestLimitExceeded']
        return False

    @retry(
        stop=stop_after_attempt(8),
        wait=wait_exponential(multiplier=2, min=4, max=120),
        retry=retry_if_exception_type(ClientError),
        before_sleep=lambda retry_state: (
            logger.warning(
                f"S3 Vectors APIレート制限検出 (試行 {retry_state.attempt_number}/8) - "
                f"{retry_state.next_action.sleep}秒待機してリトライします..."
            ),
            print(
                f"    [リトライ] S3 Vectors APIレート制限 (試行 {retry_state.attempt_number}/8) - "
                f"{retry_state.next_action.sleep:.0f}秒待機中..."
            )
        )
    )
    def _put_vectors_with_retry(self, vectors: List[Dict]) -> None:
        """
        ベクトルをS3 Vectorsに保存（リトライ機能付き）

        Args:
            vectors: 保存するベクトルのリスト

        Raises:
            ClientError: スロットリングエラー以外のエラー時
        """
        try:
            self.client.put_vectors(
                vectorBucketName=self.vector_bucket_name,
                indexName=self.index_name,
                vectors=vectors
            )
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            error_message = e.response.get('Error', {}).get('Message', '')

            # ValidationExceptionの詳細ログ
            if error_code == 'ValidationException':
                logger.error(f"ValidationException詳細: {error_message}")
                logger.error(f"エラー全体: {e.response}")
                # ベクトルデータの一部をログ出力（デバッグ用）
                if vectors:
                    # すべてのキーをチェックして重複を検出
                    all_keys = [v.get('key', 'なし') for v in vectors]
                    unique_keys = set(all_keys)
                    if len(all_keys) != len(unique_keys):
                        logger.error(f"[重要] バッチ内にキーの重複があります！")
                        logger.error(f"全キー数: {len(all_keys)}, ユニークキー数: {len(unique_keys)}")
                        # 重複キーを特定
                        from collections import Counter
                        key_counts = Counter(all_keys)
                        duplicates = [k for k, c in key_counts.items() if c > 1]
                        logger.error(f"重複キー: {duplicates[:5]}")  # 最初の5つまで表示

                    first_vector = vectors[0]
                    logger.error(f"最初のベクトルのキー: {first_vector.get('key', 'なし')}")
                    logger.error(f"ベクトルの次元数: {len(first_vector.get('data', {}).get('float32', []))}")
                    logger.error(f"メタデータのキー: {list(first_vector.get('metadata', {}).keys())}")

            if self._is_throttling_error(e):
                logger.warning(f"S3 Vectors APIレート制限: {error_code}")
                raise  # リトライ
            else:
                logger.error(f"ベクトルの追加に失敗（リトライ不可）: {error_code} - {error_message}")
                raise

    def _initialize_storage(self):
        """ベクトルバケットとインデックスを初期化"""
        try:
            # ベクトルバケットの作成または確認
            self._create_vector_bucket()

            # インデックスの作成または確認
            self._create_index()

        except ClientError as e:
            logger.error(f"ストレージの初期化に失敗: {e}")
            raise

    def _create_vector_bucket(self):
        """ベクトルバケットを作成（既存の場合はスキップ）"""
        # バケットは手動作成済みなのでスキップ
        logger.info(f"S3バケット '{self.vector_bucket_name}' は既に存在します（手動作成済み）")

    def _create_index(self):
        """インデックスを作成（既存の場合はスキップ）"""
        try:
            # インデックスの存在確認
            self.client.get_index(
                vectorBucketName=self.vector_bucket_name,
                indexName=self.index_name
            )
            logger.info(f"インデックス '{self.index_name}' は既に存在します")

        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            logger.info(f"インデックス確認エラー（エラーコード: {error_code}）")

            # インデックスが見つからない場合のエラーコード
            if error_code in ['NoSuchIndex', 'ResourceNotFoundException', 'NotFoundException']:
                # インデックスが存在しない場合は作成
                logger.info(f"インデックス '{self.index_name}' が存在しないため作成します")
                try:
                    response = self.client.create_index(
                        vectorBucketName=self.vector_bucket_name,
                        indexName=self.index_name,
                        dataType="float32",
                        dimension=self.dimension,
                        distanceMetric=self.distance_metric,
                        metadataConfiguration={
                            # source_textは非フィルタブル（大きなテキストを保存可能）
                            "nonFilterableMetadataKeys": ["source_text", "analysis_result"]
                        }
                    )
                    logger.info(f"インデックス '{self.index_name}' を作成しました")
                    logger.info(f"  - 次元数: {self.dimension}")
                    logger.info(f"  - 距離メトリック: {self.distance_metric}")
                    logger.info(f"  - レスポンス: {response}")

                except ClientError as create_error:
                    logger.error(f"インデックスの作成に失敗: {create_error}")
                    logger.error(f"エラーレスポンス: {create_error.response}")
                    raise
            else:
                logger.error(f"インデックスの確認に失敗: {e}")
                raise

    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100
    ) -> int:
        """
        ドキュメントをベクトルストアに追加

        Args:
            documents: 追加するドキュメントのリスト
            batch_size: バッチサイズ（Preview制限を考慮）

        Returns:
            追加されたドキュメント数
        """
        if not documents:
            return 0

        added_count = 0
        total_batches = (len(documents) + batch_size - 1) // batch_size

        # バッチ処理
        for batch_idx, i in enumerate(range(0, len(documents), batch_size), 1):
            batch = documents[i:i + batch_size]
            print(f"    [S3 Vectors] バッチ {batch_idx}/{total_batches} ({len(batch)}件) を保存中...")

            # S3 Vectors形式に変換
            vectors = []
            seen_keys = set()  # 重複キーチェック用

            # メタデータサイズ制限ヘルパー関数
            def truncate_str(s, max_bytes=4000):
                """文字列を指定バイト数以下に切り詰める"""
                if not s:
                    return ""
                s_str = str(s)  # 確実に文字列に変換
                encoded = s_str.encode('utf-8')
                if len(encoded) <= max_bytes:
                    return s_str
                # バイト数を超える場合は切り詰め
                truncated = encoded[:max_bytes-100].decode('utf-8', 'ignore')
                return truncated + '...'

            def create_json_field(data, max_bytes=4000):
                """JSONフィールドを作成し、サイズ制限を適用"""
                json_str = json.dumps(data, ensure_ascii=False)
                return truncate_str(json_str, max_bytes)

            for doc in batch:
                if not doc.vector:
                    logger.warning(f"ドキュメント '{doc.key}' にベクトルがありません")
                    continue

                # メタデータの準備（S3 Vectorsの10キー制限に対応）
                import json

                # フィルタ用の独立フィールド（3個）- サイズ制限を適用
                metadata = {
                    "project_name": truncate_str(doc.metadata.get("project_name", ""), 500),
                    "file_name": truncate_str(doc.metadata.get("file_name", ""), 500),
                    "document_type": truncate_str(doc.metadata.get("document_type", ""), 200),

                    # 基本情報をJSON化（4個目）
                    "basic_info": create_json_field({
                        "title": doc.metadata.get("title", ""),
                        "chunk_index": doc.metadata.get("chunk_index", 0),
                        "created_at": str(doc.metadata.get("created_at", "")),
                        "modified_at": str(doc.metadata.get("modified_at", "")),
                        "doc_type": doc.metadata.get("doc_type", "document"),
                        "data_source": doc.metadata.get("data_source", ""),
                        "source_id": doc.metadata.get("source_id", "")
                    }, 3900),

                    # 階層情報をJSON化（5個目）
                    "hierarchy_info": create_json_field({
                        "heading_level": doc.metadata.get("heading_level", 0),
                        "section_path": doc.metadata.get("section_path", "")
                    }, 3900),

                    # 構造情報をJSON化（6個目）
                    "structure_info": create_json_field({
                        "has_table": doc.metadata.get("has_table", False),
                        "has_list": doc.metadata.get("has_list", False),
                        "has_code": doc.metadata.get("has_code", False),
                        "has_numbers": doc.metadata.get("has_numbers", False)
                    }, 3900),

                    # 位置情報をJSON化（7個目）
                    "position_info": create_json_field({
                        "chunk_position": doc.metadata.get("chunk_position", ""),
                        "relative_position": doc.metadata.get("relative_position", 0.0),
                        "prev_section": truncate_str(doc.metadata.get("prev_section", ""), 1000),
                        "next_section": truncate_str(doc.metadata.get("next_section", ""), 1000)
                    }, 3900),

                    # 分析情報をJSON化（8個目）
                    "analysis_info": create_json_field({
                        "information_density": doc.metadata.get("information_density", 0.0),
                        "importance": doc.metadata.get("importance", "medium"),
                        "document_type_confidence": doc.metadata.get("document_type_confidence", 0.0)
                    }, 3900),

                    # リスト型データをJSON化（9個目）- リストサイズも制限
                    "list_data": create_json_field({
                        "speakers": doc.metadata.get("speakers", [])[:10],  # 最大10人まで
                        "pages": doc.metadata.get("pages", [])[:20],  # 最大20ページまで
                        "element_types": doc.metadata.get("element_types", [])[:10],  # 最大10種類まで
                        "temporal_index": doc.metadata.get("temporal_index", 0)
                    }, 3900),

                    # テキスト内容（10個目、4KB制限）
                    "source_text": truncate_str(doc.text, 3900)
                }

                # 重複キーチェック
                if doc.key in seen_keys:
                    print(f"    [ERROR] 重複キー検出: {doc.key}")
                    logger.error(f"重複キー検出: {doc.key}")
                    # スキップして次のドキュメントへ
                    continue

                seen_keys.add(doc.key)

                vectors.append({
                    "key": doc.key,
                    "data": {"float32": doc.vector},
                    "metadata": metadata
                })

            if vectors:
                try:
                    # ベクトルを追加（リトライ機能付き）
                    self._put_vectors_with_retry(vectors)
                    added_count += len(vectors)
                    print(f"    [S3 Vectors] バッチ {batch_idx}/{total_batches} 保存完了 ({added_count}/{len(documents)})")

                    # バッチ間にレート制限対策の待機時間を追加
                    # 次のバッチがある場合のみ待機
                    if i + batch_size < len(documents):
                        print(f"    [待機] レート制限回避のため0.5秒待機中...")
                        time.sleep(0.5)  # 500ms待機

                except ClientError as e:
                    logger.error(f"ベクトルの追加に失敗（全リトライ失敗）: {e}")
                    print(f"    [ERROR] S3 Vectors保存エラー: {e}")
                    # リトライ失敗後は継続せず例外をスロー
                    raise

        return added_count

    def search_by_category(
        self,
        query_vector: List[float],
        category: str,
        k: int = 30,
        additional_filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        ドキュメント種別でフィルタした類似度検索

        Args:
            query_vector: クエリベクトル
            category: ドキュメント種別（例: "提案書", "ヒアリングシート"）
            k: 返す結果の数（最大30）
            additional_filters: 追加のフィルタ条件（例: {"project_name": {"$eq": "プロジェクトA"}}）

        Returns:
            (Document, 距離)のタプルのリスト

        Example:
            # ヒアリングシートのみを検索
            results = store.search_by_category(
                query_vector=vector,
                category="ヒアリングシート",
                k=10
            )

            # 特定プロジェクトの提案書のみを検索
            results = store.search_by_category(
                query_vector=vector,
                category="提案書",
                k=10,
                additional_filters={"project_name": {"$eq": "ヤーマン"}}
            )
        """
        # カテゴリフィルタを構築
        filter_dict = {"document_type": {"$eq": category}}

        # 追加のフィルタがある場合はマージ
        if additional_filters:
            # $andで結合
            filter_dict = {
                "$and": [
                    {"document_type": {"$eq": category}},
                    additional_filters
                ]
            }

        return self.similarity_search(
            query_vector=query_vector,
            k=k,
            filter_dict=filter_dict
        )

    def similarity_search(
        self,
        query_vector: List[float],
        k: int = 30,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        類似度検索を実行

        Args:
            query_vector: クエリベクトル
            k: 返す結果の数（最大30）
            filter_dict: フィルタ条件（例: {"project_name": {"$eq": "プロジェクトA"}}）
                        カテゴリフィルタ: {"document_type": {"$eq": "提案書"}}
                        複数条件: {"$and": [{"document_type": {"$eq": "提案書"}}, {"project_name": {"$eq": "プロジェクトA"}}]}

        Returns:
            (Document, 距離)のタプルのリスト

        Note:
            既存コードとの互換性のため、filter_dictがNoneの場合は全ドキュメントを対象とします
        """
        if k > 30:
            logger.warning("kの値が30を超えています。30に制限します。")
            k = 30

        try:
            # クエリパラメータの構築
            query_params = {
                "vectorBucketName": self.vector_bucket_name,
                "indexName": self.index_name,
                "queryVector": {"float32": query_vector},
                "topK": k,
                "returnDistance": True,
                "returnMetadata": True
            }

            # フィルタがある場合は追加
            if filter_dict:
                query_params["filter"] = filter_dict

            # 検索実行
            response = self.client.query_vectors(**query_params)

            # 結果をDocument形式に変換
            results = []
            for vector_result in response.get("vectors", []):
                metadata = vector_result.get("metadata", {})

                # Documentオブジェクトを再構築（グループ化されたメタデータを復元）
                import json

                # JSONグループを解析
                basic_info = {}
                hierarchy_info = {}
                structure_info = {}
                position_info = {}
                analysis_info = {}
                list_data = {}

                # 基本情報のパース
                try:
                    basic_info = json.loads(metadata.get("basic_info", "{}"))
                except (json.JSONDecodeError, TypeError):
                    basic_info = {}

                # 階層情報のパース
                try:
                    hierarchy_info = json.loads(metadata.get("hierarchy_info", "{}"))
                except (json.JSONDecodeError, TypeError):
                    hierarchy_info = {}

                # 構造情報のパース
                try:
                    structure_info = json.loads(metadata.get("structure_info", "{}"))
                except (json.JSONDecodeError, TypeError):
                    structure_info = {}

                # 位置情報のパース
                try:
                    position_info = json.loads(metadata.get("position_info", "{}"))
                except (json.JSONDecodeError, TypeError):
                    position_info = {}

                # 分析情報のパース
                try:
                    analysis_info = json.loads(metadata.get("analysis_info", "{}"))
                except (json.JSONDecodeError, TypeError):
                    analysis_info = {}

                # リスト型データのパース
                try:
                    list_data = json.loads(metadata.get("list_data", "{}"))
                except (json.JSONDecodeError, TypeError):
                    list_data = {}

                # メタデータを再構築
                doc = Document(
                    key=vector_result["key"],
                    text=metadata.get("source_text", ""),
                    metadata={
                        # フィルタ用フィールド
                        "project_name": metadata.get("project_name", ""),
                        "file_name": metadata.get("file_name", ""),
                        "document_type": metadata.get("document_type", ""),

                        # 基本情報
                        "title": basic_info.get("title", ""),
                        "chunk_index": basic_info.get("chunk_index", 0),
                        "created_at": basic_info.get("created_at", ""),
                        "modified_at": basic_info.get("modified_at", ""),
                        "doc_type": basic_info.get("doc_type", "document"),
                        "data_source": basic_info.get("data_source", ""),
                        "source_id": basic_info.get("source_id", ""),

                        # 階層情報
                        "heading_level": hierarchy_info.get("heading_level", 0),
                        "section_path": hierarchy_info.get("section_path", ""),

                        # 構造情報
                        "has_table": structure_info.get("has_table", False),
                        "has_list": structure_info.get("has_list", False),
                        "has_code": structure_info.get("has_code", False),
                        "has_numbers": structure_info.get("has_numbers", False),

                        # 位置情報
                        "chunk_position": position_info.get("chunk_position", ""),
                        "relative_position": position_info.get("relative_position", 0.0),
                        "prev_section": position_info.get("prev_section", ""),
                        "next_section": position_info.get("next_section", ""),

                        # 分析情報
                        "information_density": analysis_info.get("information_density", 0.0),
                        "importance": analysis_info.get("importance", "medium"),
                        "document_type_confidence": analysis_info.get("document_type_confidence", 0.0),

                        # リスト型データ
                        "speakers": list_data.get("speakers", []),
                        "pages": list_data.get("pages", []),
                        "element_types": list_data.get("element_types", []),
                        "temporal_index": list_data.get("temporal_index", 0)
                    }
                )

                # 距離（類似度）を取得
                distance = vector_result.get("distance", 0.0)

                results.append((doc, distance))

            logger.info(f"検索完了: {len(results)}件の結果を取得")
            return results

        except ClientError as e:
            logger.error(f"ベクトル検索に失敗: {e}")
            raise

    def delete_documents(self, keys: List[str]) -> int:
        """
        指定されたキーのドキュメントを削除

        Args:
            keys: 削除するドキュメントのキーリスト

        Returns:
            削除されたドキュメント数
        """
        if not keys:
            return 0

        try:
            self.client.delete_vectors(
                vectorBucketName=self.vector_bucket_name,
                indexName=self.index_name,
                keys=keys
            )
            logger.info(f"{len(keys)}件のドキュメントを削除しました")
            return len(keys)

        except ClientError as e:
            logger.error(f"ドキュメントの削除に失敗: {e}")
            raise

    def get_document(self, key: str) -> Optional[Document]:
        """
        指定されたキーのドキュメントを取得

        Args:
            key: ドキュメントのキー

        Returns:
            Document または None
        """
        try:
            response = self.client.get_vectors(
                vectorBucketName=self.vector_bucket_name,
                indexName=self.index_name,
                keys=[key]
            )

            vectors = response.get("vectors", [])
            if vectors:
                vector_data = vectors[0]
                metadata = vector_data.get("metadata", {})

                return Document(
                    key=vector_data["key"],
                    text=metadata.get("source_text", ""),
                    metadata={
                        "project_name": metadata.get("project_name", ""),
                        "file_name": metadata.get("file_name", ""),
                        "created_at": metadata.get("created_at", ""),
                        "modified_at": metadata.get("modified_at", ""),
                        "doc_type": metadata.get("doc_type", "document")
                    },
                    vector=vector_data.get("data", {}).get("float32", [])
                )

            return None

        except ClientError as e:
            logger.error(f"ドキュメントの取得に失敗: {e}")
            return None

    def list_documents(
        self,
        max_results: int = 100,
        next_token: Optional[str] = None
    ) -> Tuple[List[str], Optional[str]]:
        """
        インデックス内のドキュメントキーをリスト

        Args:
            max_results: 最大結果数
            next_token: ページネーショントークン

        Returns:
            (キーのリスト, 次のページトークン)
        """
        try:
            params = {
                "vectorBucketName": self.vector_bucket_name,
                "indexName": self.index_name,
                "maxResults": min(max_results, 1000)
            }

            if next_token:
                params["nextToken"] = next_token

            response = self.client.list_vectors(**params)

            keys = [v["key"] for v in response.get("vectors", [])]
            next_token = response.get("nextToken")

            return keys, next_token

        except ClientError as e:
            logger.error(f"ドキュメントのリストに失敗: {e}")
            raise

    def delete_index(self):
        """インデックスを削除"""
        try:
            self.client.delete_index(
                vectorBucketName=self.vector_bucket_name,
                indexName=self.index_name
            )
            logger.info(f"インデックス '{self.index_name}' を削除しました")

        except ClientError as e:
            logger.error(f"インデックスの削除に失敗: {e}")
            raise

    def delete_bucket(self):
        """ベクトルバケットを削除"""
        try:
            # まずインデックスを削除
            self.delete_index()

            # バケットを削除
            self.client.delete_vector_bucket(
                vectorBucketName=self.vector_bucket_name
            )
            logger.info(f"ベクトルバケット '{self.vector_bucket_name}' を削除しました")

        except ClientError as e:
            logger.error(f"ベクトルバケットの削除に失敗: {e}")
            raise
