#!/usr/bin/env python3
"""
プロジェクト設定管理モジュール

複数のデータソース（Google Drive、Slack、Backlog等）に対応した
プロジェクト設定を管理します。
"""
import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any


class ProjectConfig:
    """プロジェクト設定を管理するクラス"""

    def __init__(self, config_file: str = 'project_config.yaml'):
        """
        Args:
            config_file: 設定ファイルのパス（デフォルト: project_config.yaml）
        """
        self.config_path = self._find_config_file(config_file)
        self.config = self._load_config()
        self.projects = self.config.get('projects', {})
        self.options = self.config.get('options', {})

    def _find_config_file(self, config_file: str) -> Optional[Path]:
        """設定ファイルを探す"""
        # 絶対パスの場合
        if os.path.isabs(config_file):
            path = Path(config_file)
            if path.exists():
                return path
            return None

        # 相対パスの場合、いくつかの場所を探す
        search_paths = [
            Path.cwd() / config_file,  # カレントディレクトリ
            Path(__file__).parent / config_file,  # このスクリプトと同じディレクトリ
            Path.home() / '.lisa' / config_file,  # ホームディレクトリの.lisa
        ]

        for path in search_paths:
            if path.exists():
                return path

        return None

    def _load_config(self) -> Dict:
        """YAML設定ファイルを読み込み"""
        if not self.config_path:
            print(f"[WARN] プロジェクト設定ファイルが見つかりません。デフォルト設定を使用します。")
            return {}

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
                print(f"[INFO] プロジェクト設定を読み込みました: {self.config_path}")
                return config
        except Exception as e:
            print(f"[ERROR] 設定ファイルの読み込みに失敗しました: {e}")
            return {}

    def get_projects(self) -> List[str]:
        """全プロジェクト名のリストを取得"""
        return list(self.projects.keys())

    def get_project_sources(self, project_name: str) -> Dict[str, List[str]]:
        """
        特定プロジェクトのデータソース設定を取得

        Args:
            project_name: プロジェクト名

        Returns:
            データソース名とIDリストの辞書
            例: {'google_drive': ['folder_id1', 'folder_id2'], 'slack': ['channel_id1']}
        """
        if project_name not in self.projects:
            return {}

        project = self.projects[project_name]
        sources = {}

        # 有効なデータソースのみ返す
        for source in ["google_drive","slack","backlog"]:
            if source in project and project[source]:
                sources[source] = project[source]

        return sources

    def get_google_drive_folders(self, project_name: str) -> List[str]:
        """
        プロジェクトのGoogle DriveフォルダIDリストを取得

        Args:
            project_name: プロジェクト名

        Returns:
            Google DriveのフォルダIDリスト
        """
        sources = self.get_project_sources(project_name)
        return sources.get('google_drive', [])

    def get_slack_channels(self, project_name: str) -> List[str]:
        """
        プロジェクトのSlackチャンネルIDリストを取得

        Args:
            project_name: プロジェクト名

        Returns:
            SlackのチャンネルIDリスト
        """
        sources = self.get_project_sources(project_name)
        return sources.get('slack', [])

    def get_backlog_projects(self, project_name: str) -> List[str]:
        """
        プロジェクトのBacklogプロジェクトIDリストを取得

        Args:
            project_name: プロジェクト名

        Returns:
            BacklogのプロジェクトIDリスト
        """
        sources = self.get_project_sources(project_name)
        return sources.get('backlog', [])

    def get_option(self, key: str, default: Any = None) -> Any:
        """
        オプション設定を取得

        Args:
            key: オプションキー（ドット記法対応 例: 'google_drive.max_file_size'）
            default: デフォルト値

        Returns:
            設定値
        """
        keys = key.split('.')
        value = self.options

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def is_config_loaded(self) -> bool:
        """設定ファイルが正常に読み込まれたか確認"""
        return bool(self.config_path and self.config)

    def has_project(self, project_name: str) -> bool:
        """指定されたプロジェクトが存在するか確認"""
        return project_name in self.projects

    def validate_config(self) -> List[str]:
        """
        設定の妥当性を検証

        Returns:
            警告メッセージのリスト
        """
        warnings = []

        # プロジェクトが定義されているか
        if not self.projects:
            warnings.append("プロジェクトが定義されていません")

        # 各プロジェクトにデータソースが定義されているか
        for project_name, project in self.projects.items():
            has_source = False
            for source in ['google_drive', 'slack', 'backlog']:
                if source in project and project[source]:
                    has_source = True
                    break

            if not has_source:
                warnings.append(f"プロジェクト '{project_name}' にデータソースが定義されていません")

        return warnings

    def merge_with_env(self, env_projects: List[str]) -> List[str]:
        """
        環境変数のプロジェクトリストとマージ

        環境変数でプロジェクトが指定されている場合、
        その中で設定ファイルにも存在するものを返す。
        環境変数が'*'の場合は設定ファイルの全プロジェクトを返す。

        Args:
            env_projects: 環境変数から取得したプロジェクト名リスト

        Returns:
            処理対象のプロジェクト名リスト
        """
        config_projects = self.get_projects()

        # 設定ファイルが読み込まれていない場合は環境変数のリストを返す
        if not config_projects:
            return env_projects

        # 環境変数が'*'または空の場合は設定ファイルの全プロジェクト
        if not env_projects or env_projects == ['*']:
            return config_projects

        # 両方に存在するプロジェクトのみ返す
        merged = []
        for project in env_projects:
            if project in config_projects:
                merged.append(project)
            else:
                print(f"[WARN] プロジェクト '{project}' は設定ファイルに存在しません")

        return merged

    def __str__(self) -> str:
        """設定の概要を文字列で返す"""
        if not self.is_config_loaded():
            return "ProjectConfig: 設定ファイル未読込"

        project_count = len(self.projects)
        project_list = ", ".join(self.get_projects()[:3])
        if project_count > 3:
            project_list += f", ... (他 {project_count - 3} プロジェクト)"

        return f"ProjectConfig: {project_count} プロジェクト ({project_list})"


# 互換性のためのヘルパー関数
def load_project_config(config_file: str = 'project_config.yaml') -> ProjectConfig:
    """プロジェクト設定をロード（シングルトン的に使用）"""
    return ProjectConfig(config_file)


# テスト用のメイン関数
if __name__ == "__main__":
    # 設定ファイルをロード
    config = ProjectConfig()

    # 設定が読み込まれたか確認
    print(f"設定読込状態: {config.is_config_loaded()}")
    print(config)

    # 全プロジェクトを表示
    projects = config.get_projects()
    print(f"\n登録プロジェクト数: {len(projects)}")
    for project in projects:
        print(f"  - {project}")
        sources = config.get_project_sources(project)
        for source, ids in sources.items():
            print(f"    {source}: {ids}")

    # 設定の検証
    warnings = config.validate_config()
    if warnings:
        print("\n[警告]")
        for warning in warnings:
            print(f"  - {warning}")

    # オプション設定の取得例
    print(f"\nGoogle Drive最大ファイルサイズ: {config.get_option('google_drive.max_file_size', 30)}MB")
