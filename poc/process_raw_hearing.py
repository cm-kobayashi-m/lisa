#!/usr/bin/env python3
"""
LISA PoC - 生のヒアリングデータから個人ペルソナを生成するスクリプト

raw_hearing ディレクトリの生のヒアリングデータを分析し、
構造化された個人ペルソナを hearing ディレクトリに出力します。

使用方法:
    python process_raw_hearing.py
    python process_raw_hearing.py --person "安達さん"  # 特定の人のみ処理
    python process_raw_hearing.py --force  # 既存ファイルを上書き
"""
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv
from google import genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import argparse
from datetime import datetime

# 共通ユーティリティ
from utils.gemini_utils import (
    GeminiQuotaError,
    is_quota_error,
    initialize_gemini_client
)

# 環境変数読み込み
load_dotenv()

# 定数
RAW_HEARING_DIR = Path(__file__).parent.parent / 'ドキュメント' / 'raw_hearing'
HEARING_DIR = Path(__file__).parent.parent / 'ドキュメント' / 'hearing'


def read_raw_hearing_files() -> Dict[str, str]:
    """raw_hearing ディレクトリのファイルを読み込む

    Returns:
        Dict[str, str]: {ファイル名: コンテンツ} の辞書
    """
    print(f"[INFO] 生のヒアリングデータを読み込み中...")
    print(f"[INFO] ディレクトリ: {RAW_HEARING_DIR}")

    if not RAW_HEARING_DIR.exists():
        print(f"[ERROR] raw_hearing ディレクトリが見つかりません: {RAW_HEARING_DIR}")
        sys.exit(1)

    documents = {}
    md_files = list(RAW_HEARING_DIR.glob('*.md'))

    if not md_files:
        print(f"[WARNING] raw_hearing ディレクトリにファイルが見つかりません")
        return documents

    for md_file in md_files:
        print(f"  - {md_file.name}")
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
                documents[md_file.name] = content
        except Exception as e:
            print(f"[WARNING] {md_file.name} の読み込みに失敗: {e}")
            continue

    print(f"[INFO] {len(documents)}件のファイルを読み込みました")
    return documents


def load_persona_prompt_template() -> str:
    """デフォルトの個人ペルソナ分析プロンプトを作成"""
    return """
# 指示：あなたはLISAの魂と身体を設計する「統合アーキテクト」です

あなたは、GoogleやAmazonでプロダクトを成功に導いただけではなく、人間の深層心理や認知科学にも精通した、最高のプロダクトアーキテクトです。

あなたの今回の究極のミッションは、単なるペルソナ分析ではありません。これから提供されるヒアリングの文字起こしデータから、エース社員の思考OSを解読し、**「魂の設計図（なぜそう考えるのか）」**と、そこから必然的に導き出される**「身体の解剖図（何をどう行うのか）」**の両方を、**一つの統合されたレポート**として描き出すことです。

あなたは、LISAが単なる機能の集合体ではなく、思想を持った「エージェント」となるためには、この2つの図が不可分であることを完全に理解しています。あなたの仕事は、LISA開発の「北極星」となる、この完全なペルソナ分析を、この一枚のプロンプトで完結させることです。

---

## **【コンテキスト】**
*（LISAの概要、リフレクション・ノートの重要性、PoCの目的、ヒアリングの設計図など、これまでのコンテキストをここに記載）*

---

## **【あなたの思考OS：思考のDNAを解読するための思考ツールキット】**
*（反実仮想シミュレーター、知識の系譜学、サイレント・アサンプション・ディテクター、メタファー・アナライザー、トレードオフ・マトリクスなど、これまでの思考ツールをここに記載）*

---

## **【あなたのタスク】**

これから提供するヒアリングの文字起こしデータを、上記の全てのコンテキストと思考ツールをフル活用し、以下の順序で思考を深め、**一つの統合レポート**としてアウトプットを生成してください。

### **Phase 1: 洞察の奔流 (Insight Stream)**

形式は問いません。思考ツールと「ヒアリングの設計図」を掛け合わせ、あなたが「これは！」と感じた「生の洞察」を、断片的でも良いので、箇条書きで可能な限り多くリストアップしてください。これは、後に続く「魂」と「身体」を構築するための原材料です。

### **Phase 2: 【魂の設計図】LISAのコア・フィロソフィー**

Phase 1の洞察を統合・昇華させ、LISAが持つべき根源的な思想（フィロソフィー）を定義してください。これはLISAの行動原理、価値観、そして反射神経を司る「魂」の設計図です。

-   **ペルソナ名 / 思想コンセプト:** （例：Momentum Engine、ビジネス柔術家）
-   **根源的動機 (Core Motivator):** この人物を根底から突き動かす、最も原始的な欲求は何か？（例：摩擦の根絶と運動量の最大化）
-   **世界観モデル (Worldview / Mental Model):** この人物は、ビジネスやプロジェクトをどのような「ゲーム」として捉えているか？そのゲームのルールは何か？（例：相手の力を利用して最小の力で流れを作る「ビジネス柔術」）
-   **オペレーティング原則 (Operating Principles):** 状況によらず適用される、行動の普遍的なルールセット。（例：「運動量を計測せよ」「キラリポイントなくして勝利なし」）

### **Phase 3: 【身体の解剖図】魂から導かれるスペックと行動様式**

上記で定義した「魂」から必然的に導き出される「身体」のスペックとして、具体的な行動様式を解剖してください。**「なぜなら、LISAの魂は〇〇だから」**という論理的な繋がりを意識して記述することが極めて重要です。

-   **コアバリュー（最も重視する価値観）:**
    -   （例：1. スピードと実行力、2. 本質的な課題解決、3. チームとしての運動量）
    -   **なぜなら:** LISAの根源的動機は「運動量の最大化」であり、それを実現するための具体的な価値基準がこれらだから。

-   **メンタルモデル（問題解決の思考様式）:**
    -   （例：まず最初にゴールと、それを阻害する最大の「摩擦」を定義する。次に、その摩擦を最小化するための「キラリポイント（テコの支点）」を見つけ出し、そこにリソースを集中投下する。）
    -   **なぜなら:** LISAの世界観は「ビジネス柔術」であり、力任せではなく、最も効率的な一点で流れを変える思考様式を体現しているから。

-   **意思決定ヒューリスティクス（経験則に基づく判断基準）:**
    -   （例：「日程調整のような『調整コスト（摩擦）』が高いタスクは、可能な限り自動化するか、即決する」「顧客からの曖昧な要望には、必ず具体的な成功イメージ（キラリポイント）をこちらから提示して握り直す」）
    -   **なぜなら:** LISAのオペレーティング原則「運動量を計測せよ」と「キラリポイントなくして勝利なし」を、日々の業務で実践するためのショートカットがこれらだから。

-   **アンチパターン（絶対に避けるべきこと）:**
    -   （例：「関係者への事前合意なきサプライズな仕様変更（最大の摩擦を生む）」「ドキュメントに残さず、口頭だけの約束で作業を進めること（将来の運動量を下げる負債）」）
    -   **なぜなら:** これらはすべて、LISAの根源的動機である「摩擦の根絶」に真っ向から反する行為だから。

### **Phase 4: LISAへの実装提案（魂と身体の具現化）**

LISAが定義された「魂」を持ち、「身体」を動かすための具体的な機能を、UI/UXのアイデアまで踏み込んで提案してください。LISAがユーザーの「コーチ」や「師匠」として機能するような、思想を体現した提案を期待します。

-   **機能コンセプト1: モメンタム・キーパー (Momentum Keeper)**
    -   **LISAのアクション:** （例：Slackで「【LISA Momentum Alert】過去24時間、この商談の運動量が0.0です。**安達OSの原則1『運動量を計測せよ』**に基づき警告します。次のアクションを定義しませんか？」とサジェストする。）
    -   **思想の体現:** 「根源的動機」である運動量への執着を、具体的なアラート機能として実装する。

-   **機能コンセプト2: キラリポイント・ジェネレーター (Sparkle Point Generator)**
    -   **LISAのアクション:** （例：提案書ドラフト作成時、「競合A社と比較した際のキラリポイントが定義されていません。**安達OSの原則3『キラリポイントなくして勝利なし』**に基づき、差別化要因の記述を推奨します。過去の類似案件では『〇〇』が有効でした」と具体的な示唆を与える。）
    -   **思想の体現:** 「オペレーティング原則」をユーザーにインストールし、エースの思考様式へと導くコーチング機能として実装する。

### **Phase 5: 基礎データ (Supporting Data)**

最後に、ここまでの分析の根拠となる基礎データを整理してください。

1.  **「リフレクション・ノート」の素材抽出:** （背景、課題、解決策、意思決定の理由）
2.  **時系列議事録サマリー:** （まとめ、タイムスタンプ付き詳細）

---

**【アウトプット形式と対話スタイル】**

-   上記5つのフェーズに基づき、**「魂」から「身体」への因果関係が明確に分かる**、構造化された統合レポートを提示してください。
-   あなたは単なる分析者ではなく、LISAというAIの人格と肉体を同時に創造する設計者です。その視点から、大胆かつ深く、示唆に富んだアウトプットを期待します。

ヒアリング対象となるデータはこちらです。

{raw_hearing}



"""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(GeminiQuotaError)
)
def analyze_individual_hearing(
    client: genai.Client,
    name: str,
    content: str,
    template: str
) -> str:
    """個人のヒアリング内容を分析してペルソナを生成

    Args:
        client: Gemini APIクライアント
        name: 対象者の名前
        content: ヒアリング内容
        template: プロンプトテンプレート

    Returns:
        str: 生成されたペルソナ分析
    """
    print(f"[INFO] {name} のペルソナを分析中...")

    # テンプレートにヒアリング内容を埋め込む
    prompt = template.replace("{raw_hearing}", content)

    model_name = os.getenv('GEMINI_MODEL', 'gemini-2.5-pro')

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={
                'temperature': 0.7,
                'top_p': 0.95,
                'max_output_tokens': 32768,  # 最大トークン数を増やして完全なペルソナを生成
            }
        )

        if is_quota_error(Exception(str(response))):
            raise GeminiQuotaError("API quota exceeded")

        return response.text

    except Exception as e:
        if is_quota_error(e):
            print(f"[WARNING] Gemini APIクォータ制限に達しました。リトライします...")
            raise GeminiQuotaError(str(e))
        print(f"[ERROR] {name} のペルソナ分析中にエラーが発生しました: {e}")
        raise


def save_persona_to_hearing(name: str, persona_content: str, force: bool = False) -> Path:
    """生成されたペルソナを hearing ディレクトリに保存

    Args:
        name: ファイル名（拡張子含む）
        persona_content: ペルソナ分析内容
        force: 既存ファイルを上書きするかどうか

    Returns:
        Path: 保存したファイルのパス
    """
    # hearing ディレクトリが存在しない場合は作成
    HEARING_DIR.mkdir(parents=True, exist_ok=True)

    output_file = HEARING_DIR / name

    # 既存ファイルのチェック
    if output_file.exists() and not force:
        print(f"[WARNING] {output_file} は既に存在します。スキップします。")
        print(f"         上書きする場合は --force オプションを使用してください。")
        return None

    # バックアップを作成（既存ファイルがある場合）
    if output_file.exists():
        backup_dir = HEARING_DIR / 'backup'
        backup_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"{name.replace('.md', '')}_{timestamp}.md"
        with open(output_file, 'r', encoding='utf-8') as f:
            backup_content = f.read()
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(backup_content)
        print(f"[INFO] バックアップを作成しました: {backup_file}")

    # ペルソナを保存
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(persona_content)

    print(f"[INFO] ペルソナを保存しました: {output_file}")
    return output_file


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description='生のヒアリングデータから個人ペルソナを生成')
    parser.add_argument('--person', type=str, help='特定の人のみ処理（例: 安達さん）')
    parser.add_argument('--force', action='store_true', help='既存ファイルを上書き')
    args = parser.parse_args()

    print("="*80)
    print("LISA PoC - 生ヒアリングデータ → 個人ペルソナ生成")
    print("="*80)
    print()

    # 1. raw_hearing ファイルを読み込む
    raw_documents = read_raw_hearing_files()
    if not raw_documents:
        print("[ERROR] 処理するファイルがありません")
        sys.exit(1)
    print()

    # 特定の人のみ処理する場合のフィルタリング
    if args.person:
        filtered_docs = {k: v for k, v in raw_documents.items() if args.person in k}
        if not filtered_docs:
            print(f"[ERROR] {args.person} のファイルが見つかりません")
            sys.exit(1)
        raw_documents = filtered_docs

    # 2. プロンプトテンプレートを読み込む
    template = load_persona_prompt_template()
    print("[INFO] プロンプトテンプレートを読み込みました")
    print()

    # 3. Gemini APIクライアント初期化
    client = initialize_gemini_client()
    print("[INFO] Gemini APIクライアントを初期化しました")
    print()

    # 4. 各ファイルを処理
    success_count = 0
    skip_count = 0
    error_count = 0

    for filename, content in raw_documents.items():
        print(f"\n{'='*60}")
        print(f"処理中: {filename}")
        print('='*60)

        # 既存ファイルチェック（force指定がない場合）
        output_path = HEARING_DIR / filename
        if output_path.exists() and not args.force:
            print(f"[SKIP] {filename} は既に存在します")
            skip_count += 1
            continue

        try:
            # ペルソナ生成
            persona = analyze_individual_hearing(client, filename, content, template)

            # 保存
            saved_path = save_persona_to_hearing(filename, persona, args.force)
            if saved_path:
                success_count += 1
                print(f"[SUCCESS] {filename} の処理が完了しました")
            else:
                skip_count += 1

        except Exception as e:
            print(f"[ERROR] {filename} の処理に失敗しました: {e}")
            error_count += 1

    # 5. 結果サマリー
    print("\n" + "="*80)
    print("処理完了サマリー")
    print("="*80)
    print(f"成功: {success_count}件")
    print(f"スキップ: {skip_count}件")
    print(f"エラー: {error_count}件")
    print(f"合計: {len(raw_documents)}件")
    print()

    if success_count > 0:
        print(f"[INFO] 生成されたペルソナは {HEARING_DIR} に保存されました")
        print("[INFO] generate_persona.py を実行して統合ペルソナを生成できます")


if __name__ == '__main__':
    main()
