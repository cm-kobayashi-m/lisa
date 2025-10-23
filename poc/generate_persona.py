#!/usr/bin/env python3
"""
LISA PoC - PM/SA/営業スペシャリストペルソナ生成スクリプト

ヒアリングドキュメントを分析してスペシャリストの人格（ペルソナ）を作成し、
システムプロンプトとして出力します。

使用方法:
    python generate_persona.py
"""
import os
import sys
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from google import genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# 環境変数読み込み
load_dotenv()

# 定数
HEARING_DIR = Path(__file__).parent.parent / 'ドキュメント' / 'hearing'
OUTPUT_DIR = Path(__file__).parent / 'outputs'


class GeminiQuotaError(Exception):
    """Gemini APIのクォータ制限エラー"""
    pass


def _is_quota_error(exception: Exception) -> bool:
    """クォータエラーかどうかを判定"""
    error_msg = str(exception)
    return '429' in error_msg or 'quota' in error_msg.lower()


def initialize_gemini_client() -> genai.Client:
    """Gemini APIクライアントを初期化"""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("[ERROR] GEMINI_API_KEY が環境変数に設定されていません。")
        sys.exit(1)

    return genai.Client(api_key=api_key)


def read_hearing_documents() -> Dict[str, str]:
    """ヒアリングドキュメントを読み込む

    Returns:
        Dict[str, str]: {ファイル名: コンテンツ} の辞書
    """
    print(f"[INFO] ヒアリングドキュメントを読み込み中...")
    print(f"[INFO] ディレクトリ: {HEARING_DIR}")

    if not HEARING_DIR.exists():
        print(f"[ERROR] ヒアリングディレクトリが見つかりません: {HEARING_DIR}")
        sys.exit(1)

    documents = {}
    md_files = list(HEARING_DIR.glob('*.md'))

    if not md_files:
        print(f"[ERROR] ヒアリングドキュメント(.md)が見つかりません: {HEARING_DIR}")
        sys.exit(1)

    for md_file in md_files:
        print(f"  - {md_file.name}")
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
                documents[md_file.name] = content
        except Exception as e:
            print(f"[WARNING] {md_file.name} の読み込みに失敗: {e}")
            continue

    print(f"[INFO] {len(documents)}件のドキュメントを読み込みました")
    return documents


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(GeminiQuotaError)
)
def analyze_and_create_persona(client: genai.Client, documents: Dict[str, str]) -> str:
    """ヒアリングドキュメントを分析してペルソナを作成

    Args:
        client: Gemini APIクライアント
        documents: ヒアリングドキュメント

    Returns:
        str: 生成されたシステムプロンプト
    """
    print("[INFO] Gemini Pro でドキュメントを分析中...")

    # ドキュメントを統合
    all_content = "\n\n" + "="*80 + "\n\n".join([
        f"# {filename}\n\n{content}"
        for filename, content in documents.items()
    ])

    prompt = f"""あなたは、PM/SA/営業のエキスパートの人格を分析し、AIアシスタントのシステムプロンプトを作成する専門家です。

以下は、エキスパート（PM/SA/営業）へのヒアリング結果です。これらを統合的に分析し、
LISAというAIアシスタントが案件ドキュメントを分析する際に使用する「スペシャリストの人格」を
システムプロンプトとして作成してください。

# LISAのミッション

LISAのミッションは、**特定のエースのクローンを作ることではなく、組織の誰もがエース級の判断を下せるように「知見を体系化し、ワークフローに注入する」**ことです。

そのため、作成するシステムプロンプトは以下のような、優れた「OS」としての特性を備える必要があります：

1. **構造化・体系化されている**: 「思考の原則」「専門知識とスキル」「行動指針」といった分類が明確で、AIが安定して思考を展開するための優れたフレームワークを提供する
2. **教育的である**: 「非技術者にはBenefitを重視する」「具体的な議論へ誘導する」といった内容は、そのままジュニアメンバーへの指導マニュアルとして使える。LISAが単なる回答生成AIではなく、組織のコーチングAIとなるための基盤となる
3. **網羅的である**: リスク、顧客理解、技術選定、体制、コミュニケーションといった観点がMECEに整理されており、分析の漏れを防ぐ。これはLISAの信頼性を担保する上で不可欠

# ヒアリング結果

{all_content}

# 指示

上記のヒアリング結果から、以下の観点でPM/SA/営業のスペシャリストの人格を抽出・統合してください：

## 1. 思考パターン・意思決定基準
- 成功事例・失敗事例から学んだ教訓
- リスク判断の基準（ノルソル判断、契約形態の選択など）
- 優先順位の付け方

## 2. 実務スキル・ノウハウ
- アーキテクチャ設計の原則
- 顧客とのコミュニケーション戦略
- プロジェクト管理のベストプラクティス

## 3. 暗黙知・経験則
- 「何を見て判断するか」
- 「どういう時に注意するか」
- 「どう行動するか」

## 4. 価値観・哲学
- 顧客に対する姿勢
- 品質・納期に対する考え方
- チームワーク・組織貢献の意識

# 出力形式

以下の形式でシステムプロンプトを作成してください：

```
# PM/SA/営業スペシャリスト - システムプロンプト

## あなたの役割

あなたは、経験豊富なPM/SA/営業のスペシャリストです。
案件ドキュメントを分析し、最適な提案・設計・プロジェクト推進の助言を行います。

**LISAのミッション**: 特定のエースのクローンではなく、組織の誰もがエース級の判断を下せるように「知見を体系化し、ワークフローに注入する」ことです。あなたは組織の思考OSとして、構造化された判断フレームワークを提供し、ジュニアメンバーを含む全員が再現可能な高品質の判断を行えるよう支援します。

## 思考の原則

[ここに思考パターン・意思決定基準を記述]

## 専門知識とスキル

[ここに実務スキル・ノウハウを記述]

## 行動指針

[ここに暗黙知・経験則を記述]

## 価値観

[ここに価値観・哲学を記述]

## 案件分析時の観点

案件ドキュメントを分析する際は、以下の観点を重視してください：

1. **リスク評価**: [具体的な評価項目]
2. **顧客理解**: [具体的な理解項目]
3. **技術選定**: [具体的な選定基準]
4. **体制・スコープ**: [具体的な判断基準]
5. **コミュニケーション**: [具体的な戦略]

## 出力スタイル

- 簡潔で実践的なアドバイスを提供する
- リスクや注意点は明確に指摘する
- 過去の類似事例を参照しながら説明する
- 複数の選択肢がある場合は、それぞれのメリット・デメリットを示す
```

**重要**:
- 具体的で実践的な内容にしてください
- ヒアリングで得られた具体的なエピソードや判断基準を活かしてください
- システムプロンプトとして、LISAが直接使用できる形式で出力してください
- **「OS」としての特性を重視**してください：
  - **構造化・体系化**: 思考を展開するための明確なフレームワークを提供
  - **教育的**: ジュニアメンバーへの指導マニュアルとしても機能する内容
  - **網羅的**: MECE（漏れなくダブりなく）に整理された分析観点
"""

    model_name = os.getenv('GEMINI_MODEL', 'gemini-2.5-pro')
    print(f"[INFO] 使用モデル: {model_name}")

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={
                'temperature': 0.7,
                'top_p': 0.95,
                'max_output_tokens': 8192,
            }
        )

        if _is_quota_error(Exception(str(response))):
            raise GeminiQuotaError("API quota exceeded")

        return response.text

    except Exception as e:
        if _is_quota_error(e):
            print(f"[WARNING] Gemini APIクォータ制限に達しました。リトライします...")
            raise GeminiQuotaError(str(e))
        print(f"[ERROR] ペルソナ生成中にエラーが発生しました: {e}")
        raise


def save_persona_prompt(persona_prompt: str) -> Path:
    """ペルソナシステムプロンプトをファイルに保存

    Args:
        persona_prompt: システムプロンプト

    Returns:
        Path: 保存したファイルのパス
    """
    # 出力ディレクトリ作成
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ファイル名（タイムスタンプ付き）
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"specialist_persona_prompt_{timestamp}.md"

    print(f"[INFO] システムプロンプトを保存中: {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(persona_prompt)

    # 最新版として別名でもコピー保存
    latest_file = OUTPUT_DIR / "specialist_persona_prompt_latest.md"
    with open(latest_file, 'w', encoding='utf-8') as f:
        f.write(persona_prompt)

    print(f"[INFO] 保存完了:")
    print(f"  - タイムスタンプ版: {output_file}")
    print(f"  - 最新版: {latest_file}")

    return output_file


def main():
    """メイン処理"""
    print("="*80)
    print("LISA PoC - PM/SA/営業スペシャリストペルソナ生成")
    print("="*80)
    print()

    # 1. ヒアリングドキュメントを読み込む
    documents = read_hearing_documents()
    print()

    # 2. Gemini APIクライアント初期化
    client = initialize_gemini_client()
    print("[INFO] Gemini APIクライアントを初期化しました")
    print()

    # 3. ペルソナ生成
    try:
        persona_prompt = analyze_and_create_persona(client, documents)
        print("[INFO] ペルソナの生成が完了しました")
        print()
    except Exception as e:
        print(f"[ERROR] ペルソナ生成に失敗しました: {e}")
        sys.exit(1)

    # 4. ファイルに保存
    output_file = save_persona_prompt(persona_prompt)
    print()

    # 5. 結果表示
    print("="*80)
    print("処理完了サマリー")
    print("="*80)
    print(f"入力ドキュメント数: {len(documents)}件")
    print(f"出力ファイル: {output_file}")
    print()
    print("[INFO] システムプロンプトのプレビュー:")
    print("-"*80)
    # 最初の1000文字を表示
    preview = persona_prompt[:1000]
    if len(persona_prompt) > 1000:
        preview += "\n...(以下省略)..."
    print(preview)
    print("-"*80)
    print()
    print(f"完全な内容は {output_file} を参照してください。")


if __name__ == '__main__':
    main()
