# VSCodeで快適な開発環境を整えよう！
更新日: 2024/04/03

VSCode（Visual Studio Code）での快適な開発環境の整え方を紹介します。さっそく見ていきましょう。

## 目次
- [共通機能の設定](#共通機能の設定)
- [Python実行関連の設定](#python実行関連の設定)

## 共通機能の設定
### 1. カラーテーマを変更しよう！(⭐⭐⭐)
色々試してみましたが、やっぱり定番の[One Dark Pro](https://marketplace.visualstudio.com/items?itemName=zhuangtongfa.Material-theme)が最高です。
Darker, Flat, Mixなどがありますが、デフォルトで十分満足です。

![image](https://github.com/groovy-phazuma/ML_DL_Notebook/assets/92911852/3b28c2dc-efd7-4c5c-8769-56393e2b9bcc)

### 2. フォルダのインデントを広げよう！(⭐⭐)
デフォルトのままではインデント表示がちょっと狭くて見づらいですね。

```Workbench › Tree: Indent```を*20*に設定すると、すっきり見やすくなります。
[この辺りの記事](https://zenn.dev/ayatokura/articles/vscode-article-7)が参考になりました。

![image](https://github.com/groovy-phazuma/ML_DL_Notebook/assets/92911852/7dcc1a7a-a386-48ba-b3b2-972b4ab0f8d2)

### 3. 折り返しで見やすく！(⭐⭐)
文字が長すぎると見づらいですね。

```Editor: Word Wrap```を*on*にして折り返してみてください。

### 4. 爆発でテンションを上げよう！(⭐)
普通にコーディングしてもつまらないですよね。[Power Mode](https://marketplace.visualstudio.com/items?itemName=hoovercj.vscode-power-mode)を用いて派手にいきましょう。

私はこの、"Magic"という設定を愛用しています。少し見づらいかも...

<img src="https://github.com/groovy-phazuma/ML_DL_Notebook/assets/92911852/0afea6ee-4410-4770-ab9d-8d9991dc0296" width=600>

<br>

## Python実行関連の設定
### 1. Jupyterを使いこなそう！(⭐⭐)
```Jupyter › Interactive Window › Text Editor: Execute Selection```の設定を有効にすると、選択した箇所でshift+enterで実行できるようになります。デバッグ時にもとても便利です。

### 2. Snippetsを準備して効率アップ！(⭐⭐)
左下の歯車から```User Snippets```を選択し、python.jsonファイルを編集しましょう。
```json
{
  "sample1": {
  "prefix": "get_start",
  "body": [
      "# -*- coding: utf-8 -*-",
      "\"\"\"",
      "Created on $CURRENT_YEAR-$CURRENT_MONTH-$CURRENT_DATE ($CURRENT_DAY_NAME_SHORT) $CURRENT_HOUR:$CURRENT_MINUTE:$CURRENT_SECOND",
      "",
      "@author: I.Azuma",
      "\"\"\""
      ]
  }
}
```
```get_start```とスクリプトに入力すると、打刻とauthor情報を記載することができます。こんな感じ。
```Python
# -*- coding: utf-8 -*-
"""
Created on 2024-02-28 (Wed) 17:13:50

@author: I.Azuma
"""
```
### 3. autoDocstringで楽々Docs作成！(⭐⭐)
Docstringを自動で生成できる[autoDocstring](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring)をインストールしましょう。

<img src="https://github.com/groovy-phazuma/ML_DL_Notebook/assets/92911852/150939ce-75cb-4d23-85a7-8af2ddd71297.gif" width=600>


