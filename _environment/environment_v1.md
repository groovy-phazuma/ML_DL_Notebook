# VSCode 環境構築
更新日: 2024/02/29

- dockerを立てた後の環境構築備忘録
- 快適な環境を構築しよう

## 各種拡張機能のインストール
- Python
- Jupyter
- One Dark Pro
- autoDocstring
- Todo Tree

## 共通機能の設定
### 1. Color Themeの変更
色々試したが、定番の[One Dark Pro](https://marketplace.visualstudio.com/items?itemName=zhuangtongfa.Material-theme)が結局良かった。
Darker, Flat, Mixなどがあるが、デフォルトで良い。

![image](https://github.com/groovy-phazuma/ML_DL_Notebook/assets/92911852/3b28c2dc-efd7-4c5c-8769-56393e2b9bcc)

### 2. Tree Indentを広げる
デフォルトだとインデント表示幅が狭くて見づらい。```Workbench › Tree: Indent```を*20*に設定。
[この辺りの記事](https://zenn.dev/ayatokura/articles/vscode-article-7)を参考に。

![image](https://github.com/groovy-phazuma/ML_DL_Notebook/assets/92911852/7dcc1a7a-a386-48ba-b3b2-972b4ab0f8d2)

### 3. Word Wrapをonにする
文字を折り返して見やすくする。```Editor: Word Wrap```を*on*に設定。

<br>

## Python実行関連の設定
### 1. Jupyterの実行設定
```Jupyter › Interactive Window › Text Editor: Execute Selection```の設定に☑を入れる。

これにより、選択した箇所でshift+enterを押すことでjupyterが実行される。デバッグなどで便利。

### 2. Snippetsの準備
左下の歯車から```User Snippets```を選択し、python.jsonファイルを編集する。
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
以上のように編集することで、```get_start```とpythonスクリプトに入力すると、打刻とauthor情報を記載することができる。
```Python
# -*- coding: utf-8 -*-
"""
Created on 2024-02-28 (Wed) 17:13:50

@author: I.Azuma
"""
```

### 3. autoDocstringのインストール
Docstringを自動で生成できて便利。[こちら](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring)からインストール。

<img src="https://github.com/groovy-phazuma/ML_DL_Notebook/assets/92911852/150939ce-75cb-4d23-85a7-8af2ddd71297.gif" width=800>
