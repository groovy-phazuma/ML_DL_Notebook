# VSCode 環境構築
更新日: 2024/01/24

- dockerを立てた後の環境構築備忘録
- 快適な環境を構築しよう

## 各種拡張機能のインストール
- Python
- Jupyter
- One Dark Pro
- autoDocstring
- Todo Tree

## Jupyterの実行設定
```Jupyter › Interactive Window › Text Editor: Execute Selection```の設定に☑を入れる。

これにより、選択した箇所でshift+enterを押すことでjupyterが実行される。デバッグなどで便利。

## Color Themeの変更
色々試したが、定番の[One Dark Pro](https://marketplace.visualstudio.com/items?itemName=zhuangtongfa.Material-theme)が結局良かった。
Darker, Flat, Mixなどがあるが、デフォルトで良い。

![image](https://github.com/groovy-phazuma/ML_DL_Notebook/assets/92911852/3b28c2dc-efd7-4c5c-8769-56393e2b9bcc)

## Tree Indentを広げる
デフォルトだとインデント表示幅が狭くて見づらい。```Workbench › Tree: Indent```を*20*に設定。
[この辺りの記事](https://zenn.dev/ayatokura/articles/vscode-article-7)を参考に。

![image](https://github.com/groovy-phazuma/ML_DL_Notebook/assets/92911852/7dcc1a7a-a386-48ba-b3b2-972b4ab0f8d2)

## Word Wrapをonにする
文字を折り返して見やすくする。```Editor: Word Wrap```を*on*に設定。

##
