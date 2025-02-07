# GitHub Copilotを使ってみた
更新日: 2024/09/09

## VS Codeへの導入
- 拡張機能で「copilot」と検索すると出てくるので、素直にインストールする。
- 認証があるので指示に従って進める。

## code-serverへの導入
- code-serverでは拡張機能でcopilotが検索結果として表示されない。
- 拡張機能をコマンドで取得することを試みる。
  - 失敗：```>> curl -L -o github.copilot.vsix https://github.com/github/copilot-vscode/releases/latest/download/github.copilot-1.226.0.vsix```だと、中身が破損しておりインストールできない。
  - 成功：ローカルでDLした1.226.0を計算機に直接移行する。[ここから](https://marketplace.visualstudio.com/items?itemName=GitHub.copilot)バージョンを指定してDL。今回はGitHub.copilot-1.226.0.vsixを選択。
- ```>> code-server --install-extension GitHub.copilot-1.226.0.vsix```で実行
  - 失敗：かなり昔に立てたcode-server (v=1.76.1)だと```Unable to install extension 'github.copilot' as it is not compatible with VS Code '1.76.1'.```のエラー。ダウングレードすると入りそう。
  - 成功：code-serverのversionが1.88.1以上で正常にインストールされる
- 特に認証のステップはなかった。git configのglobalの設定の有無とか？？
- GitHub Copilot Chatは入れることができない...(240911)。

**※ 注意**: 現状はdocerfile内に以下を記載してcode-serverを取得しており、特にバージョンを指定していない。
```
# RUN apt-get update && apt-get install -y curl
RUN curl -fsSL https://code-server.dev/install.sh | sh
```

## 導入後の設定メモ
##### 1. Copilotの表示文字の設定
- デフォルトの設定では、Copilotの推薦文と既に存在するコメントアウトの色が区別しづらい。
- Preferences: Open *Remote* Settings (JSON) を開いて、以下のように編集する。
```
{
    "workbench.colorCustomizations": {
        "editorGhostText.border": "#ffb871dc",
        "editorGhostText.foreground": "#B0B0B0",
    }
}
```

##### 2. 使いやすいショートカットを探求する。
- suggestionsの一覧が ```Ctrl + Enter```に割り当てられているが、JupyterNotebookの実行と干渉するので```Shift + Esc```などに変更。