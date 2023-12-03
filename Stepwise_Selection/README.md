## Stepwise Feature Selection
1. 変数増加法（forward stepwise selection）：すべて含むモデルからスタートし、1つずつ変数を減少
2. 変数減少法（backward stepwise selection）：含まないモデルからスタートし、1つずつ変数を増加
3. 変数増減法（forward-backward stepwise selection）：すべて含むモデルからスタートし、1つずつ変数を増加させたり減少させたりする
5. 変数減増法（backward-forward stepwise selection）：含まないモデルからスタートし、1つずつ変数を増加させたり減少させたりする

## Sample Codes
- backward_stepwise_AIC.py
  - AICを評価関数とする変数増加法
- forward_stepwise_AIC.py
  - AICを評価関数とする変数減少法
- sklearn_rfe.py
  - RFE (Recursive Feature Elimination) による特徴量選択

## References
- [AIC, BICを用いた実装](https://mimikousi.com/forward-stepwise-regression/)
- [各手法の解説](https://bellcurve.jp/statistics/glossary/957.html#:~:text=%E5%A2%97%E5%8A%A0%E6%B3%95%EF%BC%9A%E8%AA%AC%E6%98%8E%E5%A4%89%E6%95%B0%E3%82%92,%E6%B8%9B%E5%B0%91%E3%81%95%E3%81%9B%E3%81%9F%E3%82%8A%E3%81%99%E3%82%8B%E6%96%B9%E6%B3%95%E3%80%82)
