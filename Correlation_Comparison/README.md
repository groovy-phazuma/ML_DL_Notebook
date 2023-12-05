## Correlation Comparison
条件Aと条件Bの相関ネットワークを比較するシナリオ。相関係数の「差」の検定を明示的に行う必要がある。

<br>

#### フィッシャー変換を用いた検定 (```Fisher_transformation.py```)
***
フィッシャー変換は次式のように表すことができる。
```math
\begin{flalign}
  &Z_A(x_i,x_j) = \frac{1}{2} \ln \frac{1 + Cor_A (x_i,x_j)}{1 - Cor_A (x_i,x_j)}&
\end{flalign}
```
ここで、母相関係数ρについても同様に、
```math
\begin{flalign}
  &\eta = \frac{1}{2} \ln \frac{1 + \rho}{1 - \rho}&
\end{flalign}
```
とすることで、$`\sqrt{n-3} (z-\eta)`$ が正規分布に従うことが知られる。


また、各条件での相関係数の比較は、以下の式で求められるZスコアを用いて検定し、*p*値を算出することができる。
```math
\begin{flalign}
  &Z_{AB}(x_i,x_j) = \frac{Z_A(x_i,x_j) - Z_B(x_i,x_j)}{\sqrt{\frac{1}{n_A - 3} + \frac{1}{n_B - 3}}}&
\end{flalign}
```

<br>

#### パーミュテーション検定 (```permutation_test.py```)
***
さて、上記の*p*値の推定は、$`Z_{AB}(x_i,x_j)`$ が正規分布に従うことを仮定しているが、場合によっては成り立たない。
この場合はpermutation testを行い、経験的*p*値を算出する手法が有用。統計量の順位に着目することで、統計量の分布に依存せずに*p*値を算出することができる。

<br>

## References
- 竹本 和広 (2021). 生物ネットワーク解析　コロナ社
- https://qiita.com/c60evaporator/items/6510a6440eb396862dac
