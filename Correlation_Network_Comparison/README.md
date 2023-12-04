## Correlation-based network comparison
条件Aと条件Bの相関ネットワークを比較するシナリオ。
相関係数の「差」の検定を明示的に行う必要がある。

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
とすることで、以下の式が正規分布に従うことが知られる。
```math
\begin{flalign}
  &\sqrt{n-3} (z-\eta)&
\end{flalign}
```

また、各条件での相関係数の比較は、以下の式で求められるZスコアを用いて検定することができる。
```math
\begin{flalign}
  &Z_{AB} = \frac{Z_A(x_i,x_j) - Z_B(x_i,x_j)}{\sqrt{\frac{1}{n_A - 3} + \frac{1}{n_B - 3}}}&
\end{flalign}
```


## References
- 竹本 和広 (2021). 生物ネットワーク解析　コロナ社
- https://qiita.com/c60evaporator/items/6510a6440eb396862dac
