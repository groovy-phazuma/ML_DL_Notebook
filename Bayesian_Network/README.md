## Bayesian Networks

### Structure Learning
■ **Datasets**
- [wine dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html)

■ **Methods**
1. PC
2. HillClimbSearch
   - BicScore
   - K2Score
4. TreeSearch
5. ExhaustiveSearch
6. MmhcEstimator

■ **Run Time Comparison**

Methods Exhaustive and Mmhc were not measured because they were very time consuming.

| Methods | PC         | HC (BicScore) | HC (K2Score) | Tree | Exhaustive  | Mmhc       |
| --------| ---------- |  ---------- |  ---------- |  ---------- |  ---------- | ---------- | 
| Time    | 18.4 sec  |  7.0 sec  | 13.5 sec  |  0.70 sec  |  --- sec  |  --- sec |
