# HW1: Learning to Identify High Betweenness Nodes
contributed by < `erickuo5124` >
###### tags: `MLG`
### 作業說明
給定一個 network，透過 GNN 找出圖中 BC(Betweenness Centrality) 較高的點，並達到以下要求：

- 計算出前 N% 點的正確率
- 算出預測所需時間

### BC (Betweenness Centrality)
在圖論中，所有最短路徑 (All-Pairs Shortest Paths) 穿越節點 $v$ 的數量即為該節點的 $v$ Betweenness Centrality，計算方式如下：

$$
g(v)=\sum_{s\neq v\neq t}\frac{\sigma_{st}(v)}{\sigma_{st}}
$$

其中 $\sigma_{st}$ 為節點 $s$ 到節點 $t$ 的最短路徑數量，$\sigma_{st}(v)$ 為經過節點 $v$ 路徑的數量。

計算所有節點的 Betweenness Centrality 需要計算所有最短路徑，目前最知名的計算方式為 [Brandes algorithm](http://www.uvm.edu/pdodds/research/papers/others/2001/brandes2001a.pdf)，時間複雜度在 unweighted networks 為 $O(|V||E|)$。

:::info
若是節點被很多最短路徑經過，該節點的 Betweenness Centrality 就越高，在網路中擔任的角色就相對重要，應該優先被保護或摧毀，進而控制網路的傳遞效率。
:::

### Dataset

- Synthetic Data (generate by [powerlaw_cluster_graph](https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.powerlaw_cluster_graph.html#networkx.generators.random_graphs.powerlaw_cluster_graph))
- Real world data (com-Youtube)

---
## DrBC
- [github](https://github.com/erickuo5124/MLG_HW1/blob/main/DrBC.ipynb)
- [Google colab](https://colab.research.google.com/drive/1XR6pDJ9WEfs7QOPKYuNgga1s38nOPMOH?usp=sharing)
### 環境

#### python 版本
```
Python 3.7.10
```

#### 套件
```shell=
torch 1.8.0+cu101
torch-geometric 1.6.3 
networkx 2.5
```

### 正確率
- Top-1%
```shell

```
- Top-5%
- Top-10%

### Hyper-parameter

|batch-size|embeding-dimension|learning-rate|layer|episodes|
|-|-|-|-|-|
|16|128|0.01|5|100|

----

### 模型實作

DrBC 的實作使用 encoder-decoder framework，將每個節點用 encoder 投影到空間中，Betweenness Centrality 相似的節點在空間中也會比較接近。再將空間中的點利用 decoder 量化成一個數值，該數值反映節點 Betweennes Centrality 在所有節點中的相對大小。

#### Network Embedding

把每個節點投影到三維空間，將 initial feature $X_v$ 設為：

$$
X_v=[d_v,1,1]
$$

其中 $d_v$ 為節點 $v$ 的 [degree](https://zh.wikipedia.org/wiki/%E5%BA%A6_(%E5%9B%BE%E8%AE%BA))，且 $v$ 的第0層 hidden layer $h^{(0)}_v$ 即為 $X_v$

#### Encoder - Neighborhood Aggregation

DrBC 論文中使用 weighted sum aggregator 來 aggregate 鄰居，將這些資訊做 embedding，而正好與 pytorch geometric 中的 [GCNConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv) 類似，函式實作如下：

$$
x'_i=\Theta\sum_{j\in N(v)\cup\{i\}}\frac{e_{j,i}}{\sqrt{\hat{d_j}\hat{d_i}}}x_j
$$

Neighborhood Aggregation 將鄰居的資訊聚合進節點當中，當神經網路疊得越多層，就能得到離節點越遠的資訊。

:::warning
GCNConv 與論文中的計算方式有些差異，在想可能是這裡有問題
:::

#### Encoder - COMBINE Function

為了得到節點 $v$ 在第 $l$ 層的 embedding，將 $v$ 上一層的 embedding $h^{l-1}_v$ 與 $v$ 所有鄰居 $N(v)$ 在 $l$ 層的聚合 $h^{l}_{N(v)}$ 加起來，使用到的 COMBINE Funtion 是 [GRUCell](https://pytorch.org/docs/stable/generated/torch.nn.GRUCell.html)，讓模型可以決定多遠距離鄰居的 feature 可以加到下一層內。

#### Encoder - Layer Aggregation

使用 max-pooling aggregator，在每個維度選擇最大的 feature 可以讓我們得到資訊最多的 layer，存為 $z$ 即為我們要的 embedding 的 Betweenness Centrality。

----

#### Decoder

用兩層的 Linear 算出節點的 BC ranking score：
$$
y=W_5ReLU(W_4z)
$$

#### Pairwise Ranking Loss
預測出來的值會是 BC ranking score，但實際目標並不是預測出真正 Betweenness Centrality 的值，而是"相對"的排名即可。因此把每對邊$(i, j)$ 節點預測值的差 $y_i-y_j$ 與實際值的差 $b_i-b_j$ 代進 [binary cross-entropy](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss) 裡得到 loss 值。

:::warning
實際執行時發現算出來的 loss 值會非常大，每次訓練 loss 變化的值卻很少，嘗試過調大 learning rate，但會出現梯度消失或梯度爆炸等問題，不太確定是不是 loss function 出錯還是前面有哪裡出問題😢
:::

----

### 訓練
#### Inductive Learning
論文中是利用遵守 power-law 合成的較小的圖來訓練，以在短時間內可以計算出正確的 Betweenness Centrality 值，再把訓練出來的模型套用到真實的較龐大的圖上。我用作業給的 Dataset 總共 30 張圖來訓練，每張圖共有 5000 個節點，因為不需要再另外計算 Betweenness Centrality，能加快訓練的速度。

:::warning
前幾次訓練的時候每個 epoch 得到的正確率大概從 10% ~ 40% 不等，但訓練到後面漸漸趨向大約 14%。
:::

:::danger
在想因為 Neighborhood Aggregation 是透過看圖上各節點的鄰居的 feature 來聚合，而每次餵的圖都不一樣，因此會得到不一樣的 feature，那會不會這個訓練方法並不能適用 Inductive Learning？
:::

---

## 參考資料
- [介數中心性- 維基百科，自由的百科全書 - Wikipedia](https://zh.wikipedia.org/wiki/%E4%BB%8B%E6%95%B0%E4%B8%AD%E5%BF%83%E6%80%A7)
- [Learning to Identify High Betweenness Centrality Nodes from Scratch: A Novel Graph Neural Network Approach](https://arxiv.org/abs/1905.10418v4)