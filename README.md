# HW1: Learning to Identify High Betweenness Nodes

[![hackmd-github-sync-badge](https://hackmd.io/MesWdUafQJuapNoTI4AAPQ/badge)](https://hackmd.io/MesWdUafQJuapNoTI4AAPQ)

contributed by < `erickuo5124` >
###### tags: `MLG`
### 作業說明
給定一個 network，透過使用 GNN 實作 DrBC[^DrBCpaper] 找出圖中 BC(Betweenness Centrality) 較高的點，與其他計算方式(RK[^RKgithub], k-BC[^k-BCgithub], KADABRA[^KADABRAgithub])比較並重現 paper 中以下表格：

- Table 3：Top-N% accuracy on synthetic graphs of different scales
- Table 4：[Kendall tau distance](https://en.wikipedia.org/wiki/Kendall_tau_distance) on synthetic graphs
- Table 5：Runnung time on synthetic graphs
- Table 6：DrBC's generalization results on different scales (Top-N% accuracy)
- Table 7：DrBC's generalization results on different scales (Kendall tau distance)
- Table 8：Top-N% accuracy on real-world networks
- Table 9：Kendall tau distance on real-world networks

[^DrBCpaper]: [Learning to Identify High Betweenness Centrality Nodes from Scratch: A Novel Graph Neural Network Approach](https://arxiv.org/abs/1905.10418v4)
[^RKgithub]: [ecrc/BeBeCA](https://github.com/ecrc/BeBeCA)
[^k-BCgithub]: [ecrc/BeBeCA](https://github.com/ecrc/BeBeCA)
[^KADABRAgithub]: [natema/kadabra](https://github.com/natema/kadabra)

### BC (Betweenness Centrality)[^BCwiki]
在圖論中，所有最短路徑 (All-Pairs Shortest Paths) 穿越節點 $v$ 的數量即為該節點的 $v$ Betweenness Centrality，計算方式如下：

$$
g(v)=\sum_{s\neq v\neq t}\frac{\sigma_{st}(v)}{\sigma_{st}}
$$

其中 $\sigma_{st}$ 為節點 $s$ 到節點 $t$ 的最短路徑數量，$\sigma_{st}(v)$ 為經過節點 $v$ 路徑的數量。

計算所有節點的 Betweenness Centrality 需要計算所有最短路徑，目前最知名的計算方式為 [Brandes algorithm](http://www.uvm.edu/pdodds/research/papers/others/2001/brandes2001a.pdf)，時間複雜度在 unweighted networks 為 $O(|V||E|)$。

:::info
若是節點被很多最短路徑經過，該節點的 Betweenness Centrality 就越高，在網路中擔任的角色就相對重要，應該優先被保護或摧毀，進而控制網路的傳遞效率。
:::

[^BCwiki]: [介數中心性- 維基百科，自由的百科全書 - Wikipedia](https://zh.wikipedia.org/wiki/%E4%BB%8B%E6%95%B0%E4%B8%AD%E5%BF%83%E6%80%A7)

### Dataset

- Synthetic Data (generate by [powerlaw_cluster_graph](https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.powerlaw_cluster_graph.html#networkx.generators.random_graphs.powerlaw_cluster_graph))
    - 5000 nodes/graph
- Real world data (com-Youtube)


### 程式碼

- [github](https://github.com/erickuo5124/MLG_HW1/blob/main/DrBC.ipynb)
- [Google colab](https://colab.research.google.com/drive/1iykTh-oSPeVslapL_FKhTIdCYBKhhGyZ?usp=sharing)

---

## 環境

### python 版本
```
Python 3.7.10
```

### 套件
```shell=
torch 1.8.0+cu101
torch-geometric 1.6.3 
networkx 2.5
```

### Hyper-parameter

|batch-size|embeding-dimension|learning-rate|layer|episodes|
|-|-|-|-|-|
|16|128|0.0001|5|10000|

---

## 模型實作

DrBC 的實作使用 encoder-decoder framework，將每個節點用 encoder 投影到空間中，Betweenness Centrality 相似的節點在空間中也會比較接近。再將空間中的點利用 decoder 量化成一個數值，該數值反映節點 Betweennes Centrality 在所有節點中的相對大小。

### Network Embedding

把每個節點投影到三維空間，將 initial feature $X_v$ 設為：

$$
X_v=[d_v,1,1]
$$

其中 $d_v$ 為節點 $v$ 的 [degree](https://zh.wikipedia.org/wiki/%E5%BA%A6_(%E5%9B%BE%E8%AE%BA))，且 $v$ 的第0層 hidden layer $h^{(0)}_v$ 即為 $X_v$

### Encoder - Neighborhood Aggregation

DrBC 論文中使用 weighted sum aggregator 來 aggregate 鄰居，將這些資訊做 embedding，而正好與 pytorch geometric 中的 [GCNConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv) 類似，這裡是用乘法把資料合起來，函式實作如下：

$$
x'_i=\Theta\sum_{j\in N(v)\cup\{i\}}\frac{e_{j,i}}{\sqrt{\hat{d_j}\hat{d_i}}}x_j
$$

Neighborhood Aggregation 將鄰居的資訊聚合進節點當中，當神經網路疊得越多層，就能得到離節點越遠的資訊。

:::warning
GCNConv 與論文中的計算方式有些差異，不知道會不會是這裡有問題
:::

:::success
跟助教討論之後，認為這篇 paper 最重要的地方就是它的 Aggregation function，因此我利用 pytorch-geometric 的內建函式庫 [MessagePassing](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html) 手刻了作者的 Aggregation function：
$$
h_{N(v)}^{(l)}=\sum_{j\in N(v)}\frac{1}{\sqrt{d_v+1}\sqrt{d_j+1}}h_j^{(l-1)}
$$
但改過之後並沒有什麼太大的變化，不確定問題是不是出在這邊...
:::

### Encoder - COMBINE Function

為了得到節點 $v$ 在第 $l$ 層的 embedding，將 $v$ 上一層的 embedding $h^{l-1}_v$ 與 $v$ 所有鄰居 $N(v)$ 在 $l$ 層的聚合 $h^{l}_{N(v)}$ 加起來，使用到的 COMBINE Funtion 是 [GRUCell](https://pytorch.org/docs/stable/generated/torch.nn.GRUCell.html)，讓模型可以決定多遠距離鄰居的 feature 可以加到下一層內。

### Encoder - Layer Aggregation

使用 max-pooling aggregator，在每個維度選擇最大的 feature 可以讓我們得到資訊最多的 layer，存為 $z$ 即為我們要的 embedding 的 Betweenness Centrality。

:::success
在 paper 與 github 上的 max-pooling 使用方式有點差異，paper 上是將每層的 feature 記下來，跑完迴圈之後才做；而 github 上則是每經過一次 Aggregation 就做一次 max-pooling。我兩邊都嘗試過，但結果似乎沒有什麼太大的改變。
:::

----

### Decoder

用兩層的 Linear 算出節點的 BC ranking score：
$$
y=W_5ReLU(W_4z)
$$

:::success
在一次意外當中，我把輸出多加了一個 ReLU，正確率從原本的 20%~40% 直接飆升到 60%~70%，loss 沒有什麼太大的變化。

但我後來把預測的結果印出來看之後，發現大多數的值都是 0，也許是因為這些合成的圖 BC 值較大的點幾乎都在前面幾個，才會有正確值很高的結果。

BC 的值大多都在小數點兩位以後，但訓練出來的精確度似乎沒辦法到達那麼小，應該想辦法提高訓練的精確度。
:::

### Pairwise Ranking Loss
預測出來的值會是 BC ranking score，但實際目標並不是預測出真正 Betweenness Centrality 的值，而是"相對"的排名即可。因此把每對邊$(i, j)$ 節點預測值的差 $y_i-y_j$ 與實際值的差 $b_i-b_j$ 代進 [binary cross-entropy](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss) 裡得到 loss 值。

:::warning
實際執行時發現算出來的 loss 值很小，但每次訓練完經過 optimizer 的 loss 值變化卻很少，嘗試過調大 learning rate，但都會在 loss 大約等於某個值的時候卡住。

實際把 loss 值印出來發現，小數點的值會沒辦法往更精確的方向走，應該要往如何訓練更精確的數字下去走。
:::

:::success
跟助教討論後發現原先計算有點問題，我計算 loss 的時候只有考慮到所有有連接到的邊，如果是比較遠的兩點可能會有點問題。這裡 DrBC 的作者是用 random sampling 的方法找 $5|v|$ 對點來做計算，一開始做的時候沒有注意到這裡。

在我還沒有分 batch 的時候，一次訓練都是丟一張 5000 個節點的圖，使用這個 loss function 是沒什麼問題的。但當我分 batch 去訓練的時候就會變成每個 batch 的正確率很高，但在算一個 epoch 的正確率的時候就會變得很低，改成 random sampling 之後就可以解決這個問題了。
:::

---

## 訓練
#### Inductive Learning
論文中是利用遵守 power-law 合成的較小的圖來訓練，以在短時間內可以計算出正確的 Betweenness Centrality 值，再把訓練出來的模型套用到真實的較龐大的圖上。我用作業給的 Dataset 總共 30 張圖來訓練，每張圖共有 5000 個節點，因為不需要再另外計算 Betweenness Centrality，能加快訓練的速度。

:::warning
前幾次訓練的時候每個 epoch 得到的正確率大概從 0% ~ 10%，但訓練到後面頂多到 40% 就是極限了
:::

:::danger
在想因為 Neighborhood Aggregation 是透過看圖上各節點的鄰居的 feature 來聚合，而每次餵的圖都不一樣，因此會得到不一樣的 feature，那會不會這個訓練方法並不能適用 Inductive Learning？
:::

---

## 實驗結果

### Table 3
#### Top-1% accuracy on synthetic graphs of different scales
| Scale | RK | k-BC | KADABRA | DrBC |
| - | - | - | - | -|
| 5000|0.96|0.93|0.81|0.33|
| 10000|0.96|0.94|0.76|0.32|
| 20000|0.95|0.93|0.69|0.29|
|50000|0.93|0.92|0.68|0.30|
|100000|0.91|0.89|0.60|0.28|

#### Top-5% accuracy on synthetic graphs of different scales
| Scale | RK | k-BC | KADABRA | DrBC |
| - | - | - | - | -|
| 5000|0.96|0.90|0.72|0.32|
| 10000|0.95|0.89|0.72|0.36|
| 20000|0.92|0.84|0.68|0.26|
|50000|0.87|0.84|0.65|0.28|
|100000|0.87|0.82|0.60|0.26|

#### Top-10% accuracy on synthetic graphs of different scales
| Scale | RK | k-BC | KADABRA | DrBC |
| - | - | - | - | -|
| 5000|0.94|0.86|0.75|0.28|
| 10000|0.92|0.85|0.71|0.3|
| 20000|0.89|0.83|0.68|0.27|
|50000|0.88|0.78|0.61|0.28|
|100000|0.87|0.77|0.55|0.26|

### Table 4：Kendall tau distance on synthetic graphs

| Scale | RK | k-BC | KADABRA | DrBC |
| - | - | - | - | -|
| 5000|0.77|0.70||0.32|
| 10000|0.72|0.67||0.28|
| 20000|0.66|0.66||0.36|
|50000|0.54|0.66||0.27|
|100000|0.44|0.59||0.29|

### Table 5：Runnung time on synthetic graphs

| Scale | RK | k-BC | KADABRA | DrBC |
| - | - | - | - | -|
| 5000|18.1|15.3|0.5|0.3|
| 10000|19.5|55.7|1.2|0.5|
| 20000|45.6|182.3|1.5|1.2|
|50000|130.4|776.5|4.4|4.0|
|100000|345.0|4063.3|8.8|6.9|

### Table 6：DrBC's generalization results on different scales (Top-1% accuracy)

| Scale | 5000 | 10000| 20000 |50000| 100000 |
| - | - | - | - | -|-|
| 100_200|0.20|0.26|0.32|0.35|0.28|
| 200_300|0.28|0.33|0.31|0.30|0.23|
| 1000_1200|0.32|0.28|0.25|0.29|0.27|
|2000_3000|0.16|0.26|0.17|0.26|0.35|
|4000_5000|0.33|0.32|0.29|0.30|0.28|

### Table 7：DrBC's generalization results on different scales (Kendall tau distance)

| Scale | 5000 | 10000| 20000 |50000| 100000 |
| - | - | - | - | -|-|
| 100_200|0.17|0.26|0.22|0.33|0.26|
| 200_300|0.16|0.25|0.31|0.30|0.27|
| 1000_1200|0.32|0.28|0.35|0.33|0.27|
|2000_3000|0.19|0.25|0.25|0.26|0.35|
|4000_5000|0.32|0.28|0.36|0.27|0.29|

### Table 8
#### Top-1% accuracy on real-world networks
| RK |  KADABRA | DrBC |
| - |  - | - |
|0.78 |0.58|0.32|

#### Top-5% accuracy on real-world networks
| RK |  KADABRA | DrBC |
| - |  - | - |
|0.76 |0.0.46|0.28|

#### Top-10% accuracy on real-world networks
| RK |  KADABRA | DrBC |
| - |  - | - |
|1.0 |0.0.43|0.31|

### Table 9：Kendall tau distance on real-world networks

| RK |  KADABRA | DrBC |
| - | - | - | - | 
|0.12 ||0.28|

---

## insight

DrBC 使用圖論版本的 convolution 得到鄰居的 feature，加疊幾層之後經過數次訓練，得到一組能把節點投影座標空間內的參數，其中投影的點越接近則會得到越相似的 Betweenness Centrality。

而此模型使用 Inductive Learning，透過訓練由遵守 power-law 合成的小圖，來預測真實世界相對較大的圖，如此可以避免沒有 label 的資料，或需要長時間的計算的情況。

但我自己訓練的結果正確率並不高，基本上已經是完全照著 paper 的方式去做了，甚至是看著 code 來改寫。於是我統整出幾個可能出錯的地方：

- Neighbor Aggregation 的使用，但看起來 GNN 在這裡的差別沒有很多，同樣是 convolution 正確率不太可能差到那麼多。
- loss function 的選擇，最後得到的 loss 值都是 0.69 左右，訓練過程改變的範圍都是只有在 0.00001 以下，學習率的調整也沒有什麼幫助
- loss 的計算方式，按照 paper 中是用一個 edge 的兩個 node 的差別做計算，如此算出 "相對" 的 Betweenness centrality，但這個方式預測出來的值與實際的值差別其實蠻大的，在想也許應該找別種判斷標準
- inductive learning 的使用，雖然聽起來是蠻合理的，但是我自己分別訓練每張圖的時候正確率都會是穩定下降，而全部一起訓練的時候就會變成有點亂跳的感覺。
- 作者在論文中的方法與 github 帳號上的程式碼也有些出入，像：不是用 GRU 而是 graphsage 來作為 combine function

---

## 心得

第一次寫有關 GNN 的機器學習，之前有用 pytorch 寫過性格預測的模型，正確率至少都還有到 60%。但這次效果不如預期，不管什麼方式正確率大多維持在 20% ~ 40% 之間，甚至沒有做到 over-fitting，那應該就是我在中間哪一步沒有做對了。

第一個作業就花了比預想中還要多很多的時間，在看 pytorch-geometric 文件的時候才發現自己對 pytorch 好像沒有很熟；在回去看 pytorch 文件的時候才發現自己對機器學習其實沒有很熟；再回去看機器學習的時候才發現之前的數學都已經忘光光了。如此一來便花了很多時間在複習之前的東西上，實際寫程式的時間其實沒有很長。

這也是我第一次看 paper 看得這麼仔細，之前頂多就是找找關鍵的段落而已，但因為做不太出來，這次幾乎是把每一段都拿出來看了好幾遍，畫了很多重點做了很多筆記，相關 paper 也都載了不少篇，從第一次看的似懂非懂到現在已經可以說是一知"半"解了。

雖然說結果不是很理想，但也算是達到這次作業的目的：熟悉 GNN 了，大概能了解訓練流程是怎樣，GNN 與一般的 NN 差別在哪，也知道自己的基礎能力還有哪些得要去加強，希望之後的作業能有更好的成果。