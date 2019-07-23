# 摘要
通过双向文本预训练模式，以BERT为代表的基于**自编码（autoencoding）**的预训练模型在各类任务上表现突出，超越了传统的单向训练的**自回归（autoregressive）**语言模型。但由于使用mask进行训练，输入与下游任务的**fine-tunning**不一致，这种不一致使得BERT在生成类任务上相对较弱。
综合考虑**自编码（autoencoding）**模型BERT与**自回归（autoregressive）**模型transformer-XL的优缺点，作者提出了**XLNet**，它具有以下优势：
- 能够学习双向的文本信息，得到最大化期望似然（对所有随机排列，解决mask导致的问题）
- 使用自回归的方程解决BERT存在的问题，同时将最好的自回归模型transformer-XL的思想应用于pre-train中
最终，XLNet在20个NLP任务中打败large-BERT，其中18个取得了state-of-the-art。
### 1 介绍
**传统无监督表征学习（Unsupervised representation learning）**被成功应用在大量无标注文本的预训练上，其中AR（自回归）与AE（自编码）是最成功的两种预训练语言模型。
**AR方法**学习文本序列的条件概率，即学习已知上文预测下一个词或一直下文预测上一个词。但是这样预训练的语言模型与一些不需要**“重写”（rewrite）**下游任务不匹配，如情感分析，摘要提取等。
**AE方法**则重构数据，mask部分token，用上下文的双向信息来预测被mask的token，由于训练时使用了【MASK】，而在下游任务中input并没有【MASK】，因而存在**pretrain-finetune discrepancy**，同时BERT同时mask15%的token这个操作实际上做了一个**有风险的假设**：假设每一个被mask的token之间是相互独立的。
XLNet综合AE与AR的优点，同时在一定程度上避免了两者的缺点：
- 通过打乱文本sequence的token顺序，使得用前n-1的token来预测第n个token时，同时能用到上下文的信息。对所有可能的token排列（all possible permutations of the factorization order）来求期望的对数似然函数。
- XLNet是一种广泛意义上的AR模型，每次只预测一个token，所以避免了mask多个token而难以判断被mask的多个token之间的联系。

同时，XLNet使用了一些小trick，改善了Transformer-XL的一些问题：

- 使用片段复发机制（segment recurrence mechanism），使得它在长文本的任务中表现更好。
- 直接使用Transformer-XL来训练基于排序的语言模型（permutation-based language model）是不行的，会得到不确定的结果，作者重新设置了Transformer-XL的网络与参数，去除了这种随机性。

**相关工作**
基于排序的语言模（permutation-based language model）的想法已经被其他人提出过，但是他们的模型缺乏位置编positional encodings，目标是通过无序的模型来提高估计效果，但作者的模型则是用来学习双向的文本信息（bidirectional contexts），提高预训练与下游任务的匹配。

### 2 方法的提出
#####  2.1 背景
AR方法可以直接求最大化对数似然，可以逐步拆解条件概率，公式如下：
![f1](图片链接地址)
AE方法通过随机将一定比例（常为15%）的token换成【MASK】，求上下文已知情况下，被MASK的词的条件概率，其中必须假设被msak的词之间互不相关，才能将其拆开，公司如下：
![f2](图片链接地址)
作者从独立性假设（Independence Assumption）、输入噪音（Input noise）、文本依赖（Context dependency）三个方面比较了两种方法的优缺点。
#####  2.2 目标：PLM（Permutation Language Modeling）
![p1](图片链接地址)
简单来说，就是一种广义的AR方法（避免了AE的不合理的独立性假设），通过重新排序的方法，使得预训练得到双向的文本信息。但为了使输入与其后的fine-tunning阶段相同，重新排序的过程在transformer层内部实现，具体结构在2.3节有详细说明。
举个例子，比如输入为（x1，x2，x3，x4），重新排序有4！种情况。按照传统AR的方法的方法从左到右预测的话，只能通过x1，x2预测x3，则损失了x4的信息，如果得到的排序是（x4，x1，x3，x2）的话，则通过x4，x1来预测x3。当然句子长度非常长时，是不可能穷举的，通过抽样的方法来求期望，并最大化以下似然函数：
![f3](图片链接地址)
在具体实现过程中，作者采用了特别的结果以解决使用transformer训练PLM的问题。
#####  2.3 结构：双流自注意力
由于PLM任务有特殊的性质，简单的Transformer无法有效工作。由于随机排列求期望的影响，简单应用标准Softmax方程求期望时，$h_{\theta}(x_{z<t})$就不依赖于位置信息（随机打乱求期望，实际上抹杀了位置信息的影响），导致模型无法学习到有用的表征。以下第一个公式为普通transformer应用的公式，第二个公式为作者为了PLM任务设计的公式。
![f3.5](图片链接地址)
![f4](图片链接地址)
仔细考虑Transformer的结构，当它用来解决PLM预训练任务时必须解决以下两个问题：
- 当预测$x_{z_t}$时只使用位置信息$z_t$和上文信息，不能看到自己本身，即$x_{z_t}$，否则求条件概率$g_{\theta}(x_{z<t},z_t)$的问题将会变成一个平凡的问题。
- 当预测$x_{z_j}$时，其中j>t，则能看到$x_{z_t}$信息。
考虑具体的注意力机制，只需在QUERY矩阵上做文章即可，当QUERY矩阵没有对角线时，显然是无法看到自己的。根据这样的思路，作者设计了双流自注意力模型，query流与content流共享参数$\theta$，但query流QUERY矩阵没有对角线。为了实现排序效果，query流必须初始化为可训练的向量即，n行必须分别为0，1，2，......，n-1个点，且符合贝叶斯网络，即相当于实现只看到特定排序的mask矩阵。以下为更新公式与网络示意图：
![f5](图片链接地址)
![p2](图片链接地址)
**部分预测**当t比较小，仅得到t-1个token的信息，这样的预测比较难，很难收敛，所以模型设计了一个超参数K，只对后1/K的tokens进行预测。
#####  2.4 2.5 借鉴Transformer-XL
作者团队半年前发表的Transformer-XL中使用了很多有效的trick，使得Transformer-XL做到了AR模型中的state-of-the-art。本文中也借鉴使用了相对位置编码和片段复发机制分别解决绝对位置编码无法处理的2个以上文本对应输入的task和算法效率问题，详见Transformer-XL文章。
2.5主要具体阐述了Multiple Segments无法用绝对位置编码解决的问题

##### 2.6.1 与BERT对比
文章举了一个简单的例子，预训练中使用New York is a city。加入New和York为预测目标，由于BERT做了独立性假设，将两者同时mask住，所以只能通过is a city分别对两个词进行预测，而XLNet由于特殊的重新排序求期望的操作，使得XLNet在预测出New后，在预测York时能得到New的信息。
##### 2.6.2与其他语言模型对比
一种AR标准模型GPT只建立了下文对上午的依赖，也就是说根据New能预测York，但根据York无法预测New。在解决阅读理解task时，如果给了内容Thom Yorke is the singer of Radiohead，但问的问题是Who is the singer of Radiohead，经典的AR模型是不能根据Radiohead预测出Thom Yorke的。
###3 实验部分
这一部分不再详细解释，但是值得一提的是BERT-base使用的是经典的13G数据集进行预训练，XLNet-base使用的是GPT 2.0构建的优质的19G数据集进行训练，大部分比较中XLNet大幅度超越BERT其中存在水分，最后一个实验中抹平数据和trick进行比较，提升幅度虽然有，但就不那么明显了，说明XLNet使用的PLM方法虽然有用，但贡献没有大家想象的夸张，XLNet能取得优异的成果，还在于结合了BERT出现后1年来发展的各种trick和数据质量的提升。

# 论文地址与代码地址
论文：https://arxiv.org/pdf/1906.08237.pdf
代码：https://github.com/zihangdai/xlnet
公众号：六点一刻研习室
（兴趣使然进行更新，没有质量保障）
