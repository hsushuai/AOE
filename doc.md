# 大语言模型在对手建模中的应用研究

> SAP: Adaptive Competitor Exploration with Integrated Multi-strategies for Improved Planning in Large Language Models

> IM SAP: Integrated Multi-strategies with Adaptive Competitor Exploration for Improved Planning in Large Language Models

## 研究目标：

| 博弈场景下策略生成难点 | 解决方案 |
| :---: | :---: |
| 强化学习训练策略收敛慢且缺乏高质量对手 | LLM 生成高质量对手及响应策略 |
| 单一策略无法应对复杂多变的环境 | 根据策略特征生成多样化策略 |
| Policy Net 缺乏可解释性 | LLM 生成自然语言描述的策略 |
| 对手具有混合策略 | 实时识别对手策略特征并调整策略 |
| 对手策略 unseen | 网格搜索寻找高 payoff 的特征参数 |

> Pre-match：生成多种对手策略及反制策略，构建策略库
> 
> In-match：实时数据分析并调整策略
>
> Post-match：总结数据，提取有效策略

## Pre-match：多样对手策略及响应策略生成

> 目标：生成多样化的对手策略及反制策略，构建策略库（策略特征 + 规划方案）
>
> 手工设计策略特征 -> (LLM 生成对手策略 -> LLM 生成反制策略 -> 迭代优化) -> 策略库

- [x] 策略特征（手工设计特征）
- [ ] LLM 生成多种对手策略
- [ ] LLM 针对每个对手策略生成反制策略
- [ ] 迭代优化：反思（对手策略 + 反制策略 + 轨迹摘要 + 评估指标） -> 好的反制策略
- [ ] 评估指标：win/loss + 输出伤害 + 资源利用
- [ ] 策略库：策略 + 规划方案（在线时作为 few-shot）

### 策略特征

> 目标：（1）能够表示出策略；（2）区分不同策略；（3）能够根据轨迹识别

```markdown
- 经济特征
    - 参数：采矿工人数量
    - 特征空间：{1， 2}

- 兵营特征
    - 参数：建造时机（资源数量）5<= N <= 10
    - 特征空间：{resource >= N，False}

- 兵种特征
    - 参数：兵种组合
    - 特征空间：{Worker，Ranged and Light, ...}

- 侵略性特征
    - 参数：侵略性
    - 特征空间：{True，False}  # {侵略型，防守型}

- 攻击特征
    - 参数：优先攻击目标
    - 特征空间：{Ranged > Worker...}

- 防守特征
    - 参数：防守区域
    - 特征空间：{(x1, y1)-(x2, y2)，...}
```

### 策略生成

> 目标：生成高质量的对手策略和反制策略

```markdown
## 策略
- 经济策略: 2 人采矿
- 兵营策略: player.resource >= 6  # 资源 >= 6 时建造兵营
- 兵种策略: Ranged or Worker Rush（没兵营时，优先生产工人；有兵营时，优先生产远程）
- 侵略性特征: True  # 侵略型
- 攻击策略: 侵略性为 True 时，使用兵种策略指定的兵种优先攻击工人
- 防守策略: 侵略性为 False 时，优先部署防守区域 (3, 3) - (3, 5)

## 描述
...
```

### 规划方案

> 目标：生成符合策略且符合态势的方案

```python
# 两人采矿
[Harvest Mineral](0, 0)
[Harvest Mineral](0, 0)

# 优先攻击敌方工人
[Attack Enemy](worker, worker)
[Attack Enemy](worker, worker)
[Attack Enemy](worker, worker)
[Attack Enemy](worker, worker)

# 资源 >= 6 时建造兵营
[Build Building](barracks, (0, 3), (player.resource >= 6))

# 没兵营时，生产工人
[Produce Unit](worker, east)
[Produce Unit](worker, south)
[Produce Unit](worker, east)
[Produce Unit](worker, south)
```

### Payoff 设计

> 目标：评估方案质量

```markdown
- 资源平衡：消耗 - 收集
- 伤害输出
- win/lose/draw：10/-10/0
```

$$\text{payoff} = \text{win\_loss} + \alpha \cdot \text{damage\_dealt} + \beta \cdot \text{resources\_balance}\text{, where } \text{resources\_balance} = \text{resources\_spent} - \text{resources\_harvested}$$

### 轨迹特征提取

```markdown
- 采矿人数（经济策略）
- 兵营建造时间，资源，位置（兵营策略）
- 兵种数量（兵种策略）
- 攻击位置，类型（攻击策略）
- 位置热力图（防守策略）
```

```yaml
economic: 2
barracks:
  - time: 370
    resources: 11
    location: (6, 5)
military:
  - worker: 9
    heavy: 0
    light: 0
    ranged: 0
attack:
  - (2, 2): worker
    (1, 3): worker
    (1, 2): base
    # ...
position:
  - (6, 6): 155
    (7, 6): 197
    (7, 5): 121
    # ...
```

## 2 In-match：对手策略自适应

> 目标：根据对手轨迹，自适应调整策略（每 100 步，调整一次）

1.  提取轨迹特征
2.  LLM 分析对手策略
3.  策略复用/策略生成（参数搜索）

## 3 Post-match：策略优化

> 真实的对手更有价值
> 
> 目标：根据真实的对手优化策略（每局更新一次）

- [ ] 同一个对手，多次对局，每局结束后反思优化针对这个对手的策略

---

## 实验设置

| | MSGO | RL-based | LLM-based |
| - | -  | -  | - |
| RL-based |  |  |  |
| LLM-based |  |  |  |
| SAP |  |  |  |

## 悬而未决的问题

- [x] 长长长的轨迹 -> 关键事件提取出轨迹特征
