# 2026.2.19 - Phase 1: Bare-Metal Tensors

## 1. 核心概念 (Memory & Views)
- **物理真理**: 内存是一维的 (`std::vector<float> data`)。
- **数学幻觉**: 用 `shape` (`rows, cols`) 和 `stride` 逻辑来模拟多维结构。
- **Row-Major Indexing**: $Index = r \times cols + c$。

## 2. C++ 工程细节
- **Const Correctness**:
    - `float& operator()`: 用于写入 (Read-Write)。
    - `float operator() const`: 用于只读访问 (Read-Only)，承诺不修改内存。
- **Operator Overloading**: 让矩阵运算像数学公式一样直观（ `A + B` ）。
- **Reference**: 能够直接操作内存地址，避免不必要的拷贝。

## 3. 性能优化 (Memory Access Pattern)
- **Matrix Multiplication (MatMul)**: $C_{ij} = \sum_k A_{ik} \times B_{kj}$。
- **Cache Miss**: 直接访问 $B_{kj}$ (列遍历) 会导致内存跳跃，缓存未命中率高。
- **Transpose Hint**: 先计算 $B^T$，将列遍历转换为行遍历 ($B^T_{jk}$)，利用 **Spatial Locality** (空间局部性) 极大提升 CPU 效率。

## 4. 广播机制 (Broadcasting)
- **物理本质 (Zero-Copy 幻觉)**: 广播不需要在内存中实际复制一份扩展后的数据。
- **内存复用逻辑**: 在运算迭代中锁死对应维度的指针偏移。例如在矩阵相加中，当对偏置行向量进行相加时，直接强制使其行指针复位（`other(other.rows == 1 ? 0 : i, j)`），以此实现低维度向高维度的虚拟展开映射。

---

# 2026.2.21 - Phase 2: Demystifying the Gradient (Custom Training Loop)

## 1. 损失函数 (Loss Function)
- **工程意义**: 预测值与真实传感器观测值之间的“标尺”。
- **均方误差 (MSE)**: 用于回归任务，由于平方项的存在，它能成倍放大较大的误差，对系统的预测偏离具有敏感性警报作用。公式：$Loss = (y_{pred} - y_{true})^2$

## 2. 偏导数与链式法则 (Partial Derivatives & Chain Rule)
- **梯度的物理映射**: “Loss 对权重 $w$ 求偏导”就是测量“拧动 $w$ 这个旋钮 1 个单位时，Loss 是变大还是变小”。符号（正负）提供了参数优化的**方向**，绝对值（大小）提供了**陡峭程度**。
- **核心剥洋葱求导**: 对于 $Loss = (w \cdot x + b - y_{true})^2$，根据链式法则，$w$ 的偏导数为 $2 \cdot (y_{pred} - y_{true}) \cdot x$，而 $b$ 的偏导数为 $2 \cdot (y_{pred} - y_{true})$。

## 3. 梯度下降与训练闭环 (Gradient Descent & Training Loop)
- **变量刷新原则**: 沿着梯面走一步后，由于位置参数（$w$ 和 $b$）已变，对应的梯度（所处地势斜率）也必然改变。
- **发散 (Divergence)**: 如果在循环中拿着旧的梯度连续更新，会导致步子越迈越大，最终 Loss 爆炸到无穷（如你刚踩过的坑：用循环外的初次梯度闭眼走 20 次）。
- **学习率 (Learning Rate, lr)**: 物理世界中的“阻尼器”，充当步长控制缩放因子，防止用力过猛直接越过山谷最低点。
- **神圣闭环**: **Forward Pass** (取得预测) -> **Compute Loss** (估算差距) -> **Compute Gradients** (获取新坐标系下下降的方向) -> **Update Parameters** (按照学习率和方向向山谷走一步)。每一次大循环称为一个 **Epoch**。
