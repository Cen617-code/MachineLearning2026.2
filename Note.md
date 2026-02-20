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
