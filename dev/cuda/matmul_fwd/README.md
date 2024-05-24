
# 矩阵乘法分析
计算 $C=A*B$ ,矩阵 $C, A, B$， 本文是使用cuda场景下的分析

计算量: $2 * M * N * K$
## cuda调度方式1
```<<<dim3(M/Tile, N/Tile), threads>>>```
        $$C_w=MN$$
        $$A_r=MK*\frac{N}{Tile}$$
        $$B_r=KN*\frac{M}{Tile}$$
        $$total=\frac{2MNK}{Tile}+MN$$
        $$CI = \frac{2MNK}{2\frac{MNK}{Tile}+MN} = \frac{2}{2/Tile+1/K}=O(Tile)$$

# 优化
## 异步拷贝
## 双缓存 double buffer
# 参考

[Lecture 3: More MatMul and the Roofline Performance Model](https://sites.google.com/lbl.gov/cs267-spr2023)

[算子融合理论推导](https://github.com/l1351868270/LD_mma/blob/main/doc/%E7%AE%97%E5%AD%90%E8%9E%8D%E5%90%88%E7%90%86%E8%AE%BA%E6%8E%A8%E5%AF%BC.md)






