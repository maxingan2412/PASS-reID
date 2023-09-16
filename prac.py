import math

# 初始化 d_model
d_model = 6

# 初始化位置编码矩阵 PE
PE = [[0 for _ in range(d_model)] for _ in range(4)]
print(PE)

for pos in range(4):
    for i in range(d_model):
        if i % 2 == 0:
            PE[pos][i] = math.sin(pos / (10000**(i/d_model)))
        else:
            PE[pos][i] = math.cos(pos / (10000**((i-1)/d_model)))



for pos in range(4):
    print(f"PE({pos+1}, :) = {PE[pos]}")
