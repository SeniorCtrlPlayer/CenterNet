# dup

1. h: offset[0,0,0,0]=0
2. w: offset[1,0,0,0]=1
3. input [c, height_in, width_in]
4. output [c, height_in x 2, width_in x 2]
5. input[k, 1, 0] -> output[k, 0, 0]
6. i_1 * 