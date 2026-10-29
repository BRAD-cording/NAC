# 板端 512-token 复测汇总

| tag | model | md5 | n | acc(last-num) | acc(answer-is) | hit_cap | mean tok | med chars | cross-q |
|---|---|---|---|---|---|---|---|---|---|

判据：`cross-q` 与 `hit_cap` 都低说明模型能在答案后停止；`acc` 与 `acc(answer-is)` 接近说明末数字抽取没有被续写污染。
