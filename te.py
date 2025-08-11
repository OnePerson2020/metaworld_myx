def superEggDrop(k: int, n: int) -> int:
    # dp[j] 表示在当前操作次数下，使用 j 个鸡蛋能确定的最大楼层数。
    # 数组大小为 k+1，索引从 0 到 k。
    dp = [0] * (k + 1)
    
    # m 代表操作次数
    m = 0
    
    # 当 dp[k] (k个鸡蛋能确定的最大楼层数) 小于 n 时，
    # 说明当前的操作次数 m 不足以覆盖所有楼层，需要增加操作次数。
    while dp[k] < n:
        m += 1
        # 从后往前更新 dp 数组，以保证使用的是上一轮 (m-1) 的值
        # dp[j] 的新值依赖于旧的 dp[j] 和 dp[j-1]
        for j in range(k, 0, -1):
            dp[j] = dp[j-1] + dp[j] + 1
            
    return m