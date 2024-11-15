## IQL 问题

1. V的估计不准
2. 策略不是最优

## 思考
1. IQL在online之后的表现如何
2. IQL的V是波动的还是估值过高的
3. 策略如何迅速收敛

## 结论
python IQL-rethink/eval.py --env-name walker2d-medium-v2 --load-policy log/walker2d-medium-v2/11-04-24_10.54.54_tdco/policy-final.pt
稳定在3500~4000之间

效果还行的是 用IQL网络选策略

效果不行的是 IQL的Value网络直接当作PPO的value网络

考虑到IQL没有policy

所以有以下几种对照：


初始的actor和初始的critic 用IQL的policy冒充随机轨迹 效果还行


初始的actor和IQL的V（critic） 直接使用（不行 因为actor太差会导致异常）

初始的actor和IQL的V（critic） V锁定，猜测也不行

初始的actor和IQL的V 用IQL的policy冒充轨迹 不太会掉 但也不咋升

初始的actor和IQL的V 用IQL的policy冒充轨迹 V锁定 试了一下会很快掉下来 有点奇怪

有两个思路 第一个是先锁了V 让actor上升以下


用了拟合的critic和actor网络 但是训练时候还是会下降 只能认为是存在value的高估了

walker-2d 满分4600


##new idea
先预训练 然后
