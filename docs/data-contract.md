# 已披露研究效应

effect是同一方向和同一量纲的观察效应，se为已知标准误，研究相互独立。z_cut为正的原研究标准化阈值；w_grid为外给披露相对概率，0<w<=1。tau_grid是异质性标准差的有限非负域，mu_bounds是明确的有限搜索区间。

likelihood_gap表示负对数似然不超过最佳值加该差值的mu集合，在固定w下需要同时考虑tau网格；不是把各tau局部最优点连线。solver的函数误差和位置误差分别声明，不把接受域边界的有限精度当作数学上的连续区间精确端点。已观察研究的披露权重由其y/s落入哪个区域决定，归一化Z依赖(mu,tau,se)。

当前domain显示未调整的观察逆方差均值，不做任何披露调整或未发表资料重构。

# 敏感性分析输出

`research_data.selection.analyze`（CLI命令`sensitivity`）对每个预设w输出：最佳网格tau与mu及其函数值上界/下界认证、全部候选目标、mu的似然次水平接受集合、方向与边界诊断、研究权重改变。

mu的一维优化用区间分支定界覆盖整个mu_bounds：每个子区间用中点函数值、解析梯度与区间内|f''|的解析上界给出可证明的函数上下界，候选最小点区域宽度不超过location_tolerance，函数值认证间隙不超过objective_tolerance。evaluations统计目标（含梯度）求值次数，超过max_evaluations即报告incomplete，不伪造完成。

接受集合为 {mu: min_tau nll(mu) <= 最佳nll + likelihood_gap}，固定w下对tau网格取轮廓，不是把各tau局部最优点连线。segments为认证全段接受（段内每点可证明满足阈值），boundary_segments为location_tolerance内未判定的边缘；真实接受集介于二者之间，方向判定（含contains_zero、direction_uncertain）基于二者并集这一外近似。集合可以不连续，不默认称为具有某覆盖率的置信区间。

研究权重改变以最优点处每研究对mu的局部曲率（观察信息）份额对比未调整逆方差份额，曲率可以为非正，仅作诊断，不用于重新估计。单研究披露概率Z可用quadrature_selection_norm独立积分核对：返回中心估计与同时覆盖尾部截断和离散误差估计的上下界。
