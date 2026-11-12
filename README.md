# 选择性研究披露下的效应敏感性轮廓

这是研究资料维护用的独立Python项目。`data/`保存合成记录，`research_data.io`负责严格JSON读取与SHA-256来源摘要，`research_data.domain`提供当前原始资料统计，`research_data.cli`将摘要输出为JSON。字段、时间、计数和量纲见[资料约定](docs/data-contract.md)。所有研究、中心及受试者均为虚构。

Python 3.11及以上版本可直接使用，当前没有第三方依赖。在项目根目录运行：

```sh
python -m unittest discover -s tests -v
python -m compileall -q research_data tests
python -m research_data.cli inspect data/normal.json
python -m research_data.cli sensitivity data/normal.json
```

`inspect`只显示已有摘要，不修改来源。`sensitivity`运行选择性披露敏感性分析：对每个外给披露倾向w，在固定tau网格上对mu做受界一维全域最大似然，报告最佳(tau,mu)、全部候选目标、由`likelihood_gap`定义的mu接受集合（可以不连续）、方向是否仍不确定、边界解与各研究权重改变。w只是外给敏感性假设，不把似然最高的w当作资料识别出的真实发表偏倚；w=1退化为未选择的正态汇总；结果依赖正态异质性与所设选择机制，不声称恢复真实未发表资料。

全域搜索用区间分支定界并给出函数值上界/下界认证，不是单次局部搜索；求值预算耗尽时报告`incomplete`。可用密集网格与独立积分自行核对：

```python
from research_data.io import load
from research_data.selection import (validate, minimize_over_mu, acceptance_for,
    brute_force_min, brute_force_acceptance, quadrature_selection_norm, selection_norm)
v = validate(load('data/normal.json')[0])
minimize_over_mu(v, 0.1, 0.1)          # 认证全域最优（含下界与最小点区域）
brute_force_min(v, 0.1, 0.1, 6001)     # 密集网格对照
brute_force_acceptance(v, 0.1, -5.0, 12001)  # 接受域网格对照
quadrature_selection_norm(0.35, 0.806, 0.196, 1e-8)  # 独立积分核对单研究Z
```

文件名以`invalid-`起始的资料包含故意不合法的输入，其他文件是同结构的业务边界。数据摘要与来源指纹应同时保留，原始JSON不覆盖。

可使用固定镜像容器：`docker build -t research-material .`，随后`docker run --rm research-material`查看资料；`docker run --rm research-material python -m unittest discover -s tests -v`运行现有检查。全部资料在项目内，运行不需要外部账号或服务。
