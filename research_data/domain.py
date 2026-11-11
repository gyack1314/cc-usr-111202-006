"""按已披露资料展示未作选择调整的描述汇总。"""
from .io import unique,positive,finite

def summarize(v):
    unique(v['studies'],'id');weighted=0.0;precision=0.0
    if not v['studies']:raise ValueError('缺少研究')
    for r in v['studies']:
        y=finite(r['effect']);s=positive(r['se']);w=1/(s*s)
        weighted+=w*y;precision+=w
    return {'published_studies':len(v['studies']),'observed_inverse_variance_mean':weighted/precision,'precision_sum':precision}
