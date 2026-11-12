"""选择性披露敏感性分析：外给披露概率w下的选择模型轮廓似然。

模型：第i篇研究潜在效应 y ~ N(mu, se_i^2 + tau^2)；披露概率在
|y| >= z_cut * se_i 时为1，否则为外给w（0<w<=1），阈值用原研究se，
不换成总异质性标准差。已披露条件密度 = 原正态密度 × 披露权重 / Z_i(mu,tau)，
Z_i 为对每篇研究自身 (se_i, tau) 单独计算的正态尾积分，不共用归一化常数。

w=1 时 Z_i = 1，退化为未选择的正态随机效应汇总。w 只是外给敏感性假设：
不把候选中似然最高的 w 当作资料识别出的真实发表偏倚，也不声称恢复
真实未发表资料；结论依赖正态异质性与所设选择机制。

mu 的一维全域优化用区间分支定界：每个子区间给出目标函数可证明的
上下界，覆盖整个给定 mu 区间，不是单次局部搜索；求值预算耗尽时
报告 complete=False。接受集合由调用者给定的 likelihood_gap 定义，
可以不连续，不默认称为具有某覆盖率的置信区间。
"""
import math
from heapq import heapify, heappop, heappush
from .io import finite, positive, integer, identity, unique

_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)


def normal_cdf(x):
    """标准正态分布函数，用erfc计算，两端尾部稳定。"""
    return 0.5 * math.erfc(-x / _SQRT2)


def _normal_pdf(x):
    return math.exp(-0.5 * x * x) / _SQRT2PI


def selection_norm(mu, sigma, threshold, w):
    """单研究披露概率Z：对 N(mu, sigma^2) 潜变量求披露权重期望。

    Z = w + (1-w) * P(|Y| >= threshold)，threshold = z_cut * se。
    因为 sigma >= se，mu=0 处 q >= 2*Phi(-z_cut) > 0，且 w > 0，
    故 Z 严格为正，取对数安全；w=1 时精确等于1。
    """
    q = normal_cdf((mu - threshold) / sigma) + normal_cdf((-threshold - mu) / sigma)
    return w + (1.0 - w) * q


def log_selection_norm(mu, sigma, threshold, w):
    """log Z，由 selection_norm 取对数，极小w下不退化。"""
    return math.log(selection_norm(mu, sigma, threshold, w))


def quadrature_selection_norm(mu, sigma, threshold, w, half_width=12.0, intervals=24000):
    """独立数值积分核对Z：估计 q=P(|Y|>=threshold) 再组合 Z=w+(1-w)q。

    只在 |y|>=threshold 的分段上梯形积分正态密度（被积函数光滑，
    不含披露权重的跳变），区间为 mu ± half_width*sigma。截断尾部
    质量 T 由正态尾概率给出，对 q 的贡献介于 0 与 T 之间；离散
    误差用加倍网格的 Richardson 外推估计。返回 (中心估计, 下界,
    上界)，上下界同时覆盖尾部截断与离散误差估计。
    """
    lo = mu - half_width * sigma
    hi = mu + half_width * sigma
    pieces = []
    if lo < -threshold:
        pieces.append((lo, min(-threshold, hi)))
    if hi > threshold:
        pieces.append((max(threshold, lo), hi))

    def trapezoid(each):
        total = 0.0
        for a, b in pieces:
            h = (b - a) / each
            for k in range(each + 1):
                y = a + k * h
                d = (y - mu) / sigma
                total += math.exp(-0.5 * d * d) / (sigma * _SQRT2PI) * (h if 0 < k < each else 0.5 * h)
        return total

    each = max(4, intervals // max(1, len(pieces)))
    coarse = trapezoid(each)
    fine = trapezoid(2 * each)
    err = abs(fine - coarse) / 3.0
    tail = normal_cdf((lo - mu) / sigma) + normal_cdf((mu - hi) / sigma)
    q_lo = max(0.0, fine - err)
    q_hi = min(1.0, fine + err + tail)
    est = w + (1.0 - w) * fine
    return est, w + (1.0 - w) * q_lo, w + (1.0 - w) * q_hi


def validate(config):
    """严格核对公开配置，返回规范化后的副本；不合法输入抛错。"""
    if not isinstance(config, dict):
        raise ValueError('配置需要对象')
    studies = config['studies']
    if not isinstance(studies, list) or not studies:
        raise ValueError('缺少研究')
    unique(studies, 'id')
    rows = []
    for r in studies:
        rows.append({'id': identity(r['id']), 'y': finite(r['effect']), 's': positive(r['se'])})
    z_cut = positive(config['z_cut'])
    w_grid = config['w_grid']
    if not isinstance(w_grid, list) or not w_grid:
        raise ValueError('缺少w候选')
    ws = []
    for w in w_grid:
        w = finite(w)
        if w <= 0.0 or w > 1.0:
            raise ValueError('w需要在(0,1]内')
        ws.append(w)
    tau_grid = config['tau_grid']
    if not isinstance(tau_grid, list) or not tau_grid:
        raise ValueError('缺少tau网格')
    taus = []
    for t in tau_grid:
        t = finite(t)
        if t < 0.0:
            raise ValueError('tau需要非负')
        taus.append(t)
    bounds = config['mu_bounds']
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise ValueError('mu_bounds需要两个端点')
    lo, hi = finite(bounds[0]), finite(bounds[1])
    if not lo < hi:
        raise ValueError('mu_bounds需要严格递增')
    gap = finite(config['likelihood_gap'])
    if gap < 0.0:
        raise ValueError('likelihood_gap需要非负')
    solver = config['solver']
    tol_f = positive(solver['objective_tolerance'])
    tol_x = positive(solver['location_tolerance'])
    max_evals = integer(solver['max_evaluations'], 1)
    return {'studies': rows, 'z_cut': z_cut, 'w_grid': ws, 'tau_grid': taus,
            'mu_lo': lo, 'mu_hi': hi, 'gap': gap,
            'tol_f': tol_f, 'tol_x': tol_x, 'max_evals': max_evals}


def _build(v, tau, w):
    """构造 (terms, const)：nll(mu) = sum(quad + logZ) + const。

    const 含 log(sigma) 与已观察披露权重 -log(p_obs)，不随mu变化，
    但随tau与w变化，必须保留才能比较同一w下各tau的候选目标。
    """
    terms = []
    const = 0.0
    for r in v['studies']:
        s = r['s']
        sigma = math.sqrt(s * s + tau * tau)
        t = v['z_cut'] * s
        p_obs = 1.0 if abs(r['y']) >= t else w
        const += math.log(sigma) - math.log(p_obs)
        terms.append((r['y'], sigma, t))
    return terms, const


def _nll(terms, const, w, mu):
    return _eval(terms, const, w, mu)[0]


def _eval(terms, const, w, mu):
    """nll及其对mu的解析导数（一次遍历共享正态cdf/pdf求值）。"""
    total = const
    grad = 0.0
    for y, sigma, t in terms:
        d = (y - mu) / sigma
        total += 0.5 * d * d + log_selection_norm(mu, sigma, t, w)
        grad += (mu - y) / (sigma * sigma)
        a = (mu - t) / sigma
        b = (-t - mu) / sigma
        z = w + (1.0 - w) * (normal_cdf(a) + normal_cdf(b))
        q1 = (_normal_pdf(a) - _normal_pdf(b)) / sigma
        grad += (1.0 - w) * q1 / z
    return total, grad


def negative_log_likelihood(v, tau, w, mu):
    """完整负对数似然（含不依赖mu的项），v为validate后的配置。"""
    terms, const = _build(v, tau, w)
    return _eval(terms, const, w, mu)[0]


def profile_nll(v, w, mu):
    """固定w下对tau网格取轮廓：min_tau nll(mu, tau, w)。"""
    return min(negative_log_likelihood(v, tau, w, mu) for tau in v['tau_grid'])


def _phi_max(a, b, c, sigma):
    """max_{mu in [a,b]} phi((mu-c)/sigma)：在离c最近处取到。"""
    nearest = min(max(c, a), b)
    return _normal_pdf((nearest - c) / sigma)


def _abs_x_phi_max(lo_x, hi_x):
    """max |x|*phi(x) over x in [lo_x,hi_x]：峰值在x=±1。"""
    best = 0.0
    for x in (lo_x, hi_x, -1.0, 1.0):
        if lo_x <= x <= hi_x:
            val = abs(x) * _normal_pdf(x)
            if val > best:
                best = val
    return best


def _bounds(terms, const, w, a, b, oracle):
    """nll在[a,b]上的可证明下界/上界（中心形式，二阶）。

    在中点解析求值f与f'，再用|f''|在区间上的解析上界L2：
    f(mu) ∈ f(m) + f'(m)(mu-m) ± L2*(mu-m)^2/2。
    q(mu)=P(|Y|>=t)关于|mu|单调增，故Z在离0最近点取下界；
    |q'|、|q''|的分量 phi 与 |x|phi(x) 在区间上的最大值均精确可得。
    """
    m = 0.5 * (a + b)
    r = 0.5 * (b - a)
    fm, dm = oracle(m)
    l2 = 0.0
    mu_c = 0.0 if a <= 0.0 <= b else (a if abs(a) <= abs(b) else b)
    for y, sigma, t in terms:
        inv_s2 = 1.0 / (sigma * sigma)
        q_lo = normal_cdf((mu_c - t) / sigma) + normal_cdf((-t - mu_c) / sigma)
        z_lo = w + (1.0 - w) * q_lo
        q1_max = (_phi_max(a, b, t, sigma) + _phi_max(a, b, -t, sigma)) / sigma
        q2_max = (_abs_x_phi_max((a - t) / sigma, (b - t) / sigma)
                  + _abs_x_phi_max((-t - b) / sigma, (-t - a) / sigma)) / (sigma * sigma)
        d1_max = (1.0 - w) * q1_max / z_lo
        l2 += inv_s2 + (1.0 - w) * q2_max / z_lo + d1_max * d1_max
    delta = abs(dm) * r + 0.5 * l2 * r * r
    return fm - delta, fm + delta


def _merge(intervals):
    """合并相接区间（二分产生的端点完全一致）。"""
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _subtract(intervals, segments):
    """从区间集中减去另一组区间（均已合并）。"""
    out = []
    for a, b in intervals:
        pieces = [(a, b)]
        for s0, s1 in segments:
            nxt = []
            for p0, p1 in pieces:
                if s1 <= p0 or s0 >= p1:
                    nxt.append((p0, p1))
                else:
                    if p0 < s0:
                        nxt.append((p0, s0))
                    if s1 < p1:
                        nxt.append((s1, p1))
            pieces = nxt
        out.extend(pieces)
    return [[a, b] for a, b in out]


def _minimize(terms, const, w, lo, hi, tol_f, tol_x, max_evals):
    """区间分支定界全域最小化，覆盖整个[lo,hi]。

    返回 best（已达成的函数值上界）、certified lower（全域下界），
    二者差不超过 tol_f 且候选区间宽度不超过 tol_x 时 complete=True；
    预算（目标函数求值次数）耗尽则 complete=False 并保留当前界。
    """
    cache = {}
    evals = [0]
    best = [float('inf')]
    best_mu = [lo]

    def oracle(mu):
        """返回(f, f')并跟踪最优；每个新mu计一次求值。"""
        got = cache.get(mu)
        if got is None:
            got = _eval(terms, const, w, mu)
            cache[mu] = got
            evals[0] += 1
            if got[0] < best[0]:
                best[0] = got[0]
                best_mu[0] = mu
        return got

    oracle(lo)
    oracle(hi)
    low, _ = _bounds(terms, const, w, lo, hi, oracle)
    active = [(low, lo, hi)]
    exhausted = []  # 机器精度无法再分的区间：(a, b, 精确最小值)
    complete = True
    certified = low
    while True:
        active = [iv for iv in active if iv[0] <= best[0]]
        heapify(active)
        lows = [iv[0] for iv in active] + [e[2] for e in exhausted]
        certified = min(lows) if lows else best[0]
        if not active:
            break
        if best[0] - certified <= tol_f and all(b - a <= tol_x for _, a, b in active):
            break
        if evals[0] >= max_evals:
            complete = False
            break
        if best[0] - certified > tol_f:
            _, a, b = heappop(active)
        else:
            iv = max(active, key=lambda t: t[2] - t[1])
            active.remove(iv)
            _, a, b = iv
        m = 0.5 * (a + b)
        if m <= a or m >= b:
            exhausted.append((a, b, min(oracle(a)[0], oracle(b)[0])))
            continue
        for a2, b2 in ((a, m), (m, b)):
            l2, _ = _bounds(terms, const, w, a2, b2, oracle)
            if l2 <= best[0]:
                heappush(active, (l2, a2, b2))
    regions = _merge([(a, b) for _, a, b in active] + [(a, b) for a, b, _ in exhausted])
    return {'mu': best_mu[0], 'nll': best[0], 'nll_lower_bound': certified,
            'minimizer_region': regions, 'evaluations': evals[0], 'complete': complete}


def minimize_over_mu(v, tau, w):
    """固定(tau,w)对mu做受界一维全域最大似然（最小化负对数似然）。"""
    terms, const = _build(v, tau, w)
    return _minimize(terms, const, w, v['mu_lo'], v['mu_hi'],
                     v['tol_f'], v['tol_x'], v['max_evals'])


def _classify(terms, const, w, lo, hi, threshold, tol_x, max_evals):
    """把[lo,hi]分为 nll<=threshold 的认证接受段与 tol_x 内未判定的边缘段。

    区间下界高于阈值即整体排除（可证明）；认证上界不超过阈值才整段
    接受（可证明，段内每点都满足）；其余细分到 tol_x 后归入边缘段。
    预算耗尽时未判定区间并入边缘段并报告 complete=False。
    """
    cache = {}
    evals = [0]

    def oracle(mu):
        got = cache.get(mu)
        if got is None:
            got = _eval(terms, const, w, mu)
            cache[mu] = got
            evals[0] += 1
        return got

    oracle(lo)
    oracle(hi)
    in_iv = []
    edge_iv = []
    complete = True
    low, high = _bounds(terms, const, w, lo, hi, oracle)
    stack = [(lo, hi, low, high)]
    while stack:
        a, b, lo_b, hi_b = stack.pop()
        if lo_b > threshold:
            continue
        if hi_b <= threshold:
            in_iv.append((a, b))
            continue
        m = 0.5 * (a + b)
        if b - a <= tol_x or m <= a or m >= b:
            edge_iv.append((a, b))
            continue
        if evals[0] >= max_evals:
            complete = False
            edge_iv.append((a, b))
            continue
        for a2, b2 in ((a, m), (m, b)):
            l2, h2 = _bounds(terms, const, w, a2, b2, oracle)
            if l2 <= threshold:
                stack.append((a2, b2, l2, h2))
    return _merge(in_iv), _merge(edge_iv), complete, evals[0]


def acceptance_for(v, tau, w, threshold):
    """固定(tau,w)的次水平接受集 {mu: nll <= threshold}。"""
    terms, const = _build(v, tau, w)
    segments, boundary, complete, evals = _classify(
        terms, const, w, v['mu_lo'], v['mu_hi'], threshold, v['tol_x'], v['max_evals'])
    return {'segments': segments, 'boundary': boundary,
            'complete': complete, 'evaluations': evals, 'threshold': threshold}


def brute_force_min(v, tau, w, points):
    """密集网格核对用：在等距网格上求 nll 最小值（对照全域搜索）。"""
    terms, const = _build(v, tau, w)
    lo, hi = v['mu_lo'], v['mu_hi']
    best_mu = lo
    best = float('inf')
    for k in range(points):
        mu = lo + (hi - lo) * k / (points - 1)
        val = _nll(terms, const, w, mu)
        if val < best:
            best = val
            best_mu = mu
    return {'mu': best_mu, 'nll': best, 'points': points, 'step': (hi - lo) / (points - 1)}


def brute_force_acceptance(v, w, threshold, points):
    """密集网格核对用：轮廓nll（对tau网格取min）不超过阈值的网格段。"""
    lo, hi = v['mu_lo'], v['mu_hi']
    segments = []
    start = None
    prev = None
    for k in range(points):
        mu = lo + (hi - lo) * k / (points - 1)
        if profile_nll(v, w, mu) <= threshold:
            if start is None:
                start = mu
        elif start is not None:
            segments.append([start, prev])
            start = None
        prev = mu
    if start is not None:
        segments.append([start, prev])
    return segments


def effective_weights(v, tau, mu, w):
    """最优点处每研究权重诊断：未调整逆方差份额 vs 选择模型局部曲率份额。

    曲率 c_i = 1/sigma_i^2 + d^2(logZ_i)/dmu^2（观察信息），可以为非正；
    仅为诊断，不用于重新估计。返回 (rows, 曲率合计)。
    """
    rows = []
    total_c = 0.0
    total_iv = 0.0
    for r in v['studies']:
        s = r['s']
        sigma2 = s * s + tau * tau
        sigma = math.sqrt(sigma2)
        t = v['z_cut'] * s
        a = (mu - t) / sigma
        b = (-t - mu) / sigma
        z = w + (1.0 - w) * (normal_cdf(a) + normal_cdf(b))
        q1 = (_normal_pdf(a) - _normal_pdf(b)) / sigma
        q2 = (-a * _normal_pdf(a) - b * _normal_pdf(b)) / sigma2
        d1 = (1.0 - w) * q1 / z
        d2 = (1.0 - w) * q2 / z - d1 * d1
        curvature = 1.0 / sigma2 + d2
        iv = 1.0 / (s * s)
        rows.append({'id': r['id'], 'curvature': curvature, 'inverse_variance': iv})
        total_c += curvature
        total_iv += iv
    for row in rows:
        row['unadjusted_inverse_variance'] = row.pop('inverse_variance') / total_iv
        row['selected_effective'] = row['curvature'] / total_c if total_c > 0.0 else None
    return rows, total_c


_NOTES = [
    'w为外给敏感性假设，不把候选中似然最高的w解释为资料识别出的真实发表偏倚',
    '接受集合由调用者给定的likelihood_gap定义，可以不连续，不默认称为具有某覆盖率的置信区间',
    '结果依赖正态异质性假设与所设选择机制，未恢复真实未发表研究资料',
    'w=1时选择归一化Z=1，退化为未选择的正态随机效应汇总',
    'selected_effective为最优点处每研究对mu的局部曲率份额，可以为非正，仅作诊断',
]


def analyze(config):
    """对每个预设w：全域优化、全部候选目标、接受集合、方向与边界诊断。"""
    v = validate(config)
    per_w = []
    for w in v['w_grid']:
        candidates = []
        for tau in v['tau_grid']:
            r = minimize_over_mu(v, tau, w)
            candidates.append({'tau': tau, **r})
        best = min(candidates, key=lambda c: c['nll'])
        threshold = best['nll'] + v['gap']
        in_all = []
        edge_all = []
        cls_ok = True
        evals = sum(c['evaluations'] for c in candidates)
        for tau in v['tau_grid']:
            c = acceptance_for(v, tau, w, threshold)
            in_all.extend(tuple(s) for s in c['segments'])
            edge_all.extend(tuple(s) for s in c['boundary'])
            cls_ok = cls_ok and c['complete']
            evals += c['evaluations']
        segments = _merge(in_all)
        boundary = _subtract(_merge(edge_all), segments)
        complete = cls_ok and all(c['complete'] for c in candidates)
        # 方向判定用接受段与边缘段的并集（外近似）：并集跨零才报方向不确定
        outer = _merge([tuple(s) for s in segments] + [tuple(s) for s in boundary])
        has_neg = any(a < 0.0 for a, _ in outer)
        has_pos = any(b > 0.0 for _, b in outer)
        has_zero = any(a <= 0.0 <= b for a, b in outer)
        weights, curvature_sum = effective_weights(v, best['tau'], best['mu'], w)
        regions = best['minimizer_region']
        mu_at_lower = any(a <= v['mu_lo'] + v['tol_x'] for a, _ in regions)
        mu_at_upper = any(b >= v['mu_hi'] - v['tol_x'] for _, b in regions)
        notes = []
        if mu_at_lower or mu_at_upper:
            notes.append('最佳mu触及搜索区间边界，扩大mu_bounds可能改变结论')
        if best['tau'] == min(v['tau_grid']) or best['tau'] == max(v['tau_grid']):
            notes.append('最佳tau位于网格端点，加密或扩展tau_grid可能改变结论')
        per_w.append({
            'w': w,
            'status': 'complete' if complete else 'incomplete',
            'best': best,
            'candidates': candidates,
            'acceptance': {
                'definition': 'min_tau nll(mu) <= best_nll + likelihood_gap',
                'gap': v['gap'],
                'threshold_nll': threshold,
                'segments': segments,
                'boundary_segments': boundary,
                'segments_meaning': 'segments为认证全段接受，boundary_segments为tol_x内未判定边缘；方向判定基于二者并集（外近似）',
                'location_tolerance': v['tol_x'],
                'objective_tolerance': v['tol_f'],
                'contains_zero': has_zero,
                'direction_uncertain': (has_neg and has_pos) or has_zero,
            },
            'boundary': {
                'mu_at_lower': mu_at_lower,
                'mu_at_upper': mu_at_upper,
                'tau_at_grid_min': best['tau'] == min(v['tau_grid']),
                'tau_at_grid_max': best['tau'] == max(v['tau_grid']),
            },
            'weights': weights,
            'curvature_sum': curvature_sum,
            'evaluations': evals,
            'notes': notes,
        })
    return {
        'config': {
            'studies': [{'id': r['id'], 'effect': r['y'], 'se': r['s']} for r in v['studies']],
            'z_cut': v['z_cut'],
            'w_grid': v['w_grid'],
            'tau_grid': v['tau_grid'],
            'mu_bounds': [v['mu_lo'], v['mu_hi']],
            'likelihood_gap': v['gap'],
            'solver': {'objective_tolerance': v['tol_f'],
                       'location_tolerance': v['tol_x'],
                       'max_evaluations': v['max_evals']},
        },
        'model': {
            'latent': 'y ~ N(mu, se^2 + tau^2)',
            'selection': 'P(disclose|y) = 1 if |y| >= z_cut*se else w',
            'conditional_density': 'normal_pdf(y; mu, se^2+tau^2) * p(y) / Z(mu, tau, se)',
            'normalization': 'Z = w + (1-w) * P(|Y| >= z_cut*se)，逐研究正态尾积分，不共用常数',
        },
        'per_w': per_w,
        'direction_uncertain_w': [p['w'] for p in per_w if p['acceptance']['direction_uncertain']],
        'incomplete_w': [p['w'] for p in per_w if p['status'] != 'complete'],
        'notes': list(_NOTES),
    }
