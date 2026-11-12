"""核对选择性披露敏感性分析：退化、稳定、全域认证与独立积分。"""
import json
import math
import unittest
from pathlib import Path
from research_data.io import load
from research_data import selection as sel

NORMAL = sel.validate(load('data/normal.json')[0])
EXTREME = sel.validate(load('data/extreme-tail.json')[0])
NOSEL = sel.validate(load('data/no-selection.json')[0])


def closed_form_normal_nll(v, tau, mu):
    """w=1时的普通正态随机效应负对数似然（不含选择项）。"""
    total = 0.0
    for r in v['studies']:
        s2 = r['s'] * r['s'] + tau * tau
        total += 0.5 * (r['y'] - mu) ** 2 / s2 + 0.5 * math.log(s2)
    return total


class DegeneracyTests(unittest.TestCase):
    def test_w1_matches_unselected_normal_summary(self):
        for tau in NOSEL['tau_grid']:
            with self.subTest(tau=tau):
                got = sel.minimize_over_mu(NOSEL, tau, 1.0)
                s2 = [r['s'] * r['s'] + tau * tau for r in NOSEL['studies']]
                mu_star = sum(r['y'] / v for r, v in zip(NOSEL['studies'], s2)) / sum(1 / v for v in s2)
                mu_star = min(max(mu_star, NOSEL['mu_lo']), NOSEL['mu_hi'])
                self.assertTrue(got['complete'])
                self.assertAlmostEqual(got['mu'], mu_star, delta=1e-3)
                self.assertAlmostEqual(got['nll'], closed_form_normal_nll(NOSEL, tau, mu_star), delta=1e-5)
                self.assertLessEqual(got['nll_lower_bound'], got['nll'])

    def test_w1_selection_norm_is_one(self):
        for mu in (-1.5, -0.3, 0.0, 0.7, 1.5):
            self.assertEqual(sel.selection_norm(mu, 0.31, 0.4, 1.0), 1.0)
            self.assertEqual(sel.log_selection_norm(mu, 0.31, 0.4, 1.0), 0.0)

    def test_w1_effective_weights_match_inverse_variance_at_tau0(self):
        rows, total = sel.effective_weights(NOSEL, 0.0, 0.4, 1.0)
        self.assertGreater(total, 0.0)
        for row in rows:
            self.assertAlmostEqual(row['selected_effective'], row['unadjusted_inverse_variance'], places=12)


class StabilityTests(unittest.TestCase):
    def test_log_z_bounded_and_finite(self):
        w = 1e-8
        for r in EXTREME['studies']:
            for tau in EXTREME['tau_grid']:
                sigma = math.hypot(r['s'], tau)
                t = EXTREME['z_cut'] * r['s']
                for k in range(401):
                    mu = -1.5 + 3.0 * k / 400
                    z = sel.selection_norm(mu, sigma, t, w)
                    self.assertTrue(w <= z <= 1.0)
                    self.assertTrue(math.isfinite(sel.log_selection_norm(mu, sigma, t, w)))

    def test_quadrature_matches_closed_form(self):
        """独立梯形积分核对各研究Z：极小w、大标准误差异、区间两端尾部。"""
        w = 1e-8
        for r in EXTREME['studies']:
            for tau in (0.0, 0.8):
                sigma = math.hypot(r['s'], tau)
                t = EXTREME['z_cut'] * r['s']
                for mu in (-1.5, -0.3, 0.35, 1.5):
                    with self.subTest(study=r['id'], tau=tau, mu=mu):
                        z = sel.selection_norm(mu, sigma, t, w)
                        est, low, high = sel.quadrature_selection_norm(mu, sigma, t, w)
                        self.assertLessEqual(low - 1e-5, z)
                        self.assertLessEqual(z, high + 1e-5)
                        self.assertAlmostEqual(z, est, delta=1e-5)

    def test_quadrature_w1_gives_one(self):
        est, low, high = sel.quadrature_selection_norm(0.4, 0.31, 0.6, 1.0)
        self.assertAlmostEqual(est, 1.0, delta=1e-8)


class GlobalSearchTests(unittest.TestCase):
    def check_against_dense_grid(self, v, w_values, points):
        for w in w_values:
            for tau in v['tau_grid']:
                with self.subTest(w=w, tau=tau):
                    got = sel.minimize_over_mu(v, tau, w)
                    grid = sel.brute_force_min(v, tau, w, points)
                    self.assertTrue(got['complete'])
                    # 连续搜索不劣于网格（相差不超过认证函数误差）
                    self.assertLessEqual(got['nll'], grid['nll'] + 2 * v['tol_f'])
                    # 认证下界不超过真实最小值，故也不超过网格最优
                    self.assertLessEqual(got['nll_lower_bound'], grid['nll'] + 1e-12)
                    # 函数值认证间隙在声明容差内
                    self.assertLessEqual(got['nll'] - got['nll_lower_bound'], 1.5 * v['tol_f'])
                    # 网格最优落在认证最小点区域附近
                    self.assertTrue(any(a - 2 * grid['step'] <= grid['mu'] <= b + 2 * grid['step']
                                        for a, b in got['minimizer_region']))

    def test_certified_minimizer_matches_dense_grid_normal(self):
        self.check_against_dense_grid(NORMAL, (0.02, 0.5, 1.0), 6001)

    def test_certified_minimizer_matches_dense_grid_extreme(self):
        self.check_against_dense_grid(EXTREME, (1e-8,), 6001)

    def test_acceptance_matches_dense_grid(self):
        w = 0.02
        best = min(sel.minimize_over_mu(NORMAL, tau, w)['nll'] for tau in NORMAL['tau_grid'])
        threshold = best + NORMAL['gap']
        in_all = []
        edge_all = []
        for tau in NORMAL['tau_grid']:
            c = sel.acceptance_for(NORMAL, tau, w, threshold)
            self.assertTrue(c['complete'])
            in_all.extend(tuple(s) for s in c['segments'])
            edge_all.extend(tuple(s) for s in c['boundary'])
        in_segments = sel._merge(in_all)
        edge_segments = sel._subtract(sel._merge(edge_all), in_segments)
        grid_segments = sel.brute_force_acceptance(NORMAL, w, threshold, 12001)
        tol_x = NORMAL['tol_x']
        # 网格接受集的每段都必须落在认证接受段或边缘段的tol_x邻域内
        covered = sel._merge([tuple(s) for s in in_segments] + [tuple(s) for s in edge_segments])
        for a, b in grid_segments:
            self.assertTrue(any(ca - 2 * tol_x <= a and b <= cb + 2 * tol_x for ca, cb in covered),
                            msg=f'grid segment {[a, b]} not covered by {covered}')
        # 认证接受段全段可证明被接受：内部抽样必须真的满足阈值
        for a, b in in_segments:
            n = 9
            for k in range(n):
                mu = a + (b - a) * k / (n - 1)
                self.assertLessEqual(sel.profile_nll(NORMAL, w, mu), threshold + 1e-9)

    def test_whole_interval_covered_by_classification(self):
        """每个mu要么被证明nll>阈值，要么落在接受段/边缘段内。"""
        w = 0.1
        best = min(sel.minimize_over_mu(NORMAL, tau, w)['nll'] for tau in NORMAL['tau_grid'])
        threshold = best + NORMAL['gap']
        c = sel.acceptance_for(NORMAL, tau := NORMAL['tau_grid'][1], w, threshold)
        covered = sel._merge([tuple(s) for s in c['segments']] + [tuple(s) for s in c['boundary']])
        for k in range(2001):
            mu = NORMAL['mu_lo'] + (NORMAL['mu_hi'] - NORMAL['mu_lo']) * k / 2000
            inside = any(a - 1e-12 <= mu <= b + 1e-12 for a, b in covered)
            if not inside:
                self.assertGreater(sel.negative_log_likelihood(NORMAL, tau, w, mu), threshold)


class ReportTests(unittest.TestCase):
    def test_extreme_tail_boundary_and_weights(self):
        report = sel.analyze(load('data/extreme-tail.json')[0])
        p = report['per_w'][0]
        self.assertEqual(p['status'], 'complete')
        self.assertTrue(p['boundary']['mu_at_upper'])
        self.assertTrue(p['boundary']['tau_at_grid_max'])
        self.assertTrue(any('边界' in n or '端点' in n for n in p['notes']))
        weights = {w_['id']: w_ for w_ in p['weights']}
        # 选择模型下极端研究的权重占比下降
        self.assertLess(weights['extreme']['selected_effective'],
                        weights['extreme']['unadjusted_inverse_variance'])
        self.assertAlmostEqual(sum(w_['selected_effective'] for w_ in p['weights']), 1.0, places=9)

    def test_report_structure_and_direction_flags(self):
        report = sel.analyze(load('data/normal.json')[0])
        self.assertEqual(report['incomplete_w'], [])
        for p in report['per_w']:
            acc = p['acceptance']
            self.assertEqual(len(p['candidates']), len(NORMAL['tau_grid']))
            self.assertIn(p['best'], p['candidates'])
            outer = sel._merge([tuple(s) for s in acc['segments']] + [tuple(s) for s in acc['boundary_segments']])
            has_neg = any(a < 0 for a, _ in outer)
            has_pos = any(b > 0 for _, b in outer)
            has_zero = any(a <= 0 <= b for a, b in outer)
            self.assertEqual(acc['direction_uncertain'], (has_neg and has_pos) or has_zero)
            self.assertEqual(acc['contains_zero'], has_zero)
            self.assertEqual(p['best']['tau'], min(p['candidates'], key=lambda c: c['nll'])['tau'])
        self.assertEqual(report['direction_uncertain_w'],
                         [p['w'] for p in report['per_w'] if p['acceptance']['direction_uncertain']])
        # normal.json各w下接受集都在正侧：方向确定
        self.assertEqual(report['direction_uncertain_w'], [])

    def test_deterministic(self):
        a = sel.analyze(load('data/normal.json')[0])
        b = sel.analyze(load('data/normal.json')[0])
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))

    def test_resource_exhaustion_reports_incomplete(self):
        cfg = json.loads(json.dumps(load('data/normal.json')[0]))
        cfg['solver']['max_evaluations'] = 5
        report = sel.analyze(cfg)
        self.assertNotEqual(report['incomplete_w'], [])
        for p in report['per_w']:
            self.assertEqual(p['status'], 'incomplete')


class ValidationTests(unittest.TestCase):
    def test_invalid_material_rejected(self):
        for p in Path('data').glob('invalid-*.json'):
            with self.subTest(file=p.name):
                with self.assertRaises((ValueError, KeyError, TypeError)):
                    sel.validate(load(p)[0])

    def test_bad_configs_rejected(self):
        base = load('data/normal.json')[0]

        def bad(**patch):
            cfg = json.loads(json.dumps(base))
            for key, val in patch.items():
                cfg[key] = val
            with self.assertRaises((ValueError, KeyError, TypeError)):
                sel.validate(cfg)

        bad(w_grid=[0.0])
        bad(w_grid=[1.2])
        bad(w_grid=[])
        bad(tau_grid=[-0.1])
        bad(tau_grid=[])
        bad(mu_bounds=[1.0, 1.0])
        bad(mu_bounds=[2.0, -2.0])
        bad(z_cut=0.0)
        bad(z_cut=-1.0)
        bad(likelihood_gap=-0.5)
        bad(solver={'objective_tolerance': 0.0, 'location_tolerance': 1e-5, 'max_evaluations': 10})
        bad(solver={'objective_tolerance': 1e-6, 'location_tolerance': 1e-5, 'max_evaluations': 0})
        bad(studies=[])


if __name__ == '__main__':
    unittest.main()
