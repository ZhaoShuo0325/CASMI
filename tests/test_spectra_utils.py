'''
Author: Shuo Zhao && 18904530325@163.com
Date: 2026-09-29 16:07:32
LastEditors: Shuo Zhao && 18904530325@163.com
LastEditTime: 2026-09-29 19:28:57
FilePath: /00_CASMI/tests/test_spectra_utils.py
Description: 

'''
# tests/test_spectra_utils.py
"""阶段 2 单元测试：验证 spectra_utils。"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spectra_utils import simulate_spectrum, aggregate_spectra, plot_spectrum


def test_simulate_spectrum():
    spec = simulate_spectrum("c1ccccc1O", adduct="[M+H]+", ce=30.0)
    assert spec is not None
    assert spec.precursor_mz > 0
    assert len(spec.mz) > 0
    assert len(spec.mz) == len(spec.intensity)
    print(f"  苯酚模拟谱图: precursor={spec.precursor_mz:.3f}, "
          f"n_peaks={len(spec)}")


def test_normalized():
    spec = simulate_spectrum("c1ccccc1O").normalized()
    assert abs(spec.intensity.max() - 1.0) < 1e-6
    print(f"  归一化后 base peak intensity = {spec.intensity.max():.4f}")


def test_top_k():
    spec = simulate_spectrum("c1ccccc1O")
    top = spec.top_k_peaks(k=3)
    assert len(top) <= 3
    print(f"  top-3 峰数: {len(top)}")


def test_aggregate():
    s1 = simulate_spectrum("c1ccccc1O", ce=20.0, seed=1)
    s2 = simulate_spectrum("c1ccccc1O", ce=40.0, seed=2)
    agg = aggregate_spectra([s1, s2], method="concat")
    assert len(agg) == len(s1) + len(s2)
    print(f"  聚合后峰数: {len(agg)} (来自 {len(s1)} + {len(s2)})")


if __name__ == "__main__":
    test_simulate_spectrum()
    test_normalized()
    test_top_k()
    test_aggregate()
    print("✅ 阶段 2 测试通过")