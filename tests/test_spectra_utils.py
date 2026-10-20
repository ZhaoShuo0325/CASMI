# tests/test_spectra_utils.py
"""阶段 2 单元测试：验证 spectra_utils 的 7 个函数。

运行：
    pytest tests/test_spectra_utils.py -v
    python tests/test_spectra_utils.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 把 src 加入路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spectra_utils import (
    spectrum_to_vector,
    vectorize_spectra,
    load_spectra,
    filter_valid_spectra,
    group_by_molecule,
    top_k_peaks,
    MZ_MIN,
    MZ_MAX,
    N_BINS,
)


# ============================================================
# 1. spectrum_to_vector
# ============================================================
def test_spectrum_to_vector_basic():
    """基本功能：几个峰落到正确的箱。"""
    mzs = [100.0, 200.0, 300.0]
    ints = [0.5, 0.8, 0.3]
    vec = spectrum_to_vector(mzs, ints)

    assert vec.shape == (N_BINS,)
    assert vec.dtype == np.float32
    assert (vec > 0).sum() == 3
    assert abs(vec.max() - 0.8) < 1e-6


def test_spectrum_to_vector_empty():
    """空谱图返回全零向量。"""
    vec = spectrum_to_vector([], [])
    assert vec.shape == (N_BINS,)
    assert vec.sum() == 0


def test_spectrum_to_vector_same_bin():
    bin_width = (MZ_MAX - MZ_MIN) / N_BINS      # 0.15
    # 取某个箱的中心
    center = 100.0
    # 偏移 30% 箱宽，保证同箱且远离边界
    mzs = [center, center + 0.3 * bin_width]    # [100.0, 100.045]
    ints = [0.3, 0.7]
    vec = spectrum_to_vector(mzs, ints)

    assert (vec > 0).sum() == 1
    assert abs(vec.max() - 0.7) < 1e-6


def test_spectrum_to_vector_numpy_input():
    """支持 numpy array 输入。"""
    mzs = np.array([100.0, 200.0])
    ints = np.array([0.5, 0.8])
    vec = spectrum_to_vector(mzs, ints)
    assert (vec > 0).sum() == 2


def test_spectrum_to_vector_overflow():
    """超出 MZ_MAX 的峰被 clip 到边界箱，不报错。"""
    mzs = [2000.0, 5000.0]      # 都超出 MZ_MAX=1500
    ints = [0.5, 0.6]
    vec = spectrum_to_vector(mzs, ints)
    assert vec.shape == (N_BINS,)
    # 都落到最后一个箱
    assert vec[-1] > 0


# ============================================================
# 2. top_k_peaks
# ============================================================
def test_top_k_peaks():
    """保留强度最高的 k 个峰，按 m/z 排序。"""
    mzs = [100, 200, 300, 400, 500]
    ints = [0.1, 0.5, 0.9, 0.3, 0.7]
    mz_top, int_top = top_k_peaks(mzs, ints, k=3)

    assert len(mz_top) == 3
    # 强度最高三个: 300(0.9), 500(0.7), 200(0.5)
    assert set(mz_top) == {200, 300, 500}
    # 按 m/z 排序
    assert list(mz_top) == sorted(mz_top)


def test_top_k_peaks_less_than_k():
    """峰数少于 k 时，返回全部。"""
    mzs = [100, 200]
    ints = [0.5, 0.8]
    mz_top, _ = top_k_peaks(mzs, ints, k=10)
    assert len(mz_top) == 2


# ============================================================
# 3. filter_valid_spectra
# ============================================================
def test_filter_valid_spectra_prec_range():
    """过滤异常前体。"""
    df = pd.DataFrame({
        "precursor_mz": [100.0, 3000.0, 200.0, 30.0],
        "ms2_mzs": [[1]*6, [1]*6, [1]*6, [1]*6],
    })
    filtered = filter_valid_spectra(df)
    # 100 和 200 在 [50, 2000]，保留
    # 3000 和 30 超出，丢弃
    assert len(filtered) == 2
    assert list(filtered["precursor_mz"]) == [100.0, 200.0]


def test_filter_valid_spectra_min_peaks():
    """过滤峰数不足的谱图。"""
    df = pd.DataFrame({
        "precursor_mz": [100.0, 200.0, 300.0],
        "ms2_mzs": [[1]*10, [1]*2, [1]*5],   # 10 OK, 2 少, 5 OK
    })
    filtered = filter_valid_spectra(df, min_peaks=5)
    assert len(filtered) == 2
    assert list(filtered["precursor_mz"]) == [100.0, 300.0]


def test_filter_valid_spectra_empty_ms2():
    """空 ms2_mzs 被过滤。"""
    df = pd.DataFrame({
        "precursor_mz": [100.0, 200.0],
        "ms2_mzs": [[1]*10, None],
    })
    filtered = filter_valid_spectra(df)
    assert len(filtered) == 1


# ============================================================
# 4. 真实数据测试（test.parquet 前 20 行）
# ============================================================
TEST_PARQUET = Path(__file__).parent.parent / \
    "data/enveda_casmi26/test.parquet"


def _load_test_df():
    """加载 test.parquet 前 20 行；文件不存在返回 None。"""
    if not TEST_PARQUET.exists():
        return None
    return load_spectra(str(TEST_PARQUET), limit=20, verbose=False)


def test_load_spectra():
    """加载返回的 DataFrame 结构正确。"""
    df = _load_test_df()
    if df is None:
        print("  (跳过：test.parquet 不存在)")
        return
    assert len(df) == 20
    assert "ms2_mzs" in df.columns
    assert "ms2_normalized_intensities" in df.columns
    assert "precursor_mz" in df.columns
    assert "adduct" in df.columns


def test_vectorize_real():
    """批量向量化真实数据。"""
    df = _load_test_df()
    if df is None:
        print("  (跳过：test.parquet 不存在)")
        return
    vecs = vectorize_spectra(df, desc="test", verbose=False)
    assert vecs.shape == (20, N_BINS)
    assert vecs.dtype == np.float32
    # 每张谱图至少有 1 个非零元素
    assert (vecs > 0).any(axis=1).all()
    # 强度 ≤ 1.0（已归一化）
    assert vecs.max() <= 1.0 + 1e-6


def test_group_by_molecule_real():
    """按 molecule_id 分组。"""
    df = _load_test_df()
    if df is None:
        print("  (跳过：test.parquet 不存在)")
        return
    groups = group_by_molecule(df, id_col="molecule_id")
    n_molecules = len(groups)
    assert n_molecules >= 1
    for mol_id, g in groups:
        assert len(g) >= 1


def test_spectrum_to_vector_real():
    """从真实数据取一行，向量化后与原始峰数对比。"""
    df = _load_test_df()
    if df is None:
        print("  (跳过：test.parquet 不存在)")
        return
    row = df.iloc[0]
    vec = spectrum_to_vector(
        row["ms2_mzs"], row["ms2_normalized_intensities"]
    )
    n_peaks = len(row["ms2_mzs"])
    # 分箱后非零元素 ≤ 原始峰数（可能有峰合并）
    assert (vec > 0).sum() <= n_peaks
    # 最大值 = 原始最大强度
    assert abs(vec.max() - max(row["ms2_normalized_intensities"])) < 1e-5


def test_filter_valid_real():
    """对真实数据过滤（应几乎不损失）。"""
    df = _load_test_df()
    if df is None:
        print("  (跳过：test.parquet 不存在)")
        return
    filtered = filter_valid_spectra(df)
    # test 数据质量高，应几乎不丢
    assert len(filtered) >= 18
    assert len(filtered) <= len(df)


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("运行 spectra_utils 单元测试")
    print("=" * 60)

    tests = [
        test_spectrum_to_vector_basic,
        test_spectrum_to_vector_empty,
        test_spectrum_to_vector_same_bin,
        test_spectrum_to_vector_numpy_input,
        test_spectrum_to_vector_overflow,
        test_top_k_peaks,
        test_top_k_peaks_less_than_k,
        test_filter_valid_spectra_prec_range,
        test_filter_valid_spectra_min_peaks,
        test_filter_valid_spectra_empty_ms2,
        test_load_spectra,
        test_vectorize_real,
        test_group_by_molecule_real,
        test_spectrum_to_vector_real,
        test_filter_valid_real,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✅ {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ⚠️  {t.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print("=" * 60)
    print(f"通过: {passed}, 失败: {failed}")
    if failed == 0:
        print("✅ 所有测试通过")
    else:
        print(f"❌ 有 {failed} 个测试失败")