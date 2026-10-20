# src/spectra_utils.py
"""
质谱工具模块 —— 适配 CASMI 2026 真实数据

数据格式（train.parquet / test.parquet）：
    每行 = 一张谱图，关键列：
        - ms2_mzs                     : list<double>  碎片 m/z
        - ms2_normalized_intensities  : list<double>  归一化强度（0~1）
        - precursor_mz                : double        前体 m/z
        - adduct                      : string        加合物类型

全局配置依据（见 src/inspect_mz_range_v2.py 的探查结果）：
    - train 254 万张谱图，401M 个碎片
    - 碎片 m/z：min=5, 50%=161, 99.9%=1453, 99.99%=1918
    - 前体 m/z：min=57, 50%=350, 99%=976, 99.9%=1383
    - 异常：77 个前体（0.003%），20553 个碎片（0.005%）
    - 比赛 test：分子量 157-1159 Da

提供：
    1. spectrum_to_vector      单张谱图 → 固定长度向量
    2. vectorize_spectra       批量向量化 DataFrame
    3. load_spectra            分块加载 Parquet
    4. filter_valid_spectra    过滤异常谱图
    5. group_by_molecule       按分子 ID 分组（多谱聚合）
    6. top_k_peaks             保留强度最高的 k 个峰
    7. plot_spectrum_from_row  从 DataFrame 行画谱图
"""

import time
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt


# ============================================================
# 全局配置
# ============================================================
MZ_MIN = 0.0            # 碎片最低 ~5，0 覆盖所有
MZ_MAX = 1500.0         # 99.9% 碎片 < 1453，test 前体 ≤ 1159
N_BINS = 10000          # 每箱宽度 = (1500-0)/10000 = 0.15 Da
BATCH_SIZE = 100_000

# 谱图相关的列（训练和测试共有）
SPECTRUM_COLS = [
    "ms2_mzs",
    "ms2_normalized_intensities",
    "precursor_mz",
    "adduct",
]


# ============================================================
# 1. 单张谱图 → 固定长度向量
# ============================================================
def spectrum_to_vector(mzs, intensities,
                       mz_min: float = MZ_MIN,
                       mz_max: float = MZ_MAX,
                       n_bins: int = N_BINS) -> np.ndarray:
    """
    把一张谱图转成固定长度向量。

    方法：m/z 分箱（binning），每箱取最大强度。
        一个峰 (m/z, intensity)
            → bin_idx = (m/z - mz_min) / (mz_max - mz_min) * n_bins
        若两个峰落到同一箱，取较大强度（不累加，符合质谱物理）。

    参数：
        mzs         : 碎片 m/z（list 或 np.ndarray）
        intensities : 归一化强度（0~1）
        mz_min      : m/z 下界
        mz_max      : m/z 上界
        n_bins      : 分箱数

    返回：
        np.ndarray，形状 (n_bins,)，dtype float32
    """
    vec = np.zeros(n_bins, dtype=np.float32)
    if mzs is None or len(mzs) == 0:
        return vec

    mzs = np.asarray(mzs, dtype=np.float64)
    intensities = np.asarray(intensities, dtype=np.float32)

    bin_idx = ((mzs - mz_min) / (mz_max - mz_min) * n_bins).astype(np.int32)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    # np.maximum.at：散射取最大，同箱多峰取较大值
    np.maximum.at(vec, bin_idx, intensities)
    return vec


# ============================================================
# 2. 批量向量化 DataFrame
# ============================================================
def vectorize_spectra(df: pd.DataFrame,
                      mz_col: str = "ms2_mzs",
                      int_col: str = "ms2_normalized_intensities",
                      desc: str = "谱图",
                      verbose: bool = True) -> np.ndarray:
    """
    对 DataFrame 每行（一张谱图）做向量化。

    参数：
        df      : 含 mz_col 和 int_col 的 DataFrame
        mz_col  : m/z 列名
        int_col : 强度列名
        desc    : 进度显示用名
        verbose : 是否打印进度

    返回：
        np.ndarray，形状 (len(df), N_BINS)，dtype float32
    """
    n = len(df)
    vecs = np.zeros((n, N_BINS), dtype=np.float32)
    t0 = time.time()

    for i, (mzs, ints) in enumerate(zip(df[mz_col], df[int_col])):
        vecs[i] = spectrum_to_vector(mzs, ints)
        if verbose and (i + 1) % 50000 == 0:
            elapsed = time.time() - t0
            speed = (i + 1) / elapsed
            eta = (n - i - 1) / speed
            print(f"  向量化{desc}: {i+1:,}/{n:,}  "
                  f"({speed:.0f} 张/秒, 剩 {eta:.0f} 秒)", end="\r")

    if verbose:
        print(f"\n  向量化{desc}完成: {n:,} 张，"
              f"用时 {time.time()-t0:.1f} 秒")
    return vecs


# ============================================================
# 3. 分块加载 Parquet
# ============================================================
def load_spectra(path: str,
                 columns: list[str] | None = None,
                 limit: int | None = None,
                 batch_size: int = BATCH_SIZE,
                 verbose: bool = True) -> pd.DataFrame:
    """
    分块加载 Parquet，避免一次性读入 2.9 GB 爆内存。

    参数：
        path       : Parquet 路径
        columns    : 只读这些列（None = 全部）
        limit      : 最多读多少行（None = 全部）
        batch_size : 每批行数
        verbose    : 是否打印进度

    返回：
        pd.DataFrame
    """
    pf = pq.ParquetFile(path)
    parts = []
    total = 0
    t0 = time.time()

    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        parts.append(batch.to_pandas())
        total += len(parts[-1])
        if verbose:
            cap = f"/{limit:,}" if limit else ""
            print(f"  加载: {total:,}{cap}", end="\r")
        if limit is not None and total >= limit:
            break

    if verbose:
        print(f"\n  加载完成: {total:,} 行，"
              f"用时 {time.time()-t0:.1f} 秒")

    df = pd.concat(parts, ignore_index=True)
    if limit is not None:
        df = df.head(limit)
    return df


# ============================================================
# 4. 过滤异常谱图
# ============================================================
def filter_valid_spectra(df: pd.DataFrame,
                         prec_min: float = 50.0,
                         prec_max: float = 2000.0,
                         min_peaks: int = 5) -> pd.DataFrame:
    """
    过滤异常谱图。

    过滤条件：
        1. precursor_mz ∈ [prec_min, prec_max]
            - 下界 50  ：排除甲醇（32）、乙醇（46）等小分子
                        （比赛 test 分子量最小 157）
            - 上界 2000：排除大分子（多肽、糖苷）+ 录入错误
                        （比赛 test 分子量最大 1159）
        2. 至少有 min_peaks 个峰
            - 排除空谱、单峰谱

    数据依据（train 254 万行探查）：
        - 77 个异常前体（0.003%）：小分子 / 大分子 / 录入错误
        - 20553 个异常碎片（0.005%）：m/z < 5 或 > 2000

    参数：
        df        : 含 precursor_mz 和 ms2_mzs 的 DataFrame
        prec_min  : 前体 m/z 下界
        prec_max  : 前体 m/z 上界
        min_peaks : 最少峰数

    返回：
        过滤后的 DataFrame（索引已重置）
    """
    mask_prec = df["precursor_mz"].between(prec_min, prec_max)
    mask_peaks = df["ms2_mzs"].apply(
        lambda x: x is not None and len(x) >= min_peaks
    )
    return df[mask_prec & mask_peaks].reset_index(drop=True)


# ============================================================
# 5. 按分子分组
# ============================================================
def group_by_molecule(df: pd.DataFrame, id_col: str = "inchikey14"):
    """
    按分子 ID 分组，返回 groupby 对象。

    注意：
        - train 用 inchikey14 分组（化学标识）
        - test  用 molecule_id 分组（比赛 ID）

    用法：
        for mol_id, group in group_by_molecule(train):
            # group 是该分子的所有谱图（可能多行）
            ...
    """
    return df.groupby(id_col, sort=False)


# ============================================================
# 6. 峰筛选
# ============================================================
def top_k_peaks(mzs: Iterable[float],
                intensities: Iterable[float],
                k: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """
    保留强度最高的 k 个峰，按 m/z 排序返回。

    返回：
        (mz_array, intensity_array)
    """
    mzs = np.asarray(mzs, dtype=np.float64)
    intensities = np.asarray(intensities, dtype=np.float32)

    if len(mzs) <= k:
        order = np.argsort(mzs)
    else:
        idx = np.argsort(intensities)[::-1][:k]
        order = idx[np.argsort(mzs[idx])]

    return mzs[order], intensities[order]


# ============================================================
# 7. 可视化
# ============================================================
def plot_spectrum_from_row(row, ax=None,
                           title: str | None = None,
                           top_k: int | None = None):
    """
    从 DataFrame 的一行画一张谱图。

    参数：
        row   : pd.Series，含 ms2_mzs / ms2_normalized_intensities
        ax    : matplotlib Axes（None = 新建）
        title : 标题
        top_k : 只画强度最高的 k 个峰

    用法：
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_spectrum_from_row(test.iloc[0], ax=ax)
        plt.show()
    """
    mzs = np.asarray(row["ms2_mzs"])
    intens = np.asarray(row["ms2_normalized_intensities"])

    if top_k is not None:
        mzs, intens = top_k_peaks(mzs, intens, k=top_k)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    ax.vlines(mzs, 0, intens, colors="steelblue", linewidth=1.2)
    ax.set_xlabel("m/z")
    ax.set_ylabel("Normalized Intensity")
    ax.set_xlim(MZ_MIN, MZ_MAX)

    if title is None:
        p = row.get("precursor_mz", None)
        title = f"precursor_mz={p:.4f}" if p is not None else "MS/MS"
        if "adduct" in row and row["adduct"] is not None:
            title += f"  adduct={row['adduct']}"
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return ax