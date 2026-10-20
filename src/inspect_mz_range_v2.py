# src/inspect_mz_range_v2.py
"""
改进版 m/z 探查：先过滤异常，再统计。
用于确定 MZ_MIN / MZ_MAX。
"""

import sys
import time
import numpy as np
import pyarrow.parquet as pq


def inspect_mz_range(path, batch_size=100_000):
    pf = pq.ParquetFile(path)

    # ---- 合理范围（基于化学常识 + 比赛描述）----
    PREC_MIN = 50.0        # 前体最小合理值
    PREC_MAX = 2000.0      # 前体最大合理值（比赛 1159，留余量）
    FRAG_MIN = 5.0         # 碎片最小合理值
    FRAG_MAX = 2000.0      # 碎片最大合理值

    # ---- 统计 ----
    n_rows = 0
    n_anomaly_prec = 0
    n_anomaly_frag = 0
    n_empty = 0

    prec_clean = []        # 干净的前体 m/z
    frag_min_clean = float("inf")
    frag_max_clean = float("-inf")

    rng = np.random.default_rng(42)
    frag_samples = []

    t0 = time.time()
    for batch in pf.iter_batches(
        batch_size=batch_size,
        columns=["precursor_mz", "ms2_mzs"]
    ):
        df = batch.to_pandas()
        n_rows += len(df)

        prec = df["precursor_mz"].dropna().values
        # 过滤异常前体
        mask_ok = (prec >= PREC_MIN) & (prec <= PREC_MAX)
        n_anomaly_prec += (~mask_ok).sum()
        prec_clean.append(prec[mask_ok])

        # 碎片
        for mzs in df["ms2_mzs"]:
            if mzs is None or len(mzs) == 0:
                n_empty += 1
                continue
            arr = np.asarray(mzs, dtype=np.float64)
            # 过滤异常碎片
            mask = (arr >= FRAG_MIN) & (arr <= FRAG_MAX)
            n_anomaly_frag += (~mask).sum()
            arr_clean = arr[mask]
            if len(arr_clean) == 0:
                continue
            frag_min_clean = min(frag_min_clean, float(arr_clean.min()))
            frag_max_clean = max(frag_max_clean, float(arr_clean.max()))
            # 采样
            k = min(1000, len(arr_clean))
            idx = rng.choice(len(arr_clean), size=k, replace=False)
            frag_samples.append(arr_clean[idx])

        print(f"  已处理 {n_rows:,} 行  ({time.time()-t0:.0f} 秒)", end="\r")

    print(f"\n  完成: {n_rows:,} 行，用时 {time.time()-t0:.1f} 秒")

    prec_clean = np.concatenate(prec_clean)
    frag_samples = np.concatenate(frag_samples)

    # ---- 异常统计 ----
    print("\n" + "=" * 60)
    print("数据质量")
    print("=" * 60)
    print(f"  总行数:             {n_rows:>12,}")
    print(f"  空谱图:             {n_empty:>12,}")
    print(f"  异常前体 (滤掉):    {n_anomaly_prec:>12,}  "
          f"({100*n_anomaly_prec/n_rows:.3f}%)")
    print(f"  异常碎片 (滤掉):    {n_anomaly_frag:>12,}")

    # ---- 干净 precursor_mz ----
    print("\n" + "=" * 60)
    print(f"干净的 precursor_mz（过滤到 [{PREC_MIN}, {PREC_MAX}]）")
    print("=" * 60)
    print(f"  count : {len(prec_clean):,}")
    print(f"  min   : {prec_clean.min():.2f}")
    print(f"  max   : {prec_clean.max():.2f}")
    for q in [50, 90, 99, 99.9, 99.99]:
        print(f"  {q:>5}%: {np.percentile(prec_clean, q):.2f}")

    # ---- 干净碎片 ----
    print("\n" + "=" * 60)
    print(f"干净的碎片 m/z（过滤到 [{FRAG_MIN}, {FRAG_MAX}]）")
    print("=" * 60)
    print(f"  全局 min : {frag_min_clean:.2f}")
    print(f"  全局 max : {frag_max_clean:.2f}")
    print(f"  采样数   : {len(frag_samples):,}")
    for q in [0.01, 0.1, 1, 50, 99, 99.9, 99.99]:
        print(f"  {q:>6}%: {np.percentile(frag_samples, q):.2f}")

    # ---- 建议 ----
    print("\n" + "=" * 60)
    print("建议 MZ_MIN / MZ_MAX")
    print("=" * 60)
    frag_p9999 = np.percentile(frag_samples, 99.99)
    print(f"  MZ_MIN = {int(np.floor(frag_min_clean))}  "
          f"(最低碎片 {frag_min_clean:.1f})")
    print(f"  MZ_MAX = {int(np.ceil(frag_p9999 / 100) * 100)}  "
          f"(99.99% 分位数 {frag_p9999:.1f})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python inspect_mz_range_v2.py <parquet路径>")
        sys.exit(1)
    inspect_mz_range(sys.argv[1])