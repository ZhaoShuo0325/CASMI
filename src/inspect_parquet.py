# src/inspect_parquet.py
"""探查一个 Parquet 文件的结构。"""

import sys
import pyarrow.parquet as pq
import pandas as pd


def inspect(path: str, n_rows: int = 3):
    """打印 Parquet 文件的元信息、schema 和前 n 行。"""
    print("=" * 60)
    print(f"文件: {path}")
    print("=" * 60)

    # ---- 元信息 ----
    meta = pq.read_metadata(path)
    print(f"\n【元信息】")
    print(f"  总行数   : {meta.num_rows:,}")
    print(f"  总列数   : {meta.num_columns}")
    print(f"  行组数   : {meta.num_row_groups}")

    # ---- schema（Arrow 类型）----
    pf = pq.ParquetFile(path)
    print(f"\n【Schema（列名 + 类型）】")
    print(pf.schema_arrow)

    # ---- 前 n 行 ----
    print(f"\n【前 {n_rows} 行】")
    batch = next(pf.iter_batches(batch_size=n_rows))
    df = batch.to_pandas()
    print(df.to_string())

    # ---- 每列的统计 ----
    print(f"\n【每列类型 + 非空数量】")
    for col in df.columns:
        non_null = df[col].notna().sum()
        print(f"  {col:20s}  type={str(df[col].dtype):15s}  "
              f"非空={non_null}/{len(df)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python inspect_parquet.py <文件路径>")
        sys.exit(1)
    inspect(sys.argv[1])