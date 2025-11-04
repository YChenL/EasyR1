#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Shuffle and split a Parquet dataset into train/test.")
    parser.add_argument(
        "--input",
        default="./vstar30k_visdrone6k_x1y1x2y2.parquet",
        help="Input parquet file path."
    )
    parser.add_argument(
        "--out-dir",
        default="./",
        help="Output directory to save train/test parquet files."
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.10,
        help="Proportion of samples for test split (default: 0.10)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed for shuffling (default: 2025)."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files if they exist."
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.is_file():
        raise FileNotFoundError(f"Input parquet not found: {in_path}")

    # 输出文件名基于输入文件名生成
    stem = in_path.stem
    train_path = out_dir / f"{stem}_train.parquet"
    test_path  = out_dir / f"{stem}_test.parquet"

    if not args.overwrite and (train_path.exists() or test_path.exists()):
        raise FileExistsError(
            f"Output exists. Use --overwrite to replace:\n  {train_path}\n  {test_path}"
        )

    # 读取与洗牌
    # 建议安装 pyarrow： pip install pyarrow
    print(f"Reading: {in_path}")
    df = pd.read_parquet(in_path, engine="pyarrow")

    print(f"Shuffling with seed={args.seed} ...")
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # 切分
    n_total = len(df)
    n_test = int(n_total * args.test_ratio)
    n_train = n_total - n_test
    df_test = df.iloc[:n_test]
    df_train = df.iloc[n_test:]

    # 保存（snappy 是默认压缩）
    print(f"Saving train ({n_train} rows) -> {train_path}")
    df_train.to_parquet(train_path, engine="pyarrow", index=False)

    print(f"Saving test  ({n_test} rows) -> {test_path}")
    df_test.to_parquet(test_path, engine="pyarrow", index=False)

    # 简单一致性校验
    assert len(df_train) + len(df_test) == n_total
    print("Done.")

if __name__ == "__main__":
    main()
