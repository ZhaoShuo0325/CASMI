# tests/test_chem_utils.py
"""阶段 1 单元测试：验证 chem_utils 的 6 个函数。"""

import sys
from pathlib import Path

# 把 src 加入路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chem_utils import (
    canonical_smiles,
    get_inchikey14,
    get_molecular_formula,
    is_same_molecule,
    deduplicate_candidates,
    evaluate_rank,
)


def test_canonical_smiles():
    # 同一分子的不同写法，规范化后应一致
    assert canonical_smiles("OC1=CC=CC=C1") == canonical_smiles("c1ccccc1O")
    # 不合法输入
    assert canonical_smiles("这不是SMILES") is None


def test_get_inchikey14():
    # 苯酚
    assert get_inchikey14("c1ccccc1O") == "ISWSIDIOOBJBQZ"
    # 苯的两种写法
    assert get_inchikey14("c1ccccc1") == get_inchikey14("C1=CC=CC=C1")
    # 不合法
    assert get_inchikey14("XXX") is None


def test_get_molecular_formula():
    assert get_molecular_formula("CC(=O)O") == "C2H4O2"   # 乙酸
    assert get_molecular_formula("c1ccccc1") == "C6H6"     # 苯
    assert get_molecular_formula("XXX") is None


def test_is_same_molecule():
    assert is_same_molecule("c1ccccc1O", "OC1=CC=CC=C1") is True
    assert is_same_molecule("c1ccccc1", "Cc1ccccc1") is False
    assert is_same_molecule("XXX", "c1ccccc1") is False


def test_deduplicate_candidates():
    cands = ["c1ccccc1", "C1=CC=CC=C1", "Cc1ccccc1", "c1ccccc1O"]
    # 前两个是同一个分子（苯），应去重
    result = deduplicate_candidates(cands)
    assert len(result) == 3
    assert result[0] == "c1ccccc1"
    assert result[1] == "Cc1ccccc1"
    assert result[2] == "c1ccccc1O"


def test_evaluate_rank():
    # 正确答案在列表第 3 位
    preds = ["Cc1ccccc1", "c1ccccc1O", "c1ccccc1"]
    assert evaluate_rank(preds, "c1ccccc1") == 3
    # 正确答案不在列表里
    assert evaluate_rank(preds, "CC(=O)O") is None
    # 正确答案在第 1 位
    assert evaluate_rank(preds, "Cc1ccccc1") == 1


if __name__ == "__main__":
    test_canonical_smiles()
    test_get_inchikey14()
    test_get_molecular_formula()
    test_is_same_molecule()
    test_deduplicate_candidates()
    test_evaluate_rank()
    print("✅ 所有测试通过")