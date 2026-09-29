# src/chem_utils.py
"""
化学信息学工具模块 —— 阶段 1 交付物

提供：
  - SMILES 规范化
  - InChIKey14 计算（比赛评分核心）
  - 分子式提取
  - 分子同一性判断
  - 候选去重
  - 排名评估（MRR 基础单元）
"""

from rdkit import Chem
from rdkit.Chem import inchi, rdMolDescriptors
from rdkit import RDLogger

# 关闭 RDKit 冗余警告
RDLogger.DisableLog("rdApp.*")


# ============================================================
# 1. SMILES 规范化
# ============================================================
def canonical_smiles(smiles: str) -> str | None:
    """
    输入任意 SMILES，返回 RDKit 规范化的 Canonical SMILES。
    不合法返回 None。

    例：
        canonical_smiles("OC1=CC=CC=C1")  -> 'Oc1ccccc1'
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


# ============================================================
# 2. InChIKey14（比赛评分核心）
# ============================================================
def get_inchikey14(smiles: str) -> str | None:
    """
    输入 SMILES，返回标准化 InChIKey 前 14 位（连接层）。
    不合法返回 None。

    注意：
        - 手性、顺反、互变异构不会影响这 14 位
        - 只要原子连接图一致，InChIKey14 就相同
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    key = inchi.MolToInchiKey(mol)
    if not key:
        return None
    return key.split("-")[0]


# ============================================================
# 3. 分子式提取
# ============================================================
def get_molecular_formula(smiles: str) -> str | None:
    """
    返回分子式，例如 'C9H8O4'。不合法返回 None。

    用途：
        - 与质谱推断的分子式比对
        - 计算不饱和度 DBE
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return rdMolDescriptors.CalcMolFormula(mol)


# ============================================================
# 4. 分子同一性判断
# ============================================================
def is_same_molecule(smiles_a: str, smiles_b: str) -> bool:
    """
    判断两个 SMILES 是否表示同一个分子（按 InChIKey14）。
    任一不合法返回 False。
    """
    ka = get_inchikey14(smiles_a)
    kb = get_inchikey14(smiles_b)
    return ka is not None and ka == kb


# ============================================================
# 5. 候选去重
# ============================================================
def deduplicate_candidates(candidates: list[str]) -> list[str]:
    """
    对候选 SMILES 列表去重（按 InChIKey14），保持原有顺序。
    不合法的候选直接丢弃。

    用途：
        比赛只提交 25 个候选，重复的分子会浪费名额。
    """
    seen = set()
    unique = []
    for smi in candidates:
        key = get_inchikey14(smi)
        if key is None:
            continue
        if key in seen:
            continue
        seen.add(key)
        unique.append(smi)
    return unique


# ============================================================
# 6. 排名评估（MRR 基础单元）
# ============================================================
def evaluate_rank(predicted_list: list[str], answer_smiles: str) -> int | None:
    """
    给定按置信度排序的候选列表和正确答案，
    返回正确答案的排名（1-based）。不在列表里返回 None。

    用途：
        rank = evaluate_rank(preds, answer)
        score = 1.0 / rank if rank else 0.0
    """
    target = get_inchikey14(answer_smiles)
    if target is None:
        return None
    for rank, smi in enumerate(predicted_list, start=1):
        if get_inchikey14(smi) == target:
            return rank
    return None