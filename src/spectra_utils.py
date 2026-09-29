# src/spectra_utils.py
"""
质谱工具模块 —— 阶段 2 交付物

提供：
  - 谱图数据结构 Spectrum
  - 从 SMILES 生成模拟碎片谱图（不依赖外部数据）
  - 谱图可视化
  - 多谱聚合
"""

from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


# ============================================================
# 1. 谱图数据结构
# ============================================================
@dataclass
class Spectrum:
    """
    一张 MS/MS 谱图。

    属性：
        precursor_mz : 前体离子质荷比
        adduct       : 加合物类型，如 '[M+H]+'
        ce           : 碰撞能量（eV），未知填 0
        mz           : 碎片 m/z 数组
        intensity    : 碎片强度数组
        smiles       : 对应的分子 SMILES（如果有）
    """
    precursor_mz: float
    adduct: str
    ce: float
    mz: np.ndarray
    intensity: np.ndarray
    smiles: str | None = None

    def __len__(self):
        return len(self.mz)

    def base_peak(self):
        """返回 base peak 的 (m/z, intensity)。"""
        idx = int(np.argmax(self.intensity))
        return float(self.mz[idx]), float(self.intensity[idx])

    def normalized(self):
        """强度归一化：除以 base peak。"""
        bp = self.intensity.max()
        if bp <= 0:
            return self
        return Spectrum(
            precursor_mz=self.precursor_mz,
            adduct=self.adduct,
            ce=self.ce,
            mz=self.mz.copy(),
            intensity=self.intensity / bp,
            smiles=self.smiles,
        )

    def top_k_peaks(self, k: int = 50):
        """取强度最高的 k 个峰，按 m/z 排序。"""
        if len(self.mz) <= k:
            idx = np.argsort(self.mz)
        else:
            idx = np.argsort(self.intensity)[::-1][:k]
            idx = idx[np.argsort(self.mz[idx])]
        return Spectrum(
            precursor_mz=self.precursor_mz,
            adduct=self.adduct,
            ce=self.ce,
            mz=self.mz[idx],
            intensity=self.intensity[idx],
            smiles=self.smiles,
        )


# ============================================================
# 2. 从 SMILES 生成模拟碎片谱图（不依赖外部数据）
# ============================================================
# 常见中性丢失的质量
NEUTRAL_LOSSES = {
    "H2O": 18.01056,
    "CO":  27.99491,
    "CO2": 43.98983,
    "NH3": 17.02655,
    "C2H4": 28.03130,
    "HCN": 27.01090,
    "SO2": 63.96190,
}

# 加合物质量偏移
ADDUCT_MASS = {
    "[M+H]+":   1.00728,
    "[M+Na]+":  22.98922,
    "[M-H]-":  -1.00728,
    "[M+NH4]+": 18.03383,
}


def simulate_spectrum(smiles: str,
                      adduct: str = "[M+H]+",
                      ce: float = 30.0,
                      n_frags: int = 10,
                      seed: int = 42) -> Spectrum | None:
    """
    从 SMILES 生成一张模拟 MS/MS 谱图。

    方法：
        1. 计算分子量，加合得到前体 m/z
        2. 随机挑 n_frags 个中性丢失，得到碎片 m/z
        3. 随机赋强度（base peak = 1000）

    这是"玩具版"模拟，只为理解谱图结构，不追求化学真实。
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    rng = np.random.default_rng(seed)
    mw = Descriptors.ExactMolWt(mol)
    if adduct not in ADDUCT_MASS:
        return None
    precursor_mz = mw + ADDUCT_MASS[adduct]

    # 随机挑几个中性丢失
    losses = list(NEUTRAL_LOSSES.values())
    chosen = rng.choice(losses, size=min(n_frags, len(losses)), replace=False)

    frag_mz = precursor_mz - chosen
    frag_mz = frag_mz[frag_mz > 10]  # 去掉不合理的碎片

    # 随机强度，base peak = 1000
    intensity = rng.uniform(50, 1000, size=len(frag_mz))
    intensity = intensity / intensity.max() * 1000

    # 按 m/z 排序
    order = np.argsort(frag_mz)
    return Spectrum(
        precursor_mz=float(precursor_mz),
        adduct=adduct,
        ce=ce,
        mz=frag_mz[order],
        intensity=intensity[order],
        smiles=smiles,
    )


# ============================================================
# 3. 谱图可视化
# ============================================================
def plot_spectrum(spec: Spectrum, ax=None, title: str | None = None):
    """把一张谱图画成柱状图。"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    ax.vlines(spec.mz, 0, spec.intensity, colors="steelblue", linewidth=1.5)
    ax.set_xlabel("m/z")
    ax.set_ylabel("Intensity")
    ax.set_title(title or f"MS/MS  adduct={spec.adduct}  CE={spec.ce}")
    ax.grid(alpha=0.3)
    return ax


# ============================================================
# 4. 多谱聚合
# ============================================================
def aggregate_spectra(spectra: list[Spectrum],
                      method: str = "concat") -> Spectrum:
    """
    把同一分子的多张谱图聚合成一张。

    method:
        'concat' : 把所有峰拼接，按 m/z 排序
        'mean'   : 按 m/z 分箱后取平均
    """
    if not spectra:
        raise ValueError("spectra 不能为空")

    if method == "concat":
        all_mz = np.concatenate([s.mz for s in spectra])
        all_int = np.concatenate([s.intensity for s in spectra])
        order = np.argsort(all_mz)
        return Spectrum(
            precursor_mz=spectra[0].precursor_mz,
            adduct=spectra[0].adduct,
            ce=0.0,
            mz=all_mz[order],
            intensity=all_int[order],
            smiles=spectra[0].smiles,
        )
    elif method == "mean":
        # 简单分箱：按 0.1 Da 分箱取平均
        bins = {}
        for s in spectra:
            for mz, inten in zip(s.mz, s.intensity):
                key = round(mz / 0.1) * 0.1
                bins.setdefault(key, []).append(inten)
        keys = sorted(bins.keys())
        mz_arr = np.array(keys)
        int_arr = np.array([np.mean(bins[k]) for k in keys])
        return Spectrum(
            precursor_mz=spectra[0].precursor_mz,
            adduct=spectra[0].adduct,
            ce=0.0,
            mz=mz_arr,
            intensity=int_arr,
            smiles=spectra[0].smiles,
        )
    else:
        raise ValueError(f"未知 method: {method}")