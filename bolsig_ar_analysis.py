from __future__ import annotations

import re
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



DATA_PATH = r"Ar_all.dat"          # 如果脚本和 Ar_all.dat 不在同一目录，改成绝对路径
OUT_DIR = r"bolsig_output"

# COMSOL 一维模型的默认参数
COMSOL_VOLTAGE_V = 200.0
COMSOL_X0_M = 0.016
COMSOL_X1_M = 0.384
GAS_NUMBER_DENSITY_N = 0.329500e23  # 1/m3, 从 BOLSIG+ 输出文件 Conditions 读取到的Ar数密度

# 额外想提取的 E/N 点，单位 Td。
TARGET_EOVERNS_TD = [1, 5, 10, 16.5, 20, 30, 50, 100, 200, 500, 1000]


USE_CONVERGED_ONLY_FOR_TRANSFER = False

# 绘图设置
FIG_DPI = 300
FONT_FAMILY = "Arial"  


# =========================
# 2. 工具函数
# =========================
def safe_float(s: str) -> float:
    """兼容 BOLSIG+ 写法，如 -0.224021-207 -> -0.224021E-207。"""
    s = s.strip()
    if not s:
        return np.nan
    s = s.replace("D", "E")
    try:
        return float(s)
    except ValueError:
        m = re.match(r"^([+-]?\d*\.?\d+)([+-]\d{2,4})$", s)
        if m:
            return float(m.group(1) + "E" + m.group(2))
        return np.nan


def sanitize_filename(name: str, max_len: int = 120) -> str:
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name)
    name = re.sub(r"\s+", "_", name.strip())
    return name[:max_len].strip("_")


def set_plot_style() -> None:
    plt.rcParams["font.family"] = FONT_FAMILY
    plt.rcParams["font.size"] = 9
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.major.width"] = 0.8
    plt.rcParams["ytick.major.width"] = 0.8
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["figure.dpi"] = FIG_DPI


def save_line_plot(
    df: pd.DataFrame,
    x: str,
    ys: List[str],
    out_path: Path,
    xlabel: str,
    ylabel: str,
    title: Optional[str] = None,
    logx: bool = False,
    logy: bool = False,
    legend: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for y in ys:
        if y in df.columns:
            d = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(d) == 0:
                continue
            ax.plot(d[x], d[y], marker="o", markersize=3, linewidth=1.2, label=y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    if legend:
        ax.legend(fontsize=7)
    ax.grid(True, linewidth=0.3, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)


def interp_value(x: np.ndarray, y: np.ndarray, x0: float, log_y: bool = True) -> float:
    """对 y 随 x 插值。y 全为正时默认对 log(y) 插值，更适合跨数量级参数。"""
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if len(x) < 2 or x0 < x.min() or x0 > x.max():
        return np.nan
    if log_y and np.all(y > 0):
        return float(np.exp(np.interp(x0, x, np.log(y))))
    return float(np.interp(x0, x, y))



# 解析 BOLSIG+ 输出

def read_text(path: Path) -> List[str]:
    # BOLSIG+ 输出中可能有本地路径乱码，忽略即可，不影响数据解析
    text = path.read_text(encoding="utf-8", errors="ignore")
    return text.splitlines()


def parse_cross_sections(lines: List[str]) -> pd.DataFrame:
    """解析 Cn Input cross section 块。"""
    records = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = re.match(r"^(C\d+)\s+Input cross section", line)
        if not m:
            i += 1
            continue

        cid = m.group(1)
        ctype = lines[i + 1].strip() if i + 1 < len(lines) else ""
        species = lines[i + 2].strip() if i + 2 < len(lines) else ""
        threshold = np.nan

        # 找第一条分隔线
        j = i + 3
        while j < len(lines) and "--------------------------------" not in lines[j]:
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?", lines[j])
            if nums and np.isnan(threshold):
                threshold = safe_float(nums[0])
            j += 1
        j += 1

        # 读数据到下一条分隔线
        while j < len(lines):
            s = lines[j].strip()
            if "--------------------------------" in s:
                break
            parts = re.split(r"\s+", s)
            if len(parts) >= 2:
                e = safe_float(parts[0])
                val = safe_float(parts[1])
                if np.isfinite(e) and np.isfinite(val):
                    records.append({
                        "collision_id": cid,
                        "type": ctype,
                        "species": species,
                        "threshold_eV": threshold,
                        "energy_eV": e,
                        "cross_section_m2": val,
                    })
            j += 1
        i = j + 1

    return pd.DataFrame(records)


def parse_two_column_tables(lines: List[str]) -> Dict[str, pd.DataFrame]:
    """解析所有形如 `Energy (eV)某参数` 的二列表。"""
    tables: Dict[str, pd.DataFrame] = {}
    last_context = "Global"

    def update_context(idx: int) -> str:
        # 向上找最近的非空、非分隔、非表头行，作为上下文，比如 C42 Ar Ionization 15.80 eV
        for k in range(idx - 1, max(-1, idx - 8), -1):
            s = lines[k].strip()
            if not s or "----" in s:
                continue
            if "Energy (eV)" in s:
                continue
            if re.match(r"^[\d\.Ee+\-]+\s+[\d\.Ee+\-]+$", s):
                continue
            return re.sub(r"\s+", " ", s)
        return "Global"

    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("Energy (eV)"):
            # 表头第二列名称
            parts = re.split(r"\t+|\s{2,}", s)
            if len(parts) >= 2:
                y_name = parts[-1].strip()
            else:
                y_name = s.replace("Energy (eV)", "").strip()
            ctx = update_context(i)
            key = f"{ctx} | {y_name}"
            if key in tables:
                key = f"{key} #{len(tables)+1}"

            rows = []
            j = i + 1
            while j < len(lines):
                t = lines[j].strip()
                if not t:
                    break
                if t.startswith("Energy (eV)") or re.match(r"^C\d+\s+", t) or "Input cross section" in t:
                    break
                vals = re.split(r"\s+", t)
                if len(vals) < 2:
                    break
                xval = safe_float(vals[0])
                yval = safe_float(vals[1])
                if not (np.isfinite(xval) and np.isfinite(yval)):
                    break
                rows.append((xval, yval))
                j += 1
            if rows:
                tables[key] = pd.DataFrame(rows, columns=["mean_energy_eV", y_name])
                i = j
                continue
        i += 1
    return tables


def build_transport_table(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """把 Global 类输运参数合并为一个表。"""
    # 先找到 E/N 映射表
    eover_key = None
    for key in tables:
        if "Electric field / N" in key:
            eover_key = key
            break
    if eover_key is None:
        raise RuntimeError("没有找到 Electric field / N (Td) 表，无法建立 E/N 映射。")

    base = tables[eover_key].copy()
    base = base.rename(columns={base.columns[1]: "EoverN_Td"})

    for key, df in tables.items():
        if key == eover_key:
            continue
        # Global 表一般是输运参数；C1/C2 等是具体反应速率或能量损失，放到 long table 处理
        if re.search(r"\bC\d+\b", key):
            continue
        ycol = df.columns[1]
        clean_name = ycol
        if clean_name in base.columns:
            clean_name = key.split("|")[-1].strip()
        tmp = df.rename(columns={ycol: clean_name})
        base = base.merge(tmp, on="mean_energy_eV", how="left")

    return base.sort_values("EoverN_Td").reset_index(drop=True)


def build_long_table(tables: Dict[str, pd.DataFrame], keyword: str, value_name: str) -> pd.DataFrame:
    """把 Rate coefficient 或 Energy loss coefficient 的 Cn 反应表合成长表。"""
    records = []
    for key, df in tables.items():
        if keyword not in key:
            continue
        if not re.search(r"\bC\d+\b", key):
            continue
        ycol = df.columns[1]
        process = key.split("|")[0].strip()
        for _, row in df.iterrows():
            records.append({
                "process": process,
                "mean_energy_eV": row["mean_energy_eV"],
                value_name: row[ycol],
            })
    return pd.DataFrame(records)


def attach_eovern(long_df: pd.DataFrame, transport_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return long_df
    mapper = transport_df[["mean_energy_eV", "EoverN_Td"]]
    return long_df.merge(mapper, on="mean_energy_eV", how="left")



# 绘图函数

def plot_cross_sections(xs_df: pd.DataFrame, fig_dir: Path) -> None:
    if xs_df.empty:
        return
    # 总览图：所有截面
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for (cid, ctype), d in xs_df.groupby(["collision_id", "type"]):
        d = d.sort_values("energy_eV")
        label = f"{cid} {ctype}"
        ax.plot(d["energy_eV"], d["cross_section_m2"], linewidth=0.8, alpha=0.75, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Electron energy (eV)")
    ax.set_ylabel("Cross section (m$^2$)")
    ax.set_title("Ar electron collision cross sections")
    ax.grid(True, linewidth=0.3, alpha=0.35)
    # 截面很多，图例单独保存会很挤，这里不放总图图例
    fig.tight_layout()
    fig.savefig(fig_dir / "cross_sections_all.png", dpi=FIG_DPI)
    plt.close(fig)

    # 按类型分别画
    for ctype, g in xs_df.groupby("type"):
        fig, ax = plt.subplots(figsize=(5.0, 3.6))
        for cid, d in g.groupby("collision_id"):
            d = d.sort_values("energy_eV")
            ax.plot(d["energy_eV"], d["cross_section_m2"], linewidth=1.0, label=cid)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Electron energy (eV)")
        ax.set_ylabel("Cross section (m$^2$)")
        ax.set_title(f"Cross sections: {ctype}")
        ax.grid(True, linewidth=0.3, alpha=0.35)
        if g["collision_id"].nunique() <= 12:
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / f"cross_sections_{sanitize_filename(ctype)}.png", dpi=FIG_DPI)
        plt.close(fig)


def plot_transport(transport_df: pd.DataFrame, fig_dir: Path) -> None:
    # 每一个输运参数单独画，避免量纲混在一起
    x = "EoverN_Td"
    for col in transport_df.columns:
        if col in ["mean_energy_eV", "EoverN_Td"]:
            continue
        d = transport_df[[x, col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(d) == 0:
            continue
        logy = bool((d[col] > 0).all() and d[col].max() / max(d[col].min(), 1e-300) > 50)
        save_line_plot(
            transport_df, x, [col],
            fig_dir / f"transport_{sanitize_filename(col)}.png",
            xlabel="E/N (Td)", ylabel=col, title=col, logx=True, logy=logy, legend=False
        )

    # 关键传递参数总览
    key_cols = [
        "Mean energy (eV)",
        "Mobility *N (1/m/V/s)",
        "Diffusion coefficient *N (1/m/s)",
        "Townsend ioniz. coef. alpha/N (m2)",
        "Total ionization freq. /N (m3/s)",
    ]
    for col in key_cols:
        if col in transport_df.columns:
            logy = col not in ["Mean energy (eV)"]
            save_line_plot(
                transport_df, x, [col],
                fig_dir / f"transfer_key_{sanitize_filename(col)}.png",
                xlabel="E/N (Td)", ylabel=col, title=f"Transfer parameter: {col}", logx=True, logy=logy, legend=False
            )


def plot_long_reaction_table(long_df: pd.DataFrame, value_col: str, fig_dir: Path, prefix: str) -> None:
    if long_df.empty:
        return
    # 总览图
    fig, ax = plt.subplots(figsize=(5.3, 3.8))
    for proc, d in long_df.groupby("process"):
        d = d.sort_values("EoverN_Td")
        y = d[value_col].replace(0, np.nan)
        ax.plot(d["EoverN_Td"], y, linewidth=0.8, alpha=0.7, label=proc)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("E/N (Td)")
    ax.set_ylabel(value_col)
    ax.set_title(prefix.replace("_", " "))
    ax.grid(True, linewidth=0.3, alpha=0.35)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{prefix}_all.png", dpi=FIG_DPI)
    plt.close(fig)

    # 每个过程单独图
    for proc, d in long_df.groupby("process"):
        d = d.sort_values("EoverN_Td")
        yy = d[value_col].replace(0, np.nan)
        tmp = pd.DataFrame({"EoverN_Td": d["EoverN_Td"], value_col: yy})
        save_line_plot(
            tmp, "EoverN_Td", [value_col],
            fig_dir / f"{prefix}_{sanitize_filename(proc)}.png",
            xlabel="E/N (Td)", ylabel=value_col, title=proc,
            logx=True, logy=True, legend=False
        )



# 传递参数提取

def find_transfer_parameters(
    transport_df: pd.DataFrame,
    target_eoverns_td: List[float],
    gas_density_N: float,
    use_converged_only: bool = False,
) -> pd.DataFrame:
    df = transport_df.copy()
    if use_converged_only and "Error code" in df.columns:
        df = df[df["Error code"] == 0].copy()

    x = df["EoverN_Td"].to_numpy(dtype=float)
    out_rows = []
    cols_to_extract = [
        "Mean energy (eV)",
        "Mobility *N (1/m/V/s)",
        "Diffusion coefficient *N (1/m/s)",
        "Energy mobility *N (1/m/V/s)",
        "Energy diffusion coef. *N (1/m/s)",
        "Townsend ioniz. coef. alpha/N (m2)",
        "Total ionization freq. /N (m3/s)",
        "Power /N (eV m3/s)",
        "Elastic power loss /N (eV m3/s)",
        "Inelastic power loss /N (eV m3/s)",
    ]

    for en in target_eoverns_td:
        row = {"target_EoverN_Td": en}
        for col in cols_to_extract:
            if col in df.columns:
                row[col] = interp_value(x, df[col].to_numpy(dtype=float), en, log_y=(col != "Mean energy (eV)"))

        # 转成 COMSOL 可能直接需要的非约化量
        muN = row.get("Mobility *N (1/m/V/s)", np.nan)
        DN = row.get("Diffusion coefficient *N (1/m/s)", np.nan)
        alphaN = row.get("Townsend ioniz. coef. alpha/N (m2)", np.nan)
        kizN = row.get("Total ionization freq. /N (m3/s)", np.nan)

        row["mu_e_m2_V_s"] = muN / gas_density_N if np.isfinite(muN) else np.nan
        row["D_e_m2_s"] = DN / gas_density_N if np.isfinite(DN) else np.nan
        row["alpha_1_m"] = alphaN * gas_density_N if np.isfinite(alphaN) else np.nan
        # Total ionization freq./N 的量纲等价 m3/s，乘 N 得到 1/s
        row["ionization_frequency_1_s"] = kizN * gas_density_N if np.isfinite(kizN) else np.nan
        out_rows.append(row)

    return pd.DataFrame(out_rows)


def compute_mean_field_eovern(voltage_v: float, x0_m: float, x1_m: float, gas_density_N: float) -> float:
    length = abs(x1_m - x0_m)
    E = voltage_v / length  # V/m, 简化平均电场
    return E / gas_density_N / 1e-21  # Td



# 主程序

def main() -> None:
    set_plot_style()
    data_path = Path(DATA_PATH)
    if not data_path.exists():
        raise FileNotFoundError(f"找不到数据文件：{data_path.resolve()}")

    out_dir = Path(OUT_DIR)
    table_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    lines = read_text(data_path)
    xs_df = parse_cross_sections(lines)
    tables = parse_two_column_tables(lines)
    transport_df = build_transport_table(tables)
    rate_df = attach_eovern(build_long_table(tables, "Rate coefficient", "rate_coefficient_m3_s"), transport_df)
    loss_df = attach_eovern(build_long_table(tables, "Energy loss coefficient", "energy_loss_coeff_eV_m3_s"), transport_df)

    # 保存表格
    xs_df.to_csv(table_dir / "cross_sections.csv", index=False, encoding="utf-8-sig")
    transport_df.to_csv(table_dir / "transport_coefficients.csv", index=False, encoding="utf-8-sig")
    rate_df.to_csv(table_dir / "rate_coefficients_long.csv", index=False, encoding="utf-8-sig")
    loss_df.to_csv(table_dir / "energy_loss_coefficients_long.csv", index=False, encoding="utf-8-sig")

    # 绘图
    plot_cross_sections(xs_df, fig_dir)
    plot_transport(transport_df, fig_dir)
    plot_long_reaction_table(rate_df, "rate_coefficient_m3_s", fig_dir, "rate_coefficients")
    plot_long_reaction_table(loss_df, "energy_loss_coeff_eV_m3_s", fig_dir, "energy_loss_coefficients")

    # 自动计算 COMSOL 平均场对应的 E/N，并加入提取列表
    mean_eovern = compute_mean_field_eovern(
        COMSOL_VOLTAGE_V, COMSOL_X0_M, COMSOL_X1_M, GAS_NUMBER_DENSITY_N
    )
    targets = sorted(set([float(v) for v in TARGET_EOVERNS_TD] + [float(mean_eovern)]))

    transfer_df = find_transfer_parameters(
        transport_df,
        targets,
        gas_density_N=GAS_NUMBER_DENSITY_N,
        use_converged_only=USE_CONVERGED_ONLY_FOR_TRANSFER,
    )
    transfer_df.to_csv(out_dir / "recommended_transfer_parameters.csv", index=False, encoding="utf-8-sig")

    # 单独输出 COMSOL 平均电场对应的一行，方便复制
    nearest = transfer_df.iloc[(transfer_df["target_EoverN_Td"] - mean_eovern).abs().argsort()[:1]]
    nearest.to_csv(out_dir / "comsol_mean_field_transfer_parameter.csv", index=False, encoding="utf-8-sig")

    # 生成简短报告
    report = []
    report.append("BOLSIG+ Ar 数据解析完成")
    report.append(f"输入文件: {data_path.resolve()}")
    report.append(f"截面数据点数: {len(xs_df)}")
    report.append(f"碰撞过程数: {xs_df['collision_id'].nunique() if not xs_df.empty else 0}")
    report.append(f"输运参数点数: {len(transport_df)}")
    report.append(f"速率系数过程数: {rate_df['process'].nunique() if not rate_df.empty else 0}")
    report.append(f"能量损失系数过程数: {loss_df['process'].nunique() if not loss_df.empty else 0}")
    report.append("")
    report.append("COMSOL 平均场估算：")
    report.append(f"  V0 = {COMSOL_VOLTAGE_V:g} V")
    report.append(f"  L = {abs(COMSOL_X1_M - COMSOL_X0_M):.6g} m")
    report.append(f"  N = {GAS_NUMBER_DENSITY_N:.6e} 1/m3")
    report.append(f"  E/N = {mean_eovern:.6g} Td")
    report.append("")
    report.append("最推荐传递给 COMSOL 的参数：")
    report.append("  1) Mobility *N (1/m/V/s)，对应 COMSOL 中 mueN")
    report.append("  2) Diffusion coefficient *N (1/m/s)，可换算 De = DeN/N")
    report.append("  3) Townsend ioniz. coef. alpha/N (m2)，可换算 alpha = alphaN*N")
    report.append("  4) Rate coefficients / Energy loss coefficients，可用于替换或校验反应源项与能量损失项")
    report.append("")
    report.append("COMSOL 平均场对应的传递参数：")
    report.append(nearest.to_string(index=False))

    (out_dir / "analysis_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\n所有输出已保存到: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
