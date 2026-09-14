#!/usr/bin/env python3
"""
WandB Success Rate Plotter with Seaborn
---------------------------------------
WandB에 기록된 실험 결과(3개 이상의 시드)에서 성공률(eval/success_rate)을 가져와
Seaborn을 활용해 평균 실선(Solid line)과 표준편차 음영(Shaded region, ±1 SD) 그래프를 생성합니다.

사용 예시:
1) 특정 Run ID 3개 지정:
   python plot_wandb_success_rate.py \
       --project square_residual_rl \
       --run_ids run_id_1 run_id_2 run_id_3 \
       --save_path square_success_rate.png

2) Group 이름으로 지정:
   python plot_wandb_success_rate.py \
       --project square_residual_rl \
       --group "pbrs_beta1.0" \
       --save_path square_success_rate.png

3) 파이썬 코드에서 함수로 호출:
   from plot_wandb_success_rate import plot_wandb_runs
   plot_wandb_runs(
       project="square_residual_rl",
       run_ids=["id1", "id2", "id3"],
       output_path="success_rate.png"
   )
"""

import argparse
import os
from typing import List, Optional, Dict
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
import wandb


def fetch_run_data(
    run,
    metric_key: str = "eval/success_rate",
    step_key: str = "step",
    method_name: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """단일 WandB Run에서 step과 metric_key 데이터를 추출하여 DataFrame으로 반환합니다."""
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    sanitized_metric = metric_key.replace("/", "_").replace(".", "_")
    cache_file = os.path.join(cache_dir, f"wandb_{run.id}_{sanitized_metric}.csv")

    if use_cache and os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file)
            df["method"] = method_name or run.group or "Default"
            print(f"  -> Loaded {len(df)} rows from cache ({cache_file})", flush=True)
            return df
        except Exception:
            pass

    history_records = []
    step_candidates = [step_key, "_step", "trainer/step", "global_step"]

    try:
        keys_to_fetch = [metric_key] + step_candidates
        for row in run.scan_history(keys=keys_to_fetch):
            if metric_key in row and row[metric_key] is not None:
                s_val = None
                for sk in step_candidates:
                    if sk in row and row[sk] is not None:
                        s_val = row[sk]
                        break
                if s_val is not None:
                    history_records.append({
                        "step": int(s_val),
                        "success_rate": float(row[metric_key]),
                    })
    except Exception as e:
        print(f"Warning: scan_history failed for {run.id} ({e}), falling back to run.history()...")
        hist = run.history(keys=[metric_key] + step_candidates, samples=10000)
        for _, row in hist.iterrows():
            if metric_key in row and not pd.isna(row[metric_key]):
                s_val = None
                for sk in step_candidates:
                    if sk in row and not pd.isna(row[sk]):
                        s_val = row[sk]
                        break
                if s_val is not None:
                    history_records.append({
                        "step": int(s_val),
                        "success_rate": float(row[metric_key]),
                    })

    if not history_records:
        print(f"[Warning] Run {run.id} ({run.name})에서 '{metric_key}' 데이터를 찾을 수 없습니다.")
        return pd.DataFrame()

    df = pd.DataFrame(history_records)

    # 중복 step 처리 (동일 step에 여러 번 로깅된 경우 마지막 값 사용)
    df = df.groupby("step", as_index=False).last()

    # seed 및 메타데이터 추가
    seed = run.config.get("seed", None)
    if seed is None:
        seed = run.name
    df["seed"] = seed
    df["run_id"] = run.id
    df["run_name"] = run.name
    df["method"] = method_name or run.group or "Default"
    print(f"  -> Fetched {len(df)} rows from WandB API", flush=True)

    if use_cache:
        try:
            df.to_csv(cache_file, index=False)
        except Exception:
            pass

    return df


def smooth_data(df: pd.DataFrame, ema_alpha: float = 0.6) -> pd.DataFrame:
    """각 시드별로 성공률 곡선에 지수 이동 평균(EMA) 스무딩을 적용합니다."""
    if ema_alpha >= 1.0 or ema_alpha <= 0.0:
        return df

    df_smoothed = df.copy()
    df_smoothed["success_rate"] = (
        df_smoothed.groupby(["method", "seed"])["success_rate"]
        .transform(lambda x: x.ewm(alpha=ema_alpha).mean())
    )
    return df_smoothed


def plot_wandb_runs(
    project: str,
    entity: Optional[str] = None,
    run_ids: Optional[List[str]] = None,
    run_names: Optional[List[str]] = None,
    group: Optional[str] = None,
    methods_dict: Optional[Dict[str, List[str]]] = None,
    metric_key: str = "eval/success_rate",
    output_path: str = "success_rate_plot.png",
    title: Optional[str] = None,
    xlabel: str = "Environment Steps",
    ylabel: str = "Success Rate (%)",
    as_percentage: bool = True,
    smooth: float = 1.0,
    max_step: Optional[int] = None,
    errorbar: str = "minmax",
    estimator: str = "mean",
    palette: Optional[str] = "tab10",
    figsize: tuple = (8.5, 5.0),
):
    """
    WandB에서 시드별 데이터를 가져와 Seaborn으로 평균 실선 및 표준편차 음영 그래프를 생성합니다.
    """
    api = wandb.Api()
    all_dfs = []
    project_path = f"{entity}/{project}" if entity else project

    # 1. 대상 Run 수집
    if methods_dict:
        for method_name, identifiers in methods_dict.items():
            for item in identifiers:
                matched_run = None
                try:
                    run_path = f"{project_path}/{item}"
                    matched_run = api.run(run_path)
                except Exception:
                    pass
                if matched_run is None:
                    runs_found = list(api.runs(project_path, filters={"display_name": item}))
                    if runs_found:
                        matched_run = runs_found[0]
                    else:
                        for r in api.runs(project_path):
                            if r.name == item:
                                matched_run = r
                                break
                if matched_run is not None:
                    print(f"[{method_name}] Processing {matched_run.id} ({matched_run.name})...", flush=True)
                    df_run = fetch_run_data(matched_run, metric_key=metric_key, method_name=method_name)
                    if not df_run.empty:
                        all_dfs.append(df_run)
                else:
                    print(f"[Warning] '{method_name}'의 Run을 찾지 못했습니다: {item}")
    elif run_ids:
        for rid in run_ids:
            run_path = f"{entity}/{project}/{rid}" if entity else f"{project}/{rid}"
            try:
                run = api.run(run_path)
                df_run = fetch_run_data(run, metric_key=metric_key)
                if not df_run.empty:
                    all_dfs.append(df_run)
            except Exception as e:
                print(f"Failed to fetch run {run_path}: {e}")
    elif run_names:
        project_path = f"{entity}/{project}" if entity else project
        filters = {"display_name": {"$in": run_names}}
        matched_runs = list(api.runs(project_path, filters=filters))
        if not matched_runs:
            all_proj_runs = api.runs(project_path)
            matched_runs = [r for r in all_proj_runs if r.name in run_names]
        if not matched_runs:
            print(f"[Warning] 일치하는 Run 이름을 찾지 못했습니다: {run_names}")
        for run in matched_runs:
            df_run = fetch_run_data(run, metric_key=metric_key)
            if not df_run.empty:
                all_dfs.append(df_run)
    elif group:
        filters = {"group": group}
        runs = api.runs(f"{entity}/{project}" if entity else project, filters=filters)
        for run in runs:
            df_run = fetch_run_data(run, metric_key=metric_key, method_name=group)
            if not df_run.empty:
                all_dfs.append(df_run)
    else:
        raise ValueError("run_ids, run_names, group, 또는 methods_dict 중 하나를 반드시 지정해야 합니다.")

    if not all_dfs:
        print("[Error] 수집된 데이터가 없습니다. 프로젝트명, Run ID 또는 metric 키를 확인해주세요.")
        return

    full_df = pd.concat(all_dfs, ignore_index=True)

    # 1.5. 최대 step 필터링
    if max_step is not None:
        full_df = full_df[full_df["step"] <= max_step]

    # 2. 값 스케일링 (0~1 ➔ 0~100%)
    if as_percentage and full_df["success_rate"].max() <= 1.05:
        full_df["success_rate"] = full_df["success_rate"] * 100.0

    # 3. 스무딩 적용 (선택 사항)
    if smooth < 1.0:
        full_df = smooth_data(full_df, ema_alpha=smooth)

    # 4. Seaborn 스타일 설정 및 플로팅
    sns.set_theme(style="whitegrid", font_scale=1.15)
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8

    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    # 음영 대역(errorbar) 설정
    if errorbar in ("minmax", "range", "min_max"):
        errorbar_arg = ("pi", 100)
    elif errorbar == "sd":
        errorbar_arg = "sd"
    elif errorbar == "se":
        errorbar_arg = "se"
    elif errorbar == "ci":
        errorbar_arg = ("ci", 95)
    else:
        errorbar_arg = errorbar

    has_multiple_methods = full_df["method"].nunique() > 1
    
    sns.lineplot(
        data=full_df,
        x="step",
        y="success_rate",
        hue="method" if has_multiple_methods else None,
        estimator=estimator,   # 중심 실선 (mean 또는 median)
        errorbar=errorbar_arg, # min-max, sd 등 음영 대역
        err_kws={"alpha": 0.22},
        linewidth=2.5,
        palette=palette,
        ax=ax,
    )

    # 5. 축 및 레이아웃 디테일 설정
    ax.set_xlabel(xlabel, fontsize=13, fontweight="bold", labelpad=8)
    ax.set_ylabel(ylabel, fontsize=13, fontweight="bold", labelpad=8)

    # X축 천 단위(k/M) 축약 포맷터
    @ticker.FuncFormatter
    def step_formatter(x, pos):
        if x >= 1e6:
            return f"{x*1e-6:.1f}M"
        elif x >= 1e3:
            return f"{int(x*1e-3)}k"
        else:
            return f"{int(x)}"

    ax.xaxis.set_major_formatter(step_formatter)

    if as_percentage:
        ax.set_ylim(-2, 105)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(100.0, decimals=0))
    else:
        ax.set_ylim(-0.02, 1.05)

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    # 범례 설정
    if has_multiple_methods:
        ax.legend(title="", frameon=True, facecolor="white", edgecolor="#e0e0e0", loc="lower right")

    plt.tight_layout()

    # 6. 저장
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f" 그래프가 성공적으로 저장되었습니다: {output_path}")

    # 시드별 요약 통계 출력
    print("\n--- 데이터 요약 ---")
    summary = (
        full_df.groupby(["method", "step"])["success_rate"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    last_step = summary["step"].max()
    final_stats = summary[summary["step"] == last_step]
    for _, row in final_stats.iterrows():
        method_str = f"[{row['method']}]" if has_multiple_methods else ""
        std_val = f", Std = {row['std']:.2f}%" if pd.notna(row['std']) else ""
        print(f"{method_str} Final Step ({int(row['step'])}): Mean = {row['mean']:.2f}%, Min = {row['min']:.2f}%, Max = {row['max']:.2f}%{std_val} (Seeds: {int(row['count'])})")

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WandB 시드별 성공률 평균/표준편차 Seaborn 그래프 생성기")
    parser.add_argument("--project", type=str, required=True, help="WandB 프로젝트명 (예: square_residual_rl)")
    parser.add_argument("--entity", type=str, default=None, help="WandB Entity (기본: 기본 계정)")
    parser.add_argument("--run_ids", nargs="+", default=None, help="시드별 Run ID 3개 (예: --run_ids abc1234 def5678 ghi9012)")
    parser.add_argument("--run_names", nargs="+", default=None, help="시드별 Run 이름 3개 (예: --run_names 'run_s42' 'run_s43' 'run_s44')")
    parser.add_argument("--group", type=str, default=None, help="WandB Group 이름")
    parser.add_argument("--metric", type=str, default="eval/success_rate", help="성공률 메트릭 키 (기본: eval/success_rate)")
    parser.add_argument("--save_path", type=str, default="success_rate_3seeds.png", help="저장할 이미지 경로")
    parser.add_argument("--title", type=str, default=None, help="그래프 제목")
    parser.add_argument("--smooth", type=float, default=1.0, help="EMA 스무딩 계수 (1.0: 스무딩 안함, 0.6 권장)")
    parser.add_argument("--no_percent", action="store_true", help="0~100%% 변환 없이 0~1 scale로 표시")
    parser.add_argument("--max_step", type=int, default=None, help="최대 step 제한 (예: 300000)")
    parser.add_argument("--errorbar", type=str, default="minmax", choices=["minmax", "sd", "se", "ci"], help="음영 대역 (기본: minmax)")
    parser.add_argument("--estimator", type=str, default="mean", choices=["mean", "median"], help="중심선 (기본: mean)")

    args = parser.parse_args()

    plot_wandb_runs(
        project=args.project,
        entity=args.entity,
        run_ids=args.run_ids,
        run_names=args.run_names,
        group=args.group,
        metric_key=args.metric,
        output_path=args.save_path,
        title=args.title,
        as_percentage=not args.no_percent,
        smooth=args.smooth,
        max_step=args.max_step,
        errorbar=args.errorbar,
        estimator=args.estimator,
    )
