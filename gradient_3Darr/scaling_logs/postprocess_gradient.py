#!/usr/bin/env python3
"""
Reproducible post-processing for the 3D gradient benchmark.

Reads:
    scaling_logs/master.csv
    scaling_logs/passed/**/*.out

Generates:
    Figure 1: Linf, L1, L2 errors vs N
    Figure 2: Application time vs nodes

Every plotted measurement is traced to an approved Slurm job ID.
Failed, superseded, and unverified jobs are excluded.
Missing configurations remain missing.
"""

import argparse
import csv
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NODES = (1, 2, 4, 8)
MESHES = (256, 512, 1024)

# (kind, N, nodes, threads, GPUs/node): accepted job ID
ACCEPTED = {
    ("serial", 256, 1, None, None): "42602194",
    ("serial", 512, 1, None, None): "42602195",
    ("serial", 1024, 1, None, None): "42602196",

    ("mpi", 1024, 1, None, None): "42732249",
    ("mpi", 1024, 2, None, None): "42744546",
    ("mpi", 1024, 4, None, None): "42732321",
    ("mpi", 1024, 8, None, None): "42744547",

    ("omp", 1024, 1, 1, None): "42732250",
    ("omp", 1024, 1, 2, None): "42732251",
    ("omp", 1024, 1, 4, None): "42732252",

    ("omp", 1024, 2, 1, None): "42732318",
    ("omp", 1024, 2, 2, None): "42732319",
    ("omp", 1024, 2, 4, None): "42732320",

    ("omp", 1024, 4, 1, None): "42732322",
    ("omp", 1024, 4, 2, None): "42732323",
    ("omp", 1024, 4, 4, None): "42744549",

    ("omp", 1024, 8, 1, None): "42744548",
    ("omp", 1024, 8, 2, None): "42732327",
    ("omp", 1024, 8, 4, None): "42744550",

    ("gpu", 256, 2, None, 2): "42730611",
    ("gpu", 512, 2, None, 2): "42731343",
    ("gpu", 1024, 1, None, 4): "42731623",
    ("gpu", 1024, 2, None, 1): "42731624",
    ("gpu", 1024, 4, None, 1): "42730608",
    ("gpu", 1024, 4, None, 4): "42731625",
}

# label, kind, nodes, threads, GPUs/node, color, marker
FIG1 = [
    ("CPU_serial", "serial", 1, None, None, "#40464c", "o"),
    ("CPU_parallel_MPI_n1", "mpi", 1, None, None, "#1f77b4", "s"),
    ("CPU_parallel_MPI_n2", "mpi", 2, None, None, "#e07828", "^"),
    ("CPU_parallel_MPI_n2_OpenMP_t2", "omp", 2, 2, None, "#784ab1", "D"),
    ("CPU_parallel_MPI_n2_OpenMP_GPU_g2", "gpu", 2, None, 2, "#239b57", "X"),
]

# label, kind, threads, GPUs/node, color, marker
FIG2 = [
    ("CPU_parallel_MPI", "mpi", None, None, "#1f77b4", "s"),
    ("CPU_parallel_MPI_OpenMP_t1", "omp", 1, None, "#e07828", "^"),
    ("CPU_parallel_MPI_OpenMP_t2", "omp", 2, None, "#784ab1", "D"),
    ("CPU_parallel_MPI_OpenMP_t4", "omp", 4, None, "#168f94", "v"),
    ("CPU_parallel_MPI_OpenMP_GPU_g1", "gpu", None, 1, "#239b57", "X"),
    ("CPU_parallel_MPI_OpenMP_GPU_g2", "gpu", None, 2, "#b08f1d", "P"),
    ("CPU_parallel_MPI_OpenMP_GPU_g4", "gpu", None, 4, "#cb4068", "o"),
]


def as_int(value):
    return int(value) if value is not None and str(value).strip() else None


def as_float(value):
    if value is None or not str(value).strip():
        return None
    number = float(str(value).replace("D", "E").replace("d", "e"))
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"Invalid numerical measurement: {value}")
    return number


def read_manifest():
    path = ROOT / "master.csv"

    if not path.is_file():
        raise SystemExit(f"Missing manifest: {path}")

    required = {
        "kind", "job_id", "N", "nodes", "threads",
        "gpus_per_node", "time", "Linf", "L1", "L2", "file"
    }

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or [])

        if missing:
            raise SystemExit(f"Missing manifest fields: {sorted(missing)}")

        raw = list(reader)

    rows = []

    for entry in raw:
        if entry["kind"] not in ("serial", "mpi", "omp", "gpu"):
            continue

        row = dict(entry)

        for field in ("N", "nodes", "threads", "gpus_per_node"):
            row[field] = as_int(row.get(field))

        for field in ("time", "Linf", "L1", "L2"):
            row[field] = as_float(row.get(field))

        log_path = ROOT / row["file"]

        if not log_path.is_file():
            raise ValueError(f"Original log missing: {log_path}")

        text = log_path.read_text(errors="replace")

        if "Verification: PASS" not in text:
            raise ValueError(f"Verification not PASS: {log_path}")

        for field, label in (
            ("time", "Application_seconds"),
            ("Linf", "L_inf"),
            ("L1", "L1"),
            ("L2", "L2"),
        ):
            pattern = (
                r"^\s*" + re.escape(label)
                + r"\s*=\s*([-+\d.EeDd]+)"
            )

            match = re.search(pattern, text, re.MULTILINE)

            if not match:
                raise ValueError(f"Missing {label} in {log_path}")

            measured = as_float(match.group(1))
            recorded = row[field]

            if recorded is None or not math.isclose(
                measured, recorded, rel_tol=1e-6, abs_tol=1e-12
            ):
                raise ValueError(
                    f"{log_path}: {label} mismatch: "
                    f"log={measured}, CSV={recorded}"
                )

            row[field] = measured

        n_match = re.search(
            r"^\s*N\s*=\s*(\d+)", text, re.MULTILINE
        )

        if n_match and int(n_match.group(1)) != row["N"]:
            raise ValueError(f"Mesh size mismatch: {log_path}")

        if row["kind"] == "gpu":
            gpu_match = re.search(
                r"^\s*GPUs_per_rank\s*=\s*(\d+)",
                text, re.MULTILINE
            )

            if gpu_match and int(gpu_match.group(1)) != row["gpus_per_node"]:
                raise ValueError(f"GPU count mismatch: {log_path}")

        rows.append(row)

    print(f"Loaded {len(rows)} passing benchmark records")
    return rows


def select(rows, kind, mesh, nodes, threads=None, gpus=None):
    key = (kind, mesh, nodes, threads, gpus)

    matches = [
        row for row in rows
        if row["kind"] == kind
        and row["N"] == mesh
        and row["nodes"] == nodes
        and (kind != "omp" or row["threads"] == threads)
        and (kind != "gpu" or row["gpus_per_node"] == gpus)
    ]

    approved = ACCEPTED.get(key)

    if approved:
        selected = [
            row for row in matches
            if row["job_id"] == approved
        ]

        if len(selected) != 1:
            raise ValueError(
                f"Expected approved job {approved} for {key}; "
                f"found {len(selected)}"
            )

        return selected[0]

    # Do not silently include GPU jobs without prior validation.
    if kind == "gpu" and matches:
        print(
            f"Unapproved GPU measurement excluded: {key}; "
            f"jobs={[r['job_id'] for r in matches]}"
        )
        return None

    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous configuration {key}; "
            f"jobs={[r['job_id'] for r in matches]}"
        )

    return matches[0] if matches else None


def write_csv(path, rows, columns):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=columns, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def generate(rows, check_only=False):
    fig1 = []
    fig2 = []
    missing = []

    for norm in ("Linf", "L1", "L2"):
        for label, kind, nodes, threads, gpus, _, _ in FIG1:
            for mesh in MESHES:
                result = select(
                    rows, kind, mesh, nodes, threads, gpus
                )

                if result is None:
                    missing.append(
                        f"Figure 1: {norm}, {label}, N={mesh}"
                    )

                fig1.append({
                    "norm": norm,
                    "series": label,
                    "N": mesh,
                    "value": result[norm] if result else "",
                    "job_id": result["job_id"] if result else "",
                    "log_file": result["file"] if result else "MISSING"
                })

    serial = select(rows, "serial", 1024, 1)

    if serial is None:
        raise ValueError("Missing N=1024 serial CPU reference")

    fig2.append({
        "series": "CPU_serial",
        "num_nodes": 1,
        "application_seconds": serial["time"],
        "job_id": serial["job_id"],
        "log_file": serial["file"]
    })

    for label, kind, threads, gpus, _, _ in FIG2:
        for nodes in NODES:
            result = select(
                rows, kind, 1024, nodes, threads, gpus
            )

            if result is None:
                missing.append(
                    f"Figure 2: {label}, nodes={nodes}"
                )

            fig2.append({
                "series": label,
                "num_nodes": nodes,
                "application_seconds": result["time"] if result else "",
                "job_id": result["job_id"] if result else "",
                "log_file": result["file"] if result else "MISSING"
            })

    for row in fig1 + fig2:
        if row["log_file"] != "MISSING" and not row["job_id"]:
            raise ValueError(f"Untraceable measurement: {row}")

    fig1_count = sum(row["value"] != "" for row in fig1) // 3
    fig2_count = sum(row["application_seconds"] != "" for row in fig2)

    print(f"Figure 1: {fig1_count}/15 configurations per norm")
    print(f"Figure 2: {fig2_count}/29 measured points")

    print("\nMissing configurations:")

    for item in sorted(set(missing)):
        print("  ", item)

    if check_only:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = ROOT / "figures"
    out.mkdir(exist_ok=True)

    write_csv(
        out / "figure1_values.csv",
        fig1,
        ["norm", "series", "N", "value", "job_id", "log_file"]
    )

    write_csv(
        out / "figure2_values.csv",
        fig2,
        [
            "series", "num_nodes", "application_seconds",
            "job_id", "log_file"
        ]
    )

    def style(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(which="major", alpha=0.35)
        ax.tick_params(labelsize=10)

    def save(fig, name):
        for extension in ("png", "pdf", "svg"):
            fig.savefig(
                out / f"{name}.{extension}",
                dpi=300,
                bbox_inches="tight"
            )

        plt.close(fig)

    # Figure 1: three error-norm convergence plots.
    for norm, symbol in (
        ("Linf", r"L_\infty"),
        ("L1", r"L_1"),
        ("L2", r"L_2")
    ):
        fig, ax = plt.subplots(figsize=(10, 6))

        for label, kind, nodes, threads, gpus, color, marker in FIG1:
            subset = [
                row for row in fig1
                if row["norm"] == norm and row["series"] == label
            ]

            values = [
                float(row["value"])
                if row["value"] != "" else math.nan
                for row in subset
            ]

            ax.plot(
                MESHES,
                values,
                color=color,
                marker=marker,
                markerfacecolor=color,
                markeredgecolor=color,
                markersize=6,
                linewidth=1.8,
                label=label
            )

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(
            MESHES,
            labels=[str(n) for n in MESHES]
        )

        ax.set_xlabel("N")
        ax.set_ylabel(rf"${symbol}$ error")
        ax.set_title(rf"${symbol}$ error vs mesh size")

        style(ax)

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.17),
            ncol=2,
            fontsize=8,
            frameon=False
        )

        save(fig, f"Fig_1_{norm}")

    # Figure 2: strong scaling at N=1024.
    fig, ax = plt.subplots(figsize=(10.8, 6.4))

    # Serial is one off-scale reference, not a scaling curve.
    ax.plot(
        [1], [serial["time"]],
        linestyle="none",
        marker="o",
        color="#40464c",
        label="CPU_serial"
    )

    for label, kind, threads, gpus, color, marker in FIG2:
        subset = [
            row for row in fig2
            if row["series"] == label
        ]

        times = [
            float(row["application_seconds"])
            if row["application_seconds"] != ""
            else math.nan
            for row in subset
        ]

        # NaN breaks connecting lines across missing measurements.
        ax.plot(
            NODES,
            times,
            color=color,
            marker=marker,
            markerfacecolor=color,
            markeredgecolor=color,
            markersize=6,
            linewidth=1.8,
            linestyle="--" if kind == "gpu" else "-",
            label=label
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")

    ax.set_xlim(0.95, 8.4)
    ax.set_ylim(0.3, 3)

    ax.set_xticks(
        NODES,
        labels=["1", "2", "4", "8"]
    )

    ax.set_yticks(
        [0.3, 0.5, 1, 2, 3],
        labels=["0.3", "0.5", "1", "2", "3"]
    )

    ax.minorticks_off()

    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Application time (s)")
    ax.set_title("Strong scaling, N = 1024³")

    ax.text(
        0.01,
        0.98,
        f"CPU_serial: {serial['time']:.5f} s (outside Y-axis range)",
        transform=ax.transAxes,
        va="top",
        fontsize=9
    )

    style(ax)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=2,
        fontsize=8,
        frameon=False
    )

    save(fig, "Fig_2_scaling")

    print(f"\nFigures and CSVs written to {out}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate measurements without generating figures"
    )

    args = parser.parse_args()

    generate(
        read_manifest(),
        check_only=args.check_only
    )


if __name__ == "__main__":
    main()
