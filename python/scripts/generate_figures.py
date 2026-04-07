from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "output"

TEST_START_DATE = pd.Timestamp("2021-11-25")


def load_csv_with_date_index(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    first_column = data.columns[0]
    if first_column.startswith("Unnamed:") or first_column == "X":
        data = data.rename(columns={first_column: "Date"})
    data["Date"] = pd.to_datetime(data["Date"])
    return data


def configure_axis(ax: plt.Axes, ylabel: str, title: str) -> None:
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.tick_params(axis="x", rotation=45)


def save_current_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_two_series(data: pd.DataFrame, series_map: list[tuple[str, str, str, str]], ylabel: str, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for column, label, color, linestyle in series_map:
        ax.plot(data["Date"], data[column], label=label, color=color, linewidth=1.8, linestyle=linestyle)
    configure_axis(ax, ylabel=ylabel, title=title)
    ax.legend(frameon=False)
    save_current_figure(output_path)


def plot_three_series(data: pd.DataFrame, series_map: list[tuple[str, str, str, str]], ylabel: str, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for column, label, color, linestyle in series_map:
        ax.plot(data["Date"], data[column], label=label, color=color, linewidth=1.8, linestyle=linestyle)
    configure_axis(ax, ylabel=ylabel, title=title)
    ax.legend(frameon=False)
    save_current_figure(output_path)


def build_diebold_li_frame() -> pd.DataFrame:
    yields = load_csv_with_date_index(OUTPUT_DIR / "yields.csv")
    yields = yields.loc[yields["Date"] >= TEST_START_DATE].reset_index(drop=True)

    model = pd.read_csv(OUTPUT_DIR / "MMSEForecasts.csv", header=None)
    maturities = [str(maturity) for maturity in range(48, 301, 6)]
    model.columns = [f"X{maturity}_model" for maturity in maturities]

    if len(model) != len(yields):
        raise ValueError("MMSEForecasts.csv and filtered yields.csv do not have matching lengths.")

    return pd.concat([yields[["Date"]].reset_index(drop=True), yields.add_prefix("X"), model], axis=1)


def plot_diebold_li_figures() -> None:
    data = build_diebold_li_frame()

    configs = [
        ("48", "4y_dieboldli.png", "4 years", "firebrick"),
        ("180", "15y_dieboldli.png", "15 years", "darkseagreen"),
        ("300", "25y_dieboldli.png", "25 years", "mediumorchid"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    legend_handles = None

    for ax, (maturity, filename, title, model_color) in zip(axes, configs):
        ax.plot(data["Date"], data[f"X{maturity}"], label="Actual forward curve", color="steelblue", linewidth=1.8)
        ax.plot(data["Date"], data[f"X{maturity}_model"], label="Diebold Li SSM", color=model_color, linewidth=1.8, linestyle="--")
        configure_axis(ax, ylabel="Real forward curve", title=title)
        legend_handles, legend_labels = ax.get_legend_handles_labels()

        single_data = data[["Date", f"X{maturity}", f"X{maturity}_model"]].copy()
        single_data.columns = ["Date", "Actual forward curve", "Diebold Li SSM"]
        plot_two_series(
            single_data,
            [
                ("Actual forward curve", "Actual forward curve", "steelblue", "-"),
                ("Diebold Li SSM", "Diebold Li SSM", model_color, "--"),
            ],
            ylabel="Real forward curve",
            title=title,
            output_path=OUTPUT_DIR / filename,
        )

    fig.legend(legend_handles, legend_labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Diebold-Li Forecast Comparison Across Selected Maturities", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(OUTPUT_DIR / "diebold_li_panel.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    equiv = load_csv_with_date_index(OUTPUT_DIR / "equiv_rate_merged_dbli.csv")
    plot_two_series(
        equiv,
        [
            ("equiv_rate", "Actual equivalent rate", "steelblue", "-"),
            ("equiv_rate_model_dbli", "Testing equivalent rate", "darkorange", "--"),
        ],
        ylabel="Real equivalent rate",
        title="Equivalent Rate: Diebold-Li Benchmark",
        output_path=OUTPUT_DIR / "equiv_rate_dbli.png",
    )


def plot_lstm_figures() -> None:
    data = load_csv_with_date_index(OUTPUT_DIR / "merged.csv")

    configs = [
        ("48", "4y.png", "4 years", "firebrick", "darkseagreen"),
        ("180", "15y.png", "15 years", "goldenrod", "darkseagreen"),
        ("300", "25y.png", "25 years", "mediumorchid", "darkseagreen"),
    ]

    for maturity, filename, title, train_color, test_color in configs:
        subset = data[
            [
                "Date",
                f"{maturity}_actual",
                f"{maturity}_train",
                f"{maturity}_test",
            ]
        ].copy()
        subset.columns = [
            "Date",
            "Actual forward curve",
            "LSTM (training data)",
            "LSTM (testing data)",
        ]
        plot_three_series(
            subset,
            [
                ("Actual forward curve", "Actual forward curve", "steelblue", "-"),
                ("LSTM (training data)", "LSTM (training data)", train_color, "--"),
                ("LSTM (testing data)", "LSTM (testing data)", test_color, "-"),
            ],
            ylabel="Real forward curve",
            title=title,
            output_path=OUTPUT_DIR / filename,
        )

    equiv = load_csv_with_date_index(OUTPUT_DIR / "equiv_rate_merged.csv")
    equiv["Training equivalent rate"] = equiv["equiv_rate_model"].where(equiv["Date"] < TEST_START_DATE)
    equiv["Testing equivalent rate"] = equiv["equiv_rate_model"].where(equiv["Date"] >= TEST_START_DATE)
    equiv["Actual equivalent rate"] = equiv["equiv_rate"]

    plot_three_series(
        equiv,
        [
            ("Actual equivalent rate", "Actual equivalent rate", "steelblue", "-"),
            ("Training equivalent rate", "Training equivalent rate", "darkorange", "--"),
            ("Testing equivalent rate", "Testing equivalent rate", "darkseagreen", "-"),
        ],
        ylabel="Real equivalent rate",
        title="Equivalent Rate: LSTM",
        output_path=OUTPUT_DIR / "equiv_rate.png",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    plot_diebold_li_figures()
    plot_lstm_figures()
    print(f"Saved figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
