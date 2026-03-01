"""explore_dataset tool — compute column-level statistics."""

import json
from collections import Counter
from typing import Annotated, Any

from dataagent_benchmark.domain.tool_context import PerDataset


def explore_dataset(
    dataset_id: Annotated[str, "Dataset to explore"],
    method: Annotated[str, "Method: describe, nullity, value_counts, or column_stats"] = "describe",
    column: Annotated[str, "Target column (required for value_counts and column_stats)"] = "",
    # --- Injected by env ---
    dataset: Annotated[Any, PerDataset("working")] = None,
    text_column: Annotated[str, PerDataset("text_column")] = "",
    col_stats: Annotated[dict, PerDataset("column_stats")] = None,
    transforms: Annotated[list, PerDataset("transforms")] = None,
) -> str:
    """Compute real column statistics.

    Run BEFORE and AFTER transforms to verify improvements.
    """
    valid_methods = ("describe", "nullity", "value_counts", "column_stats")
    if method not in valid_methods:
        return json.dumps({"error": f"Unknown method '{method}'. Use: {', '.join(valid_methods)}"})

    ds = dataset
    current_rows = len(ds)

    if method == "describe":
        columns: dict[str, dict] = {}
        for col_name, stats in col_stats.items():
            col_info: dict = {"dtype": stats["dtype"], "count": stats["count"]}
            if stats["dtype"] == "str":
                col_info["null_count"] = stats["null_count"]
                col_info["empty_count"] = stats["empty_count"]
                col_info["length"] = {
                    "mean": stats["len_mean"],
                    "std": stats["len_std"],
                    "min": stats["len_min"],
                    "25%": stats["len_p25"],
                    "50%": stats["len_p50"],
                    "75%": stats["len_p75"],
                    "max": stats["len_max"],
                }
                if stats.get("patterns"):
                    col_info["patterns"] = {
                        k: f"{v * 100:.1f}%" for k, v in stats["patterns"].items()
                    }
            elif stats["dtype"] == "categorical":
                col_info["unique"] = stats["unique"]
                col_info["null_count"] = stats.get("null_count", 0)
            columns[col_name] = col_info
        return json.dumps(
            {
                "dataset": dataset_id,
                "total_rows": current_rows,
                "columns": columns,
                "transforms_applied": len(transforms),
            },
            indent=2,
        )

    elif method == "nullity":
        columns = {}
        for col_name, stats in col_stats.items():
            total = stats["count"]
            null_ct = stats.get("null_count", 0)
            empty_ct = stats.get("empty_count", 0)
            non_null = total - null_ct
            fill_rate = non_null / max(total, 1) * 100
            completeness = (total - null_ct - empty_ct) / max(total, 1) * 100
            columns[col_name] = {
                "null": null_ct,
                "empty": empty_ct,
                "fill_rate": f"{fill_rate:.1f}%",
                "completeness": f"{completeness:.1f}%",
            }
        return json.dumps(
            {"dataset": dataset_id, "total_rows": current_rows, "columns": columns},
            indent=2,
        )

    elif method == "value_counts":
        stats = col_stats.get(column)
        if stats is None or stats.get("dtype") != "categorical":
            cat_cols = [c for c, s in col_stats.items() if s.get("dtype") == "categorical"]
            return json.dumps(
                {
                    "error": (f"No value counts for column '{column}' in '{dataset_id}'."),
                    "categorical_columns": cat_cols,
                }
            )
        values = ds[column]
        counts = Counter(v for v in values if isinstance(v, str) and v.strip())
        top = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        top_list = [
            {
                "value": val,
                "count": cnt,
                "percentage": f"{cnt / max(current_rows, 1) * 100:.1f}%",
            }
            for val, cnt in top
        ]
        return json.dumps(
            {
                "dataset": dataset_id,
                "column": column,
                "total_rows": current_rows,
                "unique_values": len(counts),
                "top_values": top_list,
            },
            indent=2,
        )

    else:  # column_stats
        stats = col_stats.get(column)
        if stats is None:
            return json.dumps(
                {
                    "error": f"Unknown column '{column}' in '{dataset_id}'.",
                    "available_columns": list(col_stats.keys()),
                }
            )
        info: dict = {
            "dataset": dataset_id,
            "column": column,
            "dtype": stats["dtype"],
            "count": stats["count"],
        }
        if stats["dtype"] == "str":
            info["null_count"] = stats["null_count"]
            info["empty_count"] = stats["empty_count"]
            info["length"] = {
                "mean": stats["len_mean"],
                "std": stats["len_std"],
                "min": stats["len_min"],
                "25%": stats["len_p25"],
                "50%": stats["len_p50"],
                "75%": stats["len_p75"],
                "max": stats["len_max"],
            }
            if stats.get("patterns"):
                info["patterns"] = {k: f"{v * 100:.1f}%" for k, v in stats["patterns"].items()}
        elif stats["dtype"] == "categorical":
            info["unique"] = stats["unique"]
            values = ds[column]
            counts = Counter(v for v in values if isinstance(v, str) and v.strip())
            top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
            if top:
                info["top_values"] = [{"value": v, "count": c} for v, c in top]
        return json.dumps(info, indent=2)
