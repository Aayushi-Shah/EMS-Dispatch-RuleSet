import numpy as np
import pandas as pd

def kpi_row_from_metrics(m, units_count: int):
    resp = np.array(m["resp_times"]) if m["resp_times"] else np.array([])
    row = {
        "n_calls": m["n_calls"],
        "missed_calls": m["missed_calls"],
        "p50_resp_min": np.percentile(resp, 50) if len(resp) else np.nan,
        "p90_resp_min": np.percentile(resp, 90) if len(resp) else np.nan,
        "avg_resp_min": float(resp.mean()) if len(resp) else np.nan,
        "p50_wait_min": np.percentile(m["wait_minutes"], 50) if m["wait_minutes"] else np.nan,
        "p90_wait_min": np.percentile(m["wait_minutes"], 90) if m["wait_minutes"] else np.nan,
        "avg_onscene_min": np.mean(m["on_scene"]) if m["on_scene"] else np.nan,
        "avg_transport_min": np.mean(m["transport"]) if m["transport"] else np.nan,
        "avg_turnaround_min": np.mean(m["turnaround"]) if m["turnaround"] else np.nan,
        "units": units_count,
    }
    return row

def weighted_aggregate(per_shift_df: pd.DataFrame, units_count: int):
    if per_shift_df.empty:
        return pd.DataFrame({"shifts":[0], "total_calls":[0], "units":[units_count]})
    w = per_shift_df["n_calls"].values
    def wavg(col):
        v = per_shift_df[col].values
        m = ~np.isnan(v)
        return np.average(v[m], weights=w[m]) if m.any() else np.nan
    return pd.DataFrame({
        "shifts":[len(per_shift_df)],
        "total_calls":[int(per_shift_df["n_calls"].sum())],
        "w_avg_p50_resp_min":[wavg("p50_resp_min")],
        "w_avg_p90_resp_min":[wavg("p90_resp_min")],
        "w_avg_avg_resp_min":[wavg("avg_resp_min")],
        "units":[units_count],
    })