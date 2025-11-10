import pandas as pd

units = ["MICU 88-1", "MICU 88-2", "BLS 21", "QRV 7"]
rows = [
    {"unit_designator": "MICU 88-1", "day": "all", "window_start_min": 0, "window_end_min": 1440},
    {"unit_designator": "MICU 88-2", "day": "all", "window_start_min": 0, "window_end_min": 1440},
]
# BLS 21: 8AM–4PM Mon–Fri, 10AM–6PM Sat–Sun
for d in range(5):
    rows.append({"unit_designator": "BLS 21", "day": d, "window_start_min": 480, "window_end_min": 960})
for d in [5,6]:
    rows.append({"unit_designator": "BLS 21", "day": d, "window_start_min": 600, "window_end_min": 1080})
# QRV 7: evenings only
rows.append({"unit_designator": "QRV 7", "day": "all", "window_start_min": 960, "window_end_min": 1440})

pd.DataFrame(rows).to_csv("reference/lemsa_unit_duty.csv", index=False)