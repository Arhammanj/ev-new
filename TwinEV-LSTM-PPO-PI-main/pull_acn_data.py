"""
Pull and cache ACN-Data charging sessions to a local CSV.
Run this yourself — it reads your token from the environment.
Never share the token or hardcode it.
"""
import os
from datetime import datetime
import pytz
import pandas as pd
from acnportal import acndata

token = os.environ.get("ACN_API_TOKEN")
if not token:
    raise SystemExit("Set ACN_API_TOKEN first:  $env:ACN_API_TOKEN = \"your_token\"")

client = acndata.DataClient(api_token=token)
tz = pytz.timezone("America/Los_Angeles")

SITE  = "caltech"          # options: "caltech", "jpl", "office001"
START = tz.localize(datetime(2019, 1, 1))
END   = tz.localize(datetime(2019, 12, 31))   # full year for a real train/test split

print(f"Pulling {SITE} sessions {START.date()} -> {END.date()} ...")
sessions = list(client.get_sessions_by_time(SITE, START, END))
print(f"Pulled {len(sessions)} sessions")

rows = []
for s in sessions:
    ui = s.get("userInputs")
    requested_kwh = None
    if ui and isinstance(ui, list) and len(ui) > 0:
        requested_kwh = ui[0].get("kWhRequested")
    rows.append({
        "sessionID":       s.get("sessionID"),
        "stationID":       s.get("stationID"),
        "spaceID":         s.get("spaceID"),
        "siteID":          s.get("siteID"),
        "connectionTime":  s.get("connectionTime"),
        "disconnectTime":  s.get("disconnectTime"),
        "doneChargingTime": s.get("doneChargingTime"),
        "kWhDelivered":    s.get("kWhDelivered"),
        "requested_kWh":   requested_kwh,
    })

df = pd.DataFrame(rows)
out = f"acn_{SITE}_2019.csv"
df.to_csv(out, index=False)
print(f"Saved {len(df)} sessions to {out}")

# Quick sanity summary (no token in output — safe to paste to me)
print("\n--- SUMMARY ---")
print(f"Rows: {len(df)}")
print(f"Distinct stations: {df['stationID'].nunique()}")
print(f"kWhDelivered: min {df['kWhDelivered'].min():.1f}  "
      f"mean {df['kWhDelivered'].mean():.1f}  max {df['kWhDelivered'].max():.1f}")
print(f"requested_kWh present: {df['requested_kWh'].notna().sum()} / {len(df)} rows")
print(f"Date range: {df['connectionTime'].min()} -> {df['connectionTime'].max()}")