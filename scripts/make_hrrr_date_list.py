#!/usr/bin/env python3
from datetime import datetime, timedelta
from pathlib import Path
import argparse, os

p = argparse.ArgumentParser()
p.add_argument("start")
p.add_argument("end")
p.add_argument("--output", default=None)
a = p.parse_args()
start = datetime.strptime(a.start, "%Y%m%d")
end = datetime.strptime(a.end, "%Y%m%d")
if end < start:
    raise SystemExit("end precedes start")
out = Path(a.output) if a.output else Path(os.environ.get("DATA_DIR", ".")) / "metadata" / "date_lists" / f"hrrr_{a.start}_{a.end}.txt"
out.parent.mkdir(parents=True, exist_ok=True)
dates = []
d = start
while d <= end:
    dates.append(d.strftime("%Y%m%d"))
    d += timedelta(days=1)
out.write_text("\n".join(dates) + "\n")
print(f"Wrote {len(dates)} dates to {out}")

