import os
from datetime import datetime
import pytz
from acnportal import acndata

token = os.environ.get("ACN_API_TOKEN")
if not token:
    raise SystemExit("ACN_API_TOKEN not set. Set it first (see note below).")

client = acndata.DataClient(api_token=token)

tz = pytz.timezone("America/Los_Angeles")
start = tz.localize(datetime(2019, 4, 1))
end   = tz.localize(datetime(2019, 4, 8))   # one week to start small

sessions = list(client.get_sessions_by_time("caltech", start, end))

print(f"Pulled {len(sessions)} sessions")
if sessions:
    s = sessions[0]
    print("Example session keys:", list(s.keys()))
    print("connectionTime :", s.get("connectionTime"))
    print("disconnectTime :", s.get("disconnectTime"))
    print("kWhDelivered   :", s.get("kWhDelivered"))
    print("spaceID        :", s.get("spaceID"))
    