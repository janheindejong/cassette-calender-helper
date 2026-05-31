import base64
import datetime
from email.message import EmailMessage
import json
import os

import anthropic
import pydantic
import dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

dotenv.load_dotenv()

SCOPES = ["https://www.googleapis.com/auth/calendar.readonly", "https://www.googleapis.com/auth/gmail.send"]
CREDENTIALS_FILE = "credentials.json"
TOKEN_FILE = "token.json"
CASSETTE_CALENDAR_NAME = "Cassette Band Intern"
NBS_MARKER = "JH NBS"
SUMMARY_RECIPIENT = "janhein.dejong@gmail.com"


# ── Google Calendar ────────────────────────────────────────────────────────────

def get_credentials() -> Credentials:
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())
    assert isinstance(creds, Credentials)
    return creds


def get_calendar_id(service, name: str) -> str | None:
    for cal in service.calendarList().list().execute().get("items", []):
        if cal.get("summary", "").strip().lower() == name.strip().lower():
            return cal["id"]
    return None


def fetch_events(service, calendar_id: str, start: datetime.datetime, end: datetime.datetime) -> list[dict]:
    return (
        service.events()
        .list(
            calendarId=calendar_id,
            timeMin=start.isoformat() + "Z",
            timeMax=end.isoformat() + "Z",
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
        .get("items", [])
    )


# ── Event helpers ──────────────────────────────────────────────────────────────

def _parse_dt(raw: str) -> datetime.datetime:
    """Parse a dateTime or date string into a naive datetime."""
    try:
        return datetime.datetime.fromisoformat(raw).replace(tzinfo=None)
    except ValueError:
        return datetime.datetime.combine(datetime.date.fromisoformat(raw), datetime.time.min)


def event_dt(event: dict, key: str = "start") -> datetime.datetime:
    raw = event.get(key, {})
    return _parse_dt(raw.get("dateTime", raw.get("date", "")))


def event_time_str(event: dict, key: str = "start") -> str:
    raw = event.get(key, {})
    if "dateTime" in raw:
        return _parse_dt(raw["dateTime"]).strftime("%H:%M")
    return "all day"



def is_covered(event: dict, nbs_intervals: list[tuple[datetime.datetime, datetime.datetime]]) -> bool:
    return any(event_dt(event, "start") < nb_end and event_dt(event, "end") > nb_start for nb_start, nb_end in nbs_intervals)

# ── LLM ───────────────────────────────────────────────────────────────────────

class Assessment(pydantic.BaseModel):
    id: int
    conflict_score: float
    reason: str


def _build_prompt(events: list[dict]) -> str:
    lines = []
    for i, e in enumerate(events):
        d = event_dt(e).strftime("%a %d %b %Y")
        t = event_time_str(e, "start")
        end_t = event_time_str(e, "end")
        title = e.get("summary", "(no title)")
        loc = f" at {e['location']}" if e.get("location") else ""
        lines.append(f'{i + 1}. [{d}] {t}–{end_t}: "{title}"{loc}')
    with open("prompts/base.txt") as f:
        return f.read().replace("<<EVENTS>>", "\n".join(lines))


def assess_events(events: list[dict]) -> list[dict]:
    client = anthropic.Anthropic()
    response = client.messages.parse(
        model="claude-sonnet-4-6",
        max_tokens=4096 * 2,
        messages=[{"role": "user", "content": _build_prompt(events)}],
        output_format=list[Assessment],
    )
    assessments = response.parsed_output
    assert assessments is not None
    return [
        {
            "date": event_dt(events[a.id - 1]).date(),
            "time": event_time_str(events[a.id - 1], "start"),
            "end_time": event_time_str(events[a.id - 1], "end"),
            "title": events[a.id - 1].get("summary", "(no title)"),
            "location": events[a.id - 1].get("location", ""),
            "conflict_score": a.conflict_score,
            "reason": a.reason,
        }
        for a in assessments
    ]


# ── Report helpers ─────────────────────────────────────────────────────────────

def _print_events(events: list[dict]) -> None:
    current_month = None
    for e in sorted(events, key=lambda x: x["date"]):
        month = e["date"].strftime("%B %Y")
        if month != current_month:
            print(f"  ── {month} ──")
            current_month = month
        loc = f"  📍 {e['location']}" if e["location"] else ""
        print(f"  {e['date'].strftime('%a %d')}  {e['time']}–{e['end_time']}  {e['title']} ({e['conflict_score'] * 100:.0f}%) {loc}")
        print(f"         → {e['reason']}")
    print()


# ── Gmail ──────────────────────────────────────────────────────────────

def send_summary(
    gmail,
    to: str,
    n_total: int,
    n_covered: int,
    potential_conflicts: list[dict],
    unlikely_conflicts: list[dict],
) -> None:
    today = datetime.date.today().strftime("%d %b %Y")
    subject = f"Gig conflict audit — {today}"
 
    body_parts = [
        f"Gig conflict audit — {today}",
        f"",
        f"Personal events: {n_total}",
        f"Covered by NBS: {n_covered}",
        f"Assessed: {len(potential_conflicts) + len(unlikely_conflicts)}",
        f"🔴  Potential conflicts: {len(potential_conflicts)}",
        f"🟡  Unlikely conflicts: {len(unlikely_conflicts)}",
    ]
 
    if potential_conflicts:
        body_parts += ["\n<b>🔴 POTENTIAL GIG CONFLICTS</b>", _format_event_block(potential_conflicts)]
    else:
        body_parts.append("\n✅ No potential conflicts — all uncovered events look fine.")
 
    if unlikely_conflicts:
        body_parts += [f"\n<b>🟡 Unlikely conflicts ({len(unlikely_conflicts)} events)</b>", _format_event_block(unlikely_conflicts)]
 
    msg = EmailMessage()
    msg["To"] = to
    msg["From"] = to
    msg["Subject"] = subject
    msg.set_content("\n".join(body_parts))
 
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    gmail.users().messages().send(userId="me", body={"raw": raw}).execute()
    print(f"📧 Summary sent to {to}")

def _format_event_block(events: list[dict]) -> str:
    lines = []
    current_month = None
    for e in sorted(events, key=lambda x: x["date"]):
        month = e["date"].strftime("%B %Y")
        if month != current_month:
            lines.append(f"\n── {month} ──")
            current_month = month
        loc = f"  📍 {e['location']}" if e["location"] else ""
        lines.append(f"  {e['date'].strftime('%a %d')}  {e['time']}–{e['end_time']}  {e['title']} ({e['conflict_score'] * 100:.0f}%) {loc}")
        lines.append(f"         → {e['reason']}")
    return "\n".join(lines)

# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    creds = get_credentials()
    service = build("calendar", "v3", credentials=creds)
    gmail = build("gmail", "v1", credentials=creds)

    now = datetime.datetime.now()
    horizon = now + datetime.timedelta(days=365)

    cassette_id = get_calendar_id(service, CASSETTE_CALENDAR_NAME)
    if not cassette_id:
        print(f"❌ Calendar '{CASSETTE_CALENDAR_NAME}' not found. Available:")
        for cal in service.calendarList().list().execute().get("items", []):
            print(f"  - {cal.get('summary')}")
        return

    personal_events = fetch_events(service, "primary", now, horizon)
    cassette_events = fetch_events(service, cassette_id, now, horizon)

    # Build NBS intervals for overlap checking
    nbs_intervals = [
        (event_dt(e, "start"), event_dt(e, "end"))
        for e in cassette_events
        if NBS_MARKER.lower() in e.get("summary", "").lower()
    ]

    uncovered = [e for e in personal_events if not is_covered(e, nbs_intervals)]
    covered_count = len(personal_events) - len(uncovered)

    if not uncovered:
        print("✅ All personal events already covered by a JH NBS entry.")
        return

    print(f"🤖 Assessing {len(uncovered)} uncovered event(s)...\n")
    assessed = assess_events(uncovered)

    potential_conflicts = [e for e in assessed if e["conflict_score"] >= 0.5]
    unlikely_conflicts = [e for e in assessed if e["conflict_score"] < 0.5]

    print(f"{'═' * 52}")
    print(f"  Gig conflict audit")
    print(f"  Personal events         : {len(personal_events)}")
    print(f"  Covered by NBS          : {covered_count}")
    print(f"  Assessed by Claude      : {len(uncovered)}")
    print(f"  🔴  Above 50%            : {len(potential_conflicts)}")
    print(f"  🟡  Between 20% and 50%  : {len(unlikely_conflicts)}")
    print(f"{'═' * 52}\n")

    if potential_conflicts:
        print("🔴  POTENTIAL GIG CONFLICTS\n")
        _print_events(potential_conflicts)
    else:
        print("✅ No potential conflicts — all uncovered events look fine.\n")

    if SUMMARY_RECIPIENT:
        send_summary(gmail, SUMMARY_RECIPIENT, len(personal_events), covered_count, potential_conflicts, unlikely_conflicts)


if __name__ == "__main__":
    main()