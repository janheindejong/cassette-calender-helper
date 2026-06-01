import base64
import dataclasses
import datetime
from email.message import EmailMessage
from enum import Enum
import logging
import os
import zoneinfo

import anthropic
import pydantic
import dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from pydantic.dataclasses import dataclass

dotenv.load_dotenv()

# Configuration
CREDENTIALS_FILE = "credentials.json"
TOKEN_FILE = "token.json"
SUMMARY_RECIPIENT = "janhein.dejong@gmail.com"
CLAUDE_MODEL = "claude-sonnet-4-6"

# Fixed configuration 
CASSETTE_CALENDAR_NAME = "Cassette Band Intern"
NBS_MARKER = "JH NBS"
SCOPES = ["https://www.googleapis.com/auth/calendar.readonly", "https://www.googleapis.com/auth/gmail.send"]


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ── Google Calendar ────────────────────────────────────────────────────────────

class AlreadyFlaggedWithNBS:
    pass 


class OmittedFromAssessmentOutput: 
    pass


class NbsIssueSeverity(Enum):
    UNKNOWN = -2
    ALREADY_FLAGGED = -1
    LOW = 0
    MEDIUM = 1
    HIGH = 2


@dataclasses.dataclass
class PossibleNBS: 
    score: float 
    reason: str 


@dataclasses.dataclass
class CalendarEvent: 
    id: int 
    start: datetime.datetime
    end: datetime.datetime
    summary: str 
    location: str | None = None
    nbs_state: None | AlreadyFlaggedWithNBS | OmittedFromAssessmentOutput | PossibleNBS = None

    @property
    def conflict_severity(self) -> NbsIssueSeverity:
        match self.nbs_state:
            case AlreadyFlaggedWithNBS():
                return NbsIssueSeverity.ALREADY_FLAGGED
            case PossibleNBS(score=score) if score >= 0.5:
                return NbsIssueSeverity.HIGH
            case PossibleNBS(score=score) if score >= 0.2:
                return NbsIssueSeverity.MEDIUM
            case PossibleNBS(score=score):
                return NbsIssueSeverity.LOW
            case OmittedFromAssessmentOutput():
                return NbsIssueSeverity.LOW
            case None:
                return NbsIssueSeverity.UNKNOWN
            case _:
                logger.warning(f"Unknown nbs_state for event {self.id}: {self.nbs_state}; {dataclasses.asdict(self)}")
                return NbsIssueSeverity.UNKNOWN

# ── Google Calender ──────────────────────────────────────────────────────────────

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

global _event_id_counter
_event_id_counter = 0

def parse_event(item: dict) -> CalendarEvent:
    def parse_dt(val: str) -> datetime.datetime:
        dt = datetime.datetime.fromisoformat(val)
        if isinstance(dt, datetime.datetime):
            return dt.astimezone(zoneinfo.ZoneInfo("Europe/Amsterdam")).replace(tzinfo=None)
        return datetime.datetime(dt.year, dt.month, dt.day)  # midnight, no tz

    start_raw = item["start"].get("dateTime") or item["start"]["date"]
    end_raw   = item["end"].get("dateTime")   or item["end"]["date"]

    global _event_id_counter
    event = CalendarEvent(
        id=_event_id_counter,
        start=parse_dt(start_raw),
        end=parse_dt(end_raw),
        summary=item.get("summary", "NO TITLE"),
        location=item.get("location"),
    )
    _event_id_counter += 1
    return event

def fetch_events(service, calendar_id: str, start: datetime.datetime, end: datetime.datetime) -> list[CalendarEvent]:
    items = (
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
    return [parse_event(event) for event in items]


# ── Event helpers ──────────────────────────────────────────────────────────────

def is_covered(event: CalendarEvent, nbs_intervals: list[tuple[datetime.datetime, datetime.datetime]]) -> bool:
    return any(event.start < nb_end and event.end > nb_start for nb_start, nb_end in nbs_intervals)

def group_events_by_level(events: list[CalendarEvent]) -> dict[NbsIssueSeverity, list[CalendarEvent]]:
    grouped = {level: [] for level in NbsIssueSeverity}
    for event in events:
        if isinstance(event.nbs_state, PossibleNBS):
            match event.nbs_state.score:
                case score if score >= NbsIssueSeverity.HIGH.value:
                    level = NbsIssueSeverity.HIGH
                case score if score >= NbsIssueSeverity.MEDIUM.value:
                    level = NbsIssueSeverity.MEDIUM
                case _:
                    level = NbsIssueSeverity.LOW
            grouped[level].append(event)
    return grouped

# ── LLM ───────────────────────────────────────────────────────────────────────

class Assessment(pydantic.BaseModel):
    id: int
    conflict_score: float
    reason: str


def _build_prompt(events: list[CalendarEvent]) -> str:
    event_lines = []
    for e in events:
        event_lines.append(
            f'{e.id}. [{e.start.isoformat()}-{e.end.isoformat()}]: '
            f'"{e.summary}"'
            f'{f" at {e.location}" if e.location else ""}')
    with open("prompts/base.txt") as f:
        return f.read().replace("<<EVENTS>>", "\n".join(event_lines))


def assess_events(events: list[CalendarEvent]) -> None:
    client = anthropic.Anthropic()
    prompt = _build_prompt(events)
    logger.debug(prompt)
    response = client.messages.parse(
        model="claude-sonnet-4-6",
        max_tokens=4096 * 2,
        messages=[{"role": "user", "content": prompt}],
        output_format=list[Assessment],
    )
    assessments = response.parsed_output
    assert assessments is not None
    for a in assessments:
        logger.debug(a.model_dump())
    assessment_map = {a.id: a for a in assessments}
    for e in events:
        a = assessment_map.get(e.id)
        e.nbs_state = PossibleNBS(score=a.conflict_score, reason=a.reason) if a else OmittedFromAssessmentOutput()


# ── Console print helpers ─────────────────────────────────────────────────────────────

def display_event(item: CalendarEvent) -> None:
    match item.nbs_state:
        case AlreadyFlaggedWithNBS():
            conflict_badge = "🔵"
            conflict_str = "Already flagged with NBS"
        case PossibleNBS(score=score, reason=reason) if score >= NbsIssueSeverity.HIGH.value:
            conflict_badge = "🔴"
            conflict_str = f"{reason} ({score:.0%})"
        case PossibleNBS(score=score, reason=reason) if score >= NbsIssueSeverity.MEDIUM.value:
            conflict_badge = "🟡"
            conflict_str = f"{reason} ({score:.0%})"
        case PossibleNBS(score=score, reason=reason) if score < NbsIssueSeverity.MEDIUM.value:
            conflict_badge = "🟢"
            conflict_str = f"{reason} ({score:.0%})"
        case OmittedFromAssessmentOutput():
            conflict_badge = "🟢"
            conflict_str = "Low probability conflict"
        case _:
            logger.warning(f"Unknown conflict type for event {item.id}: {item.nbs_state}; {dataclasses.asdict(item)}")
            conflict_badge = "⚪"
            conflict_str = "Unknown conflict status"
    location_str = f"📍 {item.location}" if item.location else ""
    same_day = item.start.date() == item.end.date()
    if same_day:
        time_str = f"{item.start:%a %d %b %Y} {item.start:%H:%M}–{item.end:%H:%M}"
    else:
        time_str = f"{item.start:%a %d %b %Y} {item.start:%H:%M} – {item.end:%a %d %b %Y} {item.end:%H:%M}"
    print(f"{conflict_badge} {time_str} {item.summary} {location_str} {conflict_str}")


# ── Gmail ──────────────────────────────────────────────────────────────

# def send_summary(
#     gmail,
#     to: str,
#     n_total: int,
#     n_covered: int,
#     potential_conflicts: list[dict],
#     unlikely_conflicts: list[dict],
# ) -> None:
#     today = datetime.date.today().strftime("%d %b %Y")
#     subject = f"Gig conflict audit — {today}"
 
#     body_parts = [
#         f"Gig conflict audit — {today}",
#         f"",
#         f"Personal events: {n_total}",
#         f"Covered by NBS: {n_covered}",
#         f"Assessed: {len(potential_conflicts) + len(unlikely_conflicts)}",
#         f"🔴  Potential conflicts: {len(potential_conflicts)}",
#         f"🟡  Unlikely conflicts: {len(unlikely_conflicts)}",
#     ]
 
#     if potential_conflicts:
#         body_parts += ["\n<b>🔴 POTENTIAL GIG CONFLICTS</b>", _format_event_block(potential_conflicts)]
#     else:
#         body_parts.append("\n✅ No potential conflicts — all uncovered events look fine.")
 
#     if unlikely_conflicts:
#         body_parts += [f"\n<b>🟡 Unlikely conflicts ({len(unlikely_conflicts)} events)</b>", _format_event_block(unlikely_conflicts)]
 
#     msg = EmailMessage()
#     msg["To"] = to
#     msg["From"] = to
#     msg["Subject"] = subject
#     msg.set_content("\n".join(body_parts))
 
#     raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
#     gmail.users().messages().send(userId="me", body={"raw": raw}).execute()
#     print(f"📧 Summary sent to {to}")

# def _format_event_block(events: list[dict]) -> str:
#     lines = []
#     current_month = None
#     for e in sorted(events, key=lambda x: x["date"]):
#         month = e["date"].strftime("%B %Y")
#         if month != current_month:
#             lines.append(f"\n── {month} ──")
#             current_month = month
#         loc = f"  📍 {e['location']}" if e["location"] else ""
#         lines.append(f"  {e['date'].strftime('%a %d')}  {e['time']}–{e['end_time']}  {e['title']} ({e['conflict_score'] * 100:.0f}%) {loc}")
#         lines.append(f"         → {e['reason']}")
#     return "\n".join(lines)

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
    nbs_intervals = [(e.start, e.end) for e in cassette_events if e.summary.lower().strip() == NBS_MARKER.lower().strip()]
    uncovered = [e for e in personal_events if not is_covered(e, nbs_intervals)]
    covered = [e for e in personal_events if e not in uncovered]
    for e in covered:
        e.nbs_state = AlreadyFlaggedWithNBS()

    if not uncovered:
        print("✅ All personal events already covered by a JH NBS entry.")
        return

    print(f"🤖 Assessing {len(uncovered)} uncovered event(s)...\n")
    assess_events(uncovered)

    for item in personal_events:
        display_event(item)

    # print(f"{'═' * 52}")
    # print(f"  Gig conflict audit")
    # print(f"  Personal events         : {len(personal_events)}")
    # print(f"  Covered by NBS          : {covered_count}")
    # print(f"  Assessed by Claude      : {len(uncovered)}")
    # # print(f"  🔴  Above 50%            : {len(potential_conflicts)}")
    # # print(f"  🟡  Between 20% and 50%  : {len(unlikely_conflicts)}")
    # print(f"{'═' * 52}\n")

    # if potential_conflicts:
    #     print("🔴  POTENTIAL GIG CONFLICTS\n")
    #     _print_events(potential_conflicts)
    # else:
    #     print("✅ No potential conflicts — all uncovered events look fine.\n")

    # if SUMMARY_RECIPIENT:
    #     send_summary(gmail, SUMMARY_RECIPIENT, len(personal_events), covered_count, potential_conflicts, unlikely_conflicts)


if __name__ == "__main__":
    main()