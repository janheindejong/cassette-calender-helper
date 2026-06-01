import base64
import dataclasses
import datetime
from email.message import EmailMessage
from enum import Enum, auto
import itertools
import logging
import os
from typing import Generator
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
MEDIUM_NBS_THRESHOLD = 0.2
HIGH_NBS_THRESHOLD = 0.5

# Fixed configuration 
CASSETTE_CALENDAR_NAME = "Cassette Band Intern"
NBS_MARKER = "JH NBS"
SCOPES = ["https://www.googleapis.com/auth/calendar.readonly", "https://www.googleapis.com/auth/gmail.send"]


# ── Logging ────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")

# ── Models ────────────────────────────────────────────────────────────

class BelowThreshold:
    pass 


@dataclasses.dataclass
class NbsAssessment: 
    score: float 
    reason: str 


@dataclasses.dataclass
class CalendarEvent: 
    id: int 
    start: datetime.datetime
    end: datetime.datetime
    summary: str 
    location: str | None = None
    flagged_nbs: bool | None = None
    nbs_assessment: None | BelowThreshold | NbsAssessment = None

    def is_covered_by_nbs(self, nbs_intervals: list[tuple[datetime.datetime, datetime.datetime]]) -> bool:
        return any(self.start < nb_end and self.end > nb_start for nb_start, nb_end in nbs_intervals)
    
    def with_flag_updated(self, nbs_intervals: list[tuple[datetime.datetime, datetime.datetime]]) -> "CalendarEvent":
        return dataclasses.replace(self, flagged_nbs=self.is_covered_by_nbs(nbs_intervals))

# @property
#     def conflict_severity(self) -> NbsIssueSeverity:
#         match self.nbs_state:
#             case AlreadyFlaggedWithNBS():
#                 return NbsIssueSeverity.ALREADY_FLAGGED
#             case PossibleNBS(score=score) if score >= 0.5:
#                 return NbsIssueSeverity.HIGH
#             case PossibleNBS(score=score) if score >= 0.2:
#                 return NbsIssueSeverity.MEDIUM
#             case PossibleNBS(score=score):
#                 return NbsIssueSeverity.LOW
#             case OmittedFromAssessmentOutput():
#                 return NbsIssueSeverity.LOW
#             case None:
#                 return NbsIssueSeverity.UNKNOWN
#             case _:
#                 logger.warning(f"Unknown nbs_state for event {self.id}: {self.nbs_state}; {dataclasses.asdict(self)}")
#                 return NbsIssueSeverity.UNKNOWN

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



# def group_events_by_level(events: list[CalendarEvent]) -> dict[NbsIssueSeverity, list[CalendarEvent]]:
#     grouped = {level: [] for level in NbsIssueSeverity}
#     for event in events:
#         if isinstance(event.nbs, PossibleNBS):
#             match event.nbs.score:
#                 case score if score >= NbsIssueSeverity.HIGH.value:
#                     level = NbsIssueSeverity.HIGH
#                 case score if score >= NbsIssueSeverity.MEDIUM.value:
#                     level = NbsIssueSeverity.MEDIUM
#                 case _:
#                     level = NbsIssueSeverity.LOW
#             grouped[level].append(event)
#     return grouped

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


def assess_events_with_llm(events: list[CalendarEvent]) -> None:
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
        e.nbs_assessment = NbsAssessment(score=a.conflict_score, reason=a.reason) if a else BelowThreshold()


# ── Classification ─────────────────────────────────────────────────────────────
class NbsGroup(Enum):
    HIGH = auto()         # 🔴
    MEDIUM = auto()       # 🟡
    LOW = auto()          # 🟢
    COVERED = auto()      # 🔵
    UNKNOWN = auto()      # ⚪


@dataclasses.dataclass
class ClassifiedEvent:
    event: CalendarEvent
    group: NbsGroup
    badge: str
    conflict_str: str


def classify_event(item: CalendarEvent) -> ClassifiedEvent:
    if item.flagged_nbs:
        return ClassifiedEvent(item, NbsGroup.COVERED, "🔵", "Covered by NBS")
    match item.nbs_assessment:
        case NbsAssessment(score=score, reason=reason) if score >= HIGH_NBS_THRESHOLD:
            return ClassifiedEvent(item, NbsGroup.HIGH, "🔴", f"{reason} ({score:.0%})")
        case NbsAssessment(score=score, reason=reason) if score >= MEDIUM_NBS_THRESHOLD:
            return ClassifiedEvent(item, NbsGroup.MEDIUM, "🟡", f"{reason} ({score:.0%})")
        case NbsAssessment(score=score, reason=reason):
            return ClassifiedEvent(item, NbsGroup.LOW, "🟢", f"{reason} ({score:.0%})")
        case BelowThreshold():
            return ClassifiedEvent(item, NbsGroup.LOW, "🟢", f"(<{MEDIUM_NBS_THRESHOLD:.0%})")
        case _:
            logger.warning(...)
            return ClassifiedEvent(item, NbsGroup.UNKNOWN, "⚪", "Unknown conflict status")


def classify_events(items: list[CalendarEvent]) -> list[ClassifiedEvent]:
    return [classify_event(e) for e in items]


def sort_events(events: list[ClassifiedEvent]) -> list[ClassifiedEvent]:
    return sorted(
        events,
        key=lambda c: (c.group.value, c.event.start),
    )

# ── Presentation ─────────────────────────────────────────────────────────────
def summary_strings(events: list[ClassifiedEvent]) -> list[str]:
    today = datetime.date.today().strftime("%d %b %Y")
    body_parts = [
        f"Gig conflict audit — {today}",
        f"",
        f"Events: {len(events)}",
        f"🔴  High probability conflicts: {len([e for e in events if e.group is NbsGroup.HIGH])}",
        f"🟡  Medium probability conflicts: {len([e for e in events if e.group is NbsGroup.MEDIUM])}",
        f"🟢  Low probability conflicts: {len([e for e in events if e.group is NbsGroup.LOW])}",
        f"🔵  Covered by NBS: {len([e for e in events if e.group is NbsGroup.COVERED])}",
        f"⚪  Unknown: {len([e for e in events if e.group is NbsGroup.UNKNOWN])}",
    ]
    return body_parts


def event_string(classified: ClassifiedEvent) -> str:
    item = classified.event
    location_str = f" - 📍 {item.location}" if item.location else ""
    same_day = item.start.date() == item.end.date()
    if same_day:
        time_str = f"{item.start:%a %d %b %Y} {item.start:%H:%M}–{item.end:%H:%M}"
    else:
        time_str = f"{item.start:%a %d %b %Y} {item.start:%H:%M} – {item.end:%a %d %b %Y} {item.end:%H:%M}"
    return f"{classified.badge} {time_str} - {item.summary.strip()}{location_str} - {classified.conflict_str}"


def event_strings(events: list[ClassifiedEvent]) -> list[str]:
    return [event_string(e) for e in events]


# ── Gmail ──────────────────────────────────────────────────────────────
def send_summary(
    gmail,
    to: str,
    body_parts: list[str],
    subject: str,
) -> None:
    msg = EmailMessage()
    msg["To"] = to
    msg["From"] = to
    msg["Subject"] = subject
    msg.set_content("\n".join(body_parts))
 
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    gmail.users().messages().send(userId="me", body={"raw": raw}).execute()
    print(f"📧 Summary sent to {to}")


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

    # Fetch events 
    events = fetch_events(service, "primary", now, horizon)

    # Check against NBS intervals
    cassette_events = fetch_events(service, cassette_id, now, horizon)
    nbs_intervals = [(e.start, e.end) for e in cassette_events if e.summary.lower().strip() == NBS_MARKER.lower().strip()]
    events = [e.with_flag_updated(nbs_intervals) for e in events]

    # Assess uncovered events with LLM
    assess_events_with_llm([e for e in events if not e.flagged_nbs])

    # Classify and sort 
    classified_events = classify_events(events)
    sorted_events = sort_events(classified_events)

    # Create output lines 
    lines = summary_strings(sorted_events) + [""] + event_strings(sorted_events)

    # Send e-mail
    send_summary(gmail, SUMMARY_RECIPIENT, lines, lines[0])

    # Print to console
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
