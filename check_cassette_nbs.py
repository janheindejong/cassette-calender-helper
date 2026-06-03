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


# ── Logging ────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")


# ── Models ────────────────────────────────────────────────────────────
class NbsClassification(Enum):
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    FLAGGED = 4
    UNKNOWN = 5


class BelowThreshold: ...


@dataclasses.dataclass(frozen=True)
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

    @property
    def classification(self) -> NbsClassification:
        if self.flagged_nbs:
            return NbsClassification.FLAGGED
        match self.nbs_assessment:
            case NbsAssessment(score=score) if score >= HIGH_NBS_THRESHOLD:
                return NbsClassification.HIGH
            case NbsAssessment(score=score) if score >= MEDIUM_NBS_THRESHOLD:
                return NbsClassification.MEDIUM
            case NbsAssessment():
                return NbsClassification.LOW
            case BelowThreshold():
                return NbsClassification.LOW
            case _:
                return NbsClassification.UNKNOWN


# ── Google Credentials ──────────────────────────────────────────────────────────────
SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/gmail.send",
]


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


# ── Google Calendar ──────────────────────────────────────────────────────────────
def get_calendar_id(service, name: str) -> str:
    cals = service.calendarList().list().execute().get("items", [])
    calendars = {cal.get("summary", "").strip().lower(): cal["id"] for cal in cals}
    res = calendars.get(name.strip().lower())
    if not res:
        cal_names = [cal.get("summary", "") for cal in cals]
        raise ValueError(f"Calendar '{name}' not found; available: {cal_names}")
    return res


def parse_event(item: dict, event_id: int) -> CalendarEvent:
    def parse_dt(val: str) -> datetime.datetime:
        dt = datetime.datetime.fromisoformat(val)
        if isinstance(dt, datetime.datetime):
            return dt.astimezone(zoneinfo.ZoneInfo("Europe/Amsterdam")).replace(tzinfo=None)
        return datetime.datetime(dt.year, dt.month, dt.day)  # midnight, no tz

    start_raw = item["start"].get("dateTime") or item["start"]["date"]
    end_raw = item["end"].get("dateTime") or item["end"]["date"]

    event = CalendarEvent(
        id=event_id,
        start=parse_dt(start_raw),
        end=parse_dt(end_raw),
        summary=item.get("summary", "NO TITLE"),
        location=item.get("location"),
    )
    return event


def fetch_events(
    service, calendar_id: str, start: datetime.datetime, end: datetime.datetime
) -> list[CalendarEvent]:
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
    return [parse_event(event, i) for i, event in enumerate(items)]


# ── Flagging ─────────────────────────────────────────────────────────────
def update_nbs_flag(events: list[CalendarEvent], cassette_events: list[CalendarEvent]) -> None:
    nbs_intervals = extract_nbs_intervals(cassette_events)
    for event in events:
        if is_covered_by_nbs(event, nbs_intervals):
            event.flagged_nbs = True


def is_covered_by_nbs(
    event: CalendarEvent,
    nbs_intervals: list[tuple[datetime.datetime, datetime.datetime]],
) -> bool:
    return any(event.start < t2 and event.end > t1 for t1, t2 in nbs_intervals)


def extract_nbs_intervals(cassette_events):
    nbs_intervals = [
        (e.start, e.end)
        for e in cassette_events
        if e.summary.lower().strip() == NBS_MARKER.lower().strip()
    ]
    return nbs_intervals


# ── Assessment ───────────────────────────────────────────────────────────────────────
class LlmAssessment(pydantic.BaseModel):
    id: int
    conflict_score: float
    reason: str


def assess_events_with_llm(events: list[CalendarEvent]) -> None:
    client = anthropic.Anthropic()
    prompt = _build_prompt(events)
    logger.debug(prompt)
    response = client.messages.parse(
        model=CLAUDE_MODEL,
        max_tokens=4096 * 2,
        messages=[{"role": "user", "content": prompt}],
        output_format=list[LlmAssessment],
    )
    assessments = response.parsed_output
    assert assessments is not None
    for a in assessments:
        logger.debug(a.model_dump())
    assessment_map = {a.id: a for a in assessments}
    for e in events:
        a = assessment_map.get(e.id)
        assessment = (
            NbsAssessment(score=a.conflict_score, reason=a.reason) if a else BelowThreshold()
        )
        e.nbs_assessment = assessment


def _build_prompt(events: list[CalendarEvent]) -> str:
    event_lines = []
    for e in events:
        event_lines.append(
            f"{e.id}. [{e.start.isoformat()}-{e.end.isoformat()}]: "
            f'"{e.summary}"'
            f"{f' at {e.location}' if e.location else ''}"
        )
    with open("prompts/base.txt") as f:
        return f.read().replace("<<EVENTS>>", "\n".join(event_lines)).replace("<<MEDIUM_NBS_THRESHOLD>>", f"{MEDIUM_NBS_THRESHOLD:.0%}")


# ── Presentation ─────────────────────────────────────────────────────────────
ICONS = {
    NbsClassification.HIGH: "🔴",
    NbsClassification.MEDIUM: "🟡",
    NbsClassification.LOW: "🟢",
    NbsClassification.FLAGGED: "🔵",
    NbsClassification.UNKNOWN: "⚪",
}


def generate_report(events: list[CalendarEvent]) -> list[str]:
    return summary_strings(events) + [""] + event_strings(events)


def summary_strings(events: list[CalendarEvent]) -> list[str]:
    today = datetime.date.today().strftime("%d %b %Y")
    body_parts = [
        f"Gig conflict audit — {today}",
        "",
        f"Events: {len(events)}",
        f"{ICONS[NbsClassification.HIGH]}  High probability conflicts: {len([e for e in events if e.classification is NbsClassification.HIGH])}",
        f"{ICONS[NbsClassification.MEDIUM]}  Medium probability conflicts: {len([e for e in events if e.classification is NbsClassification.MEDIUM])}",
        f"{ICONS[NbsClassification.LOW]}  Low probability conflicts: {len([e for e in events if e.classification is NbsClassification.LOW])}",
        f"{ICONS[NbsClassification.FLAGGED]}  Flagged with NBS: {len([e for e in events if e.classification is NbsClassification.FLAGGED])}",
        f"{ICONS[NbsClassification.UNKNOWN]}  Unknown: {len([e for e in events if e.classification is NbsClassification.UNKNOWN])}",
    ]
    return body_parts


def event_line(event: CalendarEvent) -> str:
    location_str = f" - 📍 {event.location}" if event.location else ""
    if event.start.date() == event.end.date():
        time_str = f"{event.start:%a %d %b %Y} {event.start:%H:%M}–{event.end:%H:%M}"
    else:
        time_str = f"{event.start:%a %d %b %Y} {event.start:%H:%M} – {event.end:%a %d %b %Y} {event.end:%H:%M}"
    match event.classification:
        case NbsClassification.HIGH | NbsClassification.MEDIUM | NbsClassification.LOW:
            match event.nbs_assessment:
                case NbsAssessment(reason=reason, score=score):
                    classification_str = f"{reason} ({score:.0%})"
                case BelowThreshold():
                    classification_str = f"(<{MEDIUM_NBS_THRESHOLD:.0%})"
                case None:
                    raise ValueError("Expected NbsAssessment or BelowThreshold for assessed event")
        case NbsClassification.FLAGGED:
            classification_str = "Flagged with NBS"
        case NbsClassification.UNKNOWN:
            classification_str = "No assessment available"
    return f"{ICONS[event.classification]} {time_str} - {event.summary.strip()}{location_str} - {classification_str}"


def event_strings(events: list[CalendarEvent]) -> list[str]:
    output = []
    for group, events in _group_and_sort_events(events).items():
        output.append("─" * 20 + f" {group.name} ({len(events)} events) " + "─" * 20)
        for event in events:
            output.append(f"{event_line(event)}")
        output.append("")
    return output


def _group_and_sort_events(
    events: list[CalendarEvent],
) -> dict[NbsClassification, list[CalendarEvent]]:
    grouped: dict[NbsClassification, list[CalendarEvent]] = {}
    for event in sorted(events, key=lambda c: (c.classification.value, c.start)):
        key = event.classification
        grouped.setdefault(key, []).append(event)
    return grouped


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
    dotenv.load_dotenv()

    creds = get_credentials()
    service = build("calendar", "v3", credentials=creds)
    gmail = build("gmail", "v1", credentials=creds)

    # Get calendar ID
    cassette_id = get_calendar_id(service, CASSETTE_CALENDAR_NAME)

    # Fetch events
    now = datetime.datetime.now()
    horizon = now + datetime.timedelta(days=365)
    events = fetch_events(service, "primary", now, horizon)
    cassette_events = fetch_events(service, cassette_id, now, horizon)

    # Check against NBS intervals
    update_nbs_flag(events, cassette_events)

    # Assess uncovered events with LLM
    assess_events_with_llm([e for e in events if not e.flagged_nbs])

    # Create output lines
    report = generate_report(events)

    # Send e-mail
    send_summary(gmail, SUMMARY_RECIPIENT, report, report[0])

    # Print to console
    for line in report:
        print(line)


if __name__ == "__main__":
    main()
