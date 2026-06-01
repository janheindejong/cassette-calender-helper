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
SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/gmail.send",
]


# ── Logging ────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")

# ── Models ────────────────────────────────────────────────────────────


class NbsClassification(Enum):
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    BLOCKED = 4
    UNKNOWN = 5


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

    def is_covered_by_nbs(
        self, nbs_intervals: list[tuple[datetime.datetime, datetime.datetime]]
    ) -> bool:
        return any(
            self.start < nb_end and self.end > nb_start
            for nb_start, nb_end in nbs_intervals
        )

    def with_flag_updated(
        self, nbs_intervals: list[tuple[datetime.datetime, datetime.datetime]]
    ) -> "CalendarEvent":
        return dataclasses.replace(
            self, flagged_nbs=self.is_covered_by_nbs(nbs_intervals)
        )

    def with_nbs_assessment_updated(
        self, assessment: NbsAssessment | BelowThreshold | None
    ) -> "CalendarEvent":
        return dataclasses.replace(self, nbs_assessment=assessment)

    @property
    def classification(self) -> NbsClassification:
        return self._classify_event()[0]

    @property
    def badge(self) -> str:
        return self._classify_event()[1]

    @property
    def classification_string(self) -> str:
        return self._classify_event()[2]

    def _classify_event(self) -> tuple[NbsClassification, str, str]:
        if self.flagged_nbs:
            return NbsClassification.BLOCKED, "🔵", "Covered by NBS"
        match self.nbs_assessment:
            case NbsAssessment(score=score, reason=reason) if (
                score >= HIGH_NBS_THRESHOLD
            ):
                return NbsClassification.HIGH, "🔴", f"{reason} ({score:.0%})"
            case NbsAssessment(score=score, reason=reason) if (
                score >= MEDIUM_NBS_THRESHOLD
            ):
                return NbsClassification.MEDIUM, "🟡", f"{reason} ({score:.0%})"
            case NbsAssessment(score=score, reason=reason):
                return NbsClassification.LOW, "🟢", f"{reason} ({score:.0%})"
            case BelowThreshold():
                return NbsClassification.LOW, "🟢", f"(<{MEDIUM_NBS_THRESHOLD:.0%})"
            case _:
                return NbsClassification.UNKNOWN, "⚪", "Unknown conflict status"


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


def get_calendar_id(service, name: str) -> str:
    cals = service.calendarList().list().execute().get("items", [])
    calendars = {cal.get("summary", "").strip().lower(): cal["id"] for cal in cals}
    res = calendars.get(name.strip().lower())
    if not res:
        cal_names = [cal.get("summary", "") for cal in cals]
        raise ValueError(f"Calendar '{name}' not found; available: {cal_names}")
    return res


global _event_id_counter
_event_id_counter = 0


def parse_event(item: dict) -> CalendarEvent:
    def parse_dt(val: str) -> datetime.datetime:
        dt = datetime.datetime.fromisoformat(val)
        if isinstance(dt, datetime.datetime):
            return dt.astimezone(zoneinfo.ZoneInfo("Europe/Amsterdam")).replace(
                tzinfo=None
            )
        return datetime.datetime(dt.year, dt.month, dt.day)  # midnight, no tz

    start_raw = item["start"].get("dateTime") or item["start"]["date"]
    end_raw = item["end"].get("dateTime") or item["end"]["date"]

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
    return [parse_event(event) for event in items]


# ── Assessment ───────────────────────────────────────────────────────────────────────


class Assessment(pydantic.BaseModel):
    id: int
    conflict_score: float
    reason: str


def _build_prompt(events: list[CalendarEvent]) -> str:
    event_lines = []
    for e in events:
        event_lines.append(
            f"{e.id}. [{e.start.isoformat()}-{e.end.isoformat()}]: "
            f'"{e.summary}"'
            f"{f' at {e.location}' if e.location else ''}"
        )
    with open("prompts/base.txt") as f:
        return f.read().replace("<<EVENTS>>", "\n".join(event_lines))


def assess_events_with_llm(events: list[CalendarEvent]) -> list[CalendarEvent]:
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
    assessed_events = []
    for e in events:
        a = assessment_map.get(e.id)
        assessment = (
            NbsAssessment(score=a.conflict_score, reason=a.reason)
            if a
            else BelowThreshold()
        )
        assessed_events.append(e.with_nbs_assessment_updated(assessment))
    return assessed_events


# ── Flagging ─────────────────────────────────────────────────────────────
def check_if_flagged(
    personal_events: list[CalendarEvent],
    nbs_intervals: list[tuple[datetime.datetime, datetime.datetime]],
) -> list[CalendarEvent]:
    personal_events = [e.with_flag_updated(nbs_intervals) for e in personal_events]
    return personal_events


def extract_nbs_intervals(cassette_events):
    nbs_intervals = [
        (e.start, e.end)
        for e in cassette_events
        if e.summary.lower().strip() == NBS_MARKER.lower().strip()
    ]
    return nbs_intervals


# ── Presentation ─────────────────────────────────────────────────────────────
def summary_strings(events: list[CalendarEvent]) -> list[str]:
    today = datetime.date.today().strftime("%d %b %Y")
    body_parts = [
        f"Gig conflict audit — {today}",
        "",
        f"Events: {len(events)}",
        f"🔴  High probability conflicts: {len([e for e in events if e.classification is NbsClassification.HIGH])}",
        f"🟡  Medium probability conflicts: {len([e for e in events if e.classification is NbsClassification.MEDIUM])}",
        f"🟢  Low probability conflicts: {len([e for e in events if e.classification is NbsClassification.LOW])}",
        f"🔵  Flagged with NBS: {len([e for e in events if e.classification is NbsClassification.BLOCKED])}",
        f"⚪  Unknown: {len([e for e in events if e.classification is NbsClassification.UNKNOWN])}",
    ]
    return body_parts


def group_and_sort_events(events: list[CalendarEvent]) -> dict[NbsClassification, list[CalendarEvent]]:
    grouped: dict[NbsClassification, list[CalendarEvent]] = {}
    for event in sorted(events, key=lambda c: (c.classification.value, c.start)):
        key = event.classification
        grouped.setdefault(key, []).append(event)
    return grouped
 

ICONS = {
    NbsClassification.HIGH: "🔴",
    NbsClassification.MEDIUM: "🟡",
    NbsClassification.LOW: "🟢",
    NbsClassification.BLOCKED: "🔵",
    NbsClassification.UNKNOWN: "⚪",
}


def event_string(event: CalendarEvent) -> str:
    location_str = f" - 📍 {event.location}" if event.location else ""
    same_day = event.start.date() == event.end.date()
    if same_day:
        time_str = f"{event.start:%a %d %b %Y} {event.start:%H:%M}–{event.end:%H:%M}"
    else:
        time_str = f"{event.start:%a %d %b %Y} {event.start:%H:%M} – {event.end:%a %d %b %Y} {event.end:%H:%M}"
    return f"{ICONS[event.classification]} {time_str} - {event.summary.strip()}{location_str} - {event.classification_string}"


def event_strings(events: list[CalendarEvent]) -> list[str]:
    output = []
    for group, events in group_and_sort_events(events).items():
        output.append("─" * 20 + f" {group.name} ({len(events)} events) " + "─" * 20)
        for event in events:
            output.append(f"{event_string(event)}")
        output.append("")
    return output


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

    # Fetch events
    events = fetch_events(service, "primary", now, horizon)
    cassette_events = fetch_events(service, cassette_id, now, horizon)

    # Check against NBS intervals
    nbs_intervals = extract_nbs_intervals(cassette_events)
    events = check_if_flagged(events, nbs_intervals)

    # Assess uncovered events with LLM
    events_not_flagged = [e for e in events if not e.flagged_nbs]
    events_flagged = [e for e in events if e.flagged_nbs]
    events = assess_events_with_llm(events_not_flagged) + events_flagged

    # Create output lines
    lines = summary_strings(events) + [""] + event_strings(events)

    # Send e-mail
    send_summary(gmail, SUMMARY_RECIPIENT, lines, lines[0])

    # Print to console
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
