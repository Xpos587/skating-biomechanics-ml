"""Subtitle parser for extracting element information from video captions.

This module parses VTT subtitles to extract:
- Element names (e.g., "тройка", "аксель")
- Timestamps when elements start
- Repetition counts
- Instructions and tips
"""

import re
from dataclasses import dataclass
from pathlib import Path

from ..types import ElementPhase


@dataclass
class ElementEvent:
    """A skating element detected in subtitles.

    Attributes:
        name: Element name (e.g., "тройка", "аксель").
        start_time: Start time in seconds.
        end_time: End time in seconds (if known).
        count: Repetition count (e.g., "3x" for triple jump).
        instructions: List of instructions mentioned.
    """

    name: str
    start_time: float
    end_time: float | None = None
    count: int = 1
    instructions: list[str] | None = None


# Russian element names mapping
ELEMENT_NAMES_RU = {
    # Jumps
    "аксель": "axel",
    "сальхов": "salchow",
    "лутц": "loop",
    "флип": "flip",
    "риттбергер": "rittberger",
    "тоулуп": "toe_loop",
    "перекидной": "toe_loop",  # common alias
    "вальцовый": "waltz_jump",
    # Spins
    "волчок": "scratch_spin",
    "заклон": "sit_spin",
    "либела": "upright_spin",
    "биллиман": "biellmann_spin",
    # Steps/Turns
    "тройка": "three_turn",
    "двойка": "double_three_turn",
    "скобка": "bracket_turn",
    "лост": "mohawk",
    "хoki": "choctaw",
    # Other
    "спираль": "spiral",
    "дорожка": "step_sequence",
}

# Instructions/tips keywords
INSTRUCTION_KEYWORDS = [
    "смотри",
    "обрати",
    "помни",
    "важно",
    "плечи",
    "спина",
    "колени",
    "руки",
    "голову",
    "следи",
    "держи",
    "согни",
    "раз",
    "ещё",
]


class SubtitleParser:
    """Parse VTT subtitles to extract element information."""

    def __init__(self) -> None:
        """Initialize subtitle parser."""

    def parse_vtt(self, vtt_path: Path) -> list[ElementEvent]:
        """Parse VTT subtitle file.

        Args:
            vtt_path: Path to .vtt file.

        Returns:
            List of ElementEvent objects extracted from subtitles.
        """
        content = vtt_path.read_text(encoding="utf-8")

        events: list[ElementEvent] = []
        current_start = 0.0
        current_end = 0.0
        current_text = ""

        for line in content.split("\n"):
            # Parse timestamp: 00:00:25.849 --> 00:00:30.580
            timestamp_match = re.match(
                r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})", line
            )

            if timestamp_match:
                # Process previous caption if exists
                if current_text.strip():
                    events.extend(self._parse_caption(current_start, current_end, current_text))

                # Start new caption
                current_start = self._parse_time(timestamp_match.group(1))
                current_end = self._parse_time(timestamp_match.group(2))
                current_text = ""
            # Accumulate text (skip WEBVTT headers)
            elif (
                not line.startswith("WEBVTT")
                and not line.startswith("Kind:")
                and not line.startswith("Language:")
            ):
                # Remove VTT formatting tags
                clean_line = re.sub(r"<\d{2}:\d{2}\.\d{3}>", "", line)  # timestamp tags
                clean_line = re.sub(r"<c>|</c>", "", clean_line)  # style tags
                current_text += clean_line + " "

        # Process last caption
        if current_text.strip():
            events.extend(self._parse_caption(current_start, current_end, current_text))

        # Merge consecutive events for the same element
        return self._merge_consecutive_events(events)

    def _parse_time(self, time_str: str) -> float:
        """Parse VTT timestamp to seconds.

        Args:
            time_str: Time string like "00:00:25.849".

        Returns:
            Time in seconds as float.
        """
        parts = time_str.split(":")
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds

    def _parse_caption(self, start: float, end: float, text: str) -> list[ElementEvent]:
        """Extract element events from a single caption.

        Args:
            start: Caption start time.
            end: Caption end time.
            text: Caption text.

        Returns:
            List of ElementEvent objects found in this caption.
        """
        events: list[ElementEvent] = []
        text_lower = text.lower()

        # Check for element names
        found_elements: list[str] = []

        for ru_name, en_name in ELEMENT_NAMES_RU.items():
            if ru_name in text_lower:
                found_elements.append(en_name)

        if not found_elements:
            # Check for count hints (e.g., "2 к" = 2 times)
            count_match = re.search(r"(\d+)\s*[kкx]", text_lower)
            if count_match:
                count = int(count_match.group(1))
                # No element name found, save generic event
                events.append(
                    ElementEvent(name="unknown", start_time=start, end_time=end, count=count)
                )
            return events

        # Extract count (e.g., "двойной тройка" = 2)
        count = 1
        count_patterns = [
            r"одиночн",  # single
            r"двойн",  # double
            r"тройн",  # triple
            r"(\d+)\s*[kкx]",  # "2к", "3x"
        ]

        for pattern in count_patterns:
            match = re.search(pattern, text_lower)
            if match:
                if "двойн" in text_lower:
                    count = 2
                elif "тройн" in text_lower:
                    count = 3
                elif match.group(1):
                    count = int(match.group(1))
                break

        # Extract instructions
        instructions: list[str] = []
        for keyword in INSTRUCTION_KEYWORDS:
            if keyword in text_lower:
                instructions.append(keyword)

        # Create event for each found element
        for element_name in found_elements:
            events.append(
                ElementEvent(
                    name=element_name,
                    start_time=start,
                    end_time=end,
                    count=count,
                    instructions=instructions if instructions else None,
                )
            )

        return events

    def _merge_consecutive_events(self, events: list[ElementEvent]) -> list[ElementEvent]:
        """Merge consecutive events for the same element.

        Args:
            events: List of all detected events.

        Returns:
            Merged list of events.
        """
        if not events:
            return []

        merged: list[ElementEvent] = []
        current = events[0]

        for event in events[1:]:
            # Check if same element and close in time
            time_gap = event.start_time - (current.end_time or current.start_time)

            if event.name == current.name and time_gap < 5.0:  # Within 5 seconds
                # Merge
                current.count += event.count
                current.end_time = event.end_time
                if event.instructions:
                    if current.instructions is None:
                        current.instructions = []
                    current.instructions.extend(event.instructions)
            else:
                # Different element or time gap
                merged.append(current)
                current = event

        merged.append(current)
        return merged

    def extract_phases_from_subtitles(
        self, vtt_path: Path, fps: float = 25.0
    ) -> dict[str, ElementPhase]:
        """Extract element phases from subtitles.

        Args:
            vtt_path: Path to VTT file.
            fps: Video frame rate for time-to-frame conversion.

        Returns:
            Dict mapping element names to their ElementPhase boundaries.
        """
        events = self.parse_vtt(vtt_path)

        phases: dict[str, ElementPhase] = {}

        for event in events:
            if event.name == "unknown":
                continue

            start_frame = int(event.start_time * fps)
            end_frame = int(event.end_time * fps) if event.end_time else start_frame + 100

            # Create phase (for steps/turns, no takeoff/landing)
            phase = ElementPhase(
                name=event.name,
                start=start_frame,
                takeoff=0,  # Steps have no takeoff
                peak=0,
                landing=0,  # Steps have no landing
                end=end_frame,
            )

            phases[event.name] = phase

        return phases

    def get_element_timeline(self, vtt_path: Path) -> list[dict]:
        """Get timeline of all elements from subtitles.

        Args:
            vtt_path: Path to VTT file.

        Returns:
            List of dicts with element timeline info.
        """
        events = self.parse_vtt(vtt_path)

        timeline = []
        for event in events:
            timeline.append(
                {
                    "name": event.name,
                    "name_ru": self._to_russian(event.name),
                    "start": event.start_time,
                    "end": event.end_time,
                    "count": event.count,
                    "instructions": event.instructions,
                }
            )

        return timeline

    def _to_russian(self, element_name: str) -> str:
        """Convert element name to Russian.

        Args:
            element_name: English element name.

        Returns:
            Russian name or original if not found.
        """
        ru_to_en = {v: k for k, v in ELEMENT_NAMES_RU.items()}
        return ru_to_en.get(element_name, element_name)
