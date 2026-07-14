import json
import unittest

from app.db import Database
from app.memory import MemoryStore


class EventKindNormalizationTests(unittest.TestCase):
    """A resolved event_datetime must promote a mislabeled dated event to
    'event' (so it renders upcoming/past and ages out) - WITHOUT flipping
    yearly-recurring (birthday/anniversary) or genuine recurring facts.
    Regression guard for the bug where extractor-resolved dates were nulled
    because the fact text dropped the time word."""

    def _kind(self, fact, kind, event_datetime=None, recurrence_rule=None):
        k, edt, _rr, _vf, _vu = Database._infer_temporal_defaults(
            fact, "event", kind, event_datetime, recurrence_rule, None, None)
        return k, edt

    def test_permanent_with_datetime_promoted_to_event(self):
        k, edt = self._kind("User has a dentist appointment", "permanent",
                             "2026-07-09T17:00:00+05:30")
        self.assertEqual(k, "event")
        self.assertEqual(edt, "2026-07-09T17:00:00+05:30")  # date preserved, not nulled

    def test_recurring_without_rule_and_datetime_promoted_to_event(self):
        k, edt = self._kind("User has an exam", "recurring",
                            "2026-07-10T12:00:00+05:30", None)
        self.assertEqual(k, "event")
        self.assertEqual(edt, "2026-07-10T12:00:00+05:30")

    def test_birthday_with_datetime_stays_permanent(self):
        k, edt = self._kind("User's birthday is March 3rd", "permanent",
                            "2026-03-03T00:00:00+05:30")
        self.assertEqual(k, "permanent")
        self.assertIsNone(edt)

    def test_wedding_anniversary_stays_permanent(self):
        # "wedding" is a one-off noun but "anniversary" makes it yearly.
        k, _ = self._kind("User's wedding anniversary is June 1st", "permanent",
                          "2026-06-01T00:00:00+05:30")
        self.assertEqual(k, "permanent")

    def test_genuine_recurring_with_rule_stays_recurring(self):
        k, _ = self._kind("User goes to the gym every monday", "recurring",
                          "2026-07-13T08:00:00+05:30", "weekly")
        self.assertEqual(k, "recurring")

    def test_plain_permanent_unchanged(self):
        k, edt = self._kind("User dislikes coffee", "permanent")
        self.assertEqual(k, "permanent")
        self.assertIsNone(edt)

    def test_existing_text_anchored_path_still_works(self):
        k, edt = self._kind("User has an exam tomorrow at 12", "permanent")
        self.assertEqual(k, "event")
        self.assertIsNotNone(edt)


class TemporalMemoryParsingTests(unittest.TestCase):
    def test_defaults_missing_temporal_fields_to_permanent(self):
        raw = json.dumps({
            "memories": [
                {"fact": "User's birthday is March 3rd", "category": "personal_info"}
            ]
        })
        parsed = MemoryStore._parse_facts(raw)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["kind"], "permanent")
        self.assertIsNone(parsed[0]["event_datetime"])
        self.assertIsNone(parsed[0]["recurrence_rule"])

    def test_preserves_explicit_temporal_metadata(self):
        raw = json.dumps({
            "memories": [
                {
                    "fact": "User has an exam at 12",
                    "category": "event",
                    "kind": "event",
                    "event_datetime": "2026-07-08T12:00:00Z",
                    "recurrence_rule": None,
                    "valid_from": "2026-07-08T00:00:00Z",
                    "valid_until": "2026-07-08T23:59:59Z",
                }
            ]
        })
        parsed = MemoryStore._parse_facts(raw)
        self.assertEqual(parsed[0]["kind"], "event")
        self.assertEqual(parsed[0]["event_datetime"], "2026-07-08T12:00:00Z")
        self.assertEqual(parsed[0]["valid_until"], "2026-07-08T23:59:59Z")


if __name__ == "__main__":
    unittest.main()
