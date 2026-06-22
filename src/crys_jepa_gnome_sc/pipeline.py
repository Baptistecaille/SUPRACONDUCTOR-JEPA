"""Pipeline orchestration."""

from __future__ import annotations

from typing import Iterable, List

from crys_jepa_gnome_sc.models import Candidate, RankedCandidate
from crys_jepa_gnome_sc.ranking import rank_candidates


def screen_candidates(candidates: Iterable[Candidate]) -> List[RankedCandidate]:
    """Screen and rank candidates."""

    return rank_candidates(candidates)
