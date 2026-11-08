"""Concepts in, phrasings out -- the vocabulary the models actually answer to.

Phase 15.8. The MEP photo bench (data/fixtures/photos/mep/bench_sam3.json)
measured something nobody would have guessed from the papers: SAM 3's prompt
vocabulary is VISUAL, not trade jargon. On the same photograph, "electrical
conduit pipe" scored 0 and "metal pipe on wall" scored 0.87-0.90; "junction
box" scored 0 where "electrical box" scored 0.41; "cable tray" fired zero
times across all twelve photographs. The masks that do fire are pixel-accurate
-- the models see fine, they just do not speak electrician.

So a CONCEPT -- the word a site engineer uses, the word the ground truth is
labelled with -- expands here into several plain-language phrasings, every
phrasing is asked, and the answers are folded back under the concept's name.
The report says "cable tray"; the phrasing that actually fired travels along
as attribution, because *which wording worked* is itself a research result.

WHY A SEPARATE MODULE, NOT `FIT_OUT_PROMPTS`

    `openvocab.FIT_OUT_PROMPTS` is a flat starting vocabulary tuned for
    YOLO-World's box matching, and its group names go nowhere. This mapping is
    a different artifact with a different job: its keys are the labels the
    scored bench and the eventual dashboard speak, and its phrasings are
    empirical -- seeded from the bench finding above and tuned against the
    scored ground truth, not guessed.

WHERE IT SITS

    Between the request and the provider on the way in (`expand`), and between
    the provider and fusion on the way out (`relabel_*`). Relabeling happens
    BEFORE `fuse_instances` / `merge_duplicates` on purpose: once two masks
    carry the same concept label, the ordinary same-label merge rules unite
    them, and the label vote becomes a vote among concepts. The providers
    themselves never learn any of this exists.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vision.concept import InstanceMask
    from vision.detect import RawDetection
    from vision.flat import FlatHit


@dataclass(frozen=True, slots=True)
class Concept:
    """One thing worth finding, and the words that actually find it."""

    name: str  # the reported label -- the site's word
    phrasings: tuple[str, ...]  # visual, plain-language, lowercase
    group: str = ""  # documentation only; nothing dispatches on it


# Phrasings are lowercase because every prompted provider lowercases its
# prompts, and the relabel step matches returned labels against them exactly.
# Two or three each, deliberately: SAM 3 pays one full forward pass per
# phrase per view, so a phrasing has to earn its seconds.
CONCEPTS: dict[str, Concept] = {
    concept.name: concept
    for concept in (
        # --- MEP: the wall/ceiling/floor service stages -------------------
        Concept(
            "electrical conduit",
            ("metal pipe on wall", "thin pipes running along ceiling"),
            "mep",
        ),
        Concept(
            "junction box",
            ("electrical box", "metal box on wall with cables"),
            "mep",
        ),
        Concept(
            "cable tray",
            ("metal ladder rack on ceiling", "metal tray holding cables"),
            "mep",
        ),
        Concept(
            "exposed wiring",
            ("exposed electrical wires", "bundle of hanging wires"),
            "mep",
        ),
        Concept(
            "distribution board",
            ("electrical distribution board", "electrical panel with breakers"),
            "mep",
        ),
        Concept(
            "ventilation duct",
            ("large metal duct on ceiling", "air conditioning duct"),
            "mep",
        ),
        Concept(
            "plumbing pipe",
            ("plastic water pipe", "pvc pipe on wall"),
            "mep",
        ),
        # --- site services, by COLOUR (research experiment 05) ------------
        # On the site photos, without a colour word SAM 3 cannot tell the
        # services apart -- "fire sprinkler pipe" landed on black conduit as
        # often as on red pipe, and "exposed electrical wires" split between
        # the two. With one, "red pipe" and "black pipe" barely overlap, and
        # "black electrical conduit" adds to "black pipe". The
        # colours are THIS site's (fire red, electrical black, ducts in silver
        # foil); another site's colours are another set of items.
        Concept("fire pipe", ("red pipe",), "site services"),
        Concept("electrical pipe", ("black pipe", "black electrical conduit"), "site services"),
        Concept("hvac duct", ("silver duct",), "site services"),
        # --- fixtures: presence and placement per room --------------------
        Concept("chair", ("chair",), "fixtures"),
        Concept("table", ("table", "desk"), "fixtures"),
        Concept("television", ("television screen", "tv mounted on wall"), "fixtures"),
        Concept("ceiling fan", ("ceiling fan",), "fixtures"),
        Concept("rug", ("rug on floor",), "fixtures"),
        Concept("curtains", ("window curtains",), "fixtures"),
        # --- finishes ------------------------------------------------------
        Concept(
            "suspended ceiling",
            ("suspended ceiling grid", "ceiling tiles"),
            "finishes",
        ),
    )
}

# What an empty prompt box means for the prompted models: the MEP set, because
# that is the research question being measured right now, and because prompts
# are a per-view forward pass each on SAM 3 -- all fourteen concepts at once
# would be a coffee break, not a bench run.
# Words that make an item PIPE-LIKE: long, and a pipe's width wherever it is
# seen. Such an item is checked for thickness (`fuse_instances`): a mask much
# broader than a pipe -- a band of red paint on the vault, which "red pipe"
# also finds -- is not pipe. Not duct or tray: those are broad by nature.
PIPE_WORDS: tuple[str, ...] = ("pipe", "conduit", "wiring", "wire", "cable")


def is_pipe_like(name: str) -> bool:
    return any(word in name for word in PIPE_WORDS)


DEFAULT_CONCEPT_NAMES: tuple[str, ...] = (
    "electrical conduit",
    "junction box",
    "cable tray",
    "exposed wiring",
    "distribution board",
)


def with_extra(extra: Iterable[Concept]) -> dict[str, Concept]:
    """The built-in vocabulary plus concepts added from the bench.

    A built-in name always wins: the added ones cannot redefine what "cable
    tray" means under an existing scored bench, only name something new --
    "sprinkler pipe", on a site whose ceilings are full of them.
    """
    merged = dict(CONCEPTS)
    for concept in extra:
        merged.setdefault(concept.name, concept)
    return merged


def expand(
    requests: list[str] | None,
    vocabulary: Mapping[str, Concept] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Requests -> (phrases to actually ask, phrase -> concept to fold back by).

    A request matching a concept name (case-insensitive) becomes that
    concept's phrasings; anything else is a raw phrase and passes through
    mapping to itself -- typing "metal pipe on wall" directly behaves exactly
    as it always has. None means the default concept set. Order is preserved,
    duplicates dropped, everything lowercased to match what the providers
    return as labels. `vocabulary` defaults to the built-in concepts; the
    bench passes `with_extra(...)` so concepts added there expand too.
    """
    if requests is None:
        requests = list(DEFAULT_CONCEPT_NAMES)
    known = CONCEPTS if vocabulary is None else vocabulary

    phrases: list[str] = []
    phrase_map: dict[str, str] = {}
    for request in requests:
        token = request.strip().lower()
        if not token:
            continue
        concept = known.get(token)
        expanded = concept.phrasings if concept else (token,)
        target = concept.name if concept else token
        for phrase in expanded:
            if phrase not in phrase_map:
                phrases.append(phrase)
                phrase_map[phrase] = target
    return phrases, phrase_map


# ---------------------------------------------------------------------------
# Folding answers back under their concept
# ---------------------------------------------------------------------------
#
# Each relabeler is `dataclasses.replace` and nothing else: label becomes the
# concept, the original label survives as `phrase` -- but only when the two
# differ, so an unexpanded run keeps phrase == "" and old stored results stay
# byte-identical in meaning.


def _fold(label: str, phrase_map: dict[str, str]) -> tuple[str, str]:
    concept = phrase_map.get(label, label)
    return (concept, label) if concept != label else (label, "")


def relabel_instances(
    instances_by_view: list[list[InstanceMask]], phrase_map: dict[str, str]
) -> list[list[InstanceMask]]:
    return [
        [
            replace(one, label=concept, phrase=phrase)
            for one in view
            for concept, phrase in (_fold(one.label, phrase_map),)
        ]
        for view in instances_by_view
    ]


def relabel_raw(
    raw_by_view: list[list[RawDetection]], phrase_map: dict[str, str]
) -> list[list[RawDetection]]:
    return [
        [
            replace(one, label=concept, phrase=phrase)
            for one in view
            for concept, phrase in (_fold(one.label, phrase_map),)
        ]
        for view in raw_by_view
    ]


def relabel_hits(hits: list[FlatHit], phrase_map: dict[str, str]) -> list[FlatHit]:
    return [
        replace(one, label=concept, phrase=phrase)
        for one in hits
        for concept, phrase in (_fold(one.label, phrase_map),)
    ]
