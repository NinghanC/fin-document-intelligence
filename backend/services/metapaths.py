"""
Finance-domain metapath definitions and rule-based routing.

Metapaths encode analyst-style graph patterns such as fund exposure,
sector concentration, and regulatory scope. They are explicit domain
knowledge, not learned model parameters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Direction = Literal["out", "in"]

ENTITY_TYPES = {
    "Company",
    "Concept",
    "Event",
    "Fund",
    "Location",
    "Organization",
    "Person",
    "Product",
    "Region",
    "Regulation",
    "RiskFactor",
    "Sector",
    "Supplier",
    "Technology",
    "Time",
}

RELATION_TYPES = {
    "belongs_to",
    "depends_on",
    "developed_by",
    "holds",
    "located_in",
    "owns",
    "part_of",
    "regulated_by",
    "related_to",
    "subject_to",
    "uses",
    "works_at",
}


@dataclass(frozen=True)
class MetapathStep:
    from_type: str
    relation: str
    direction: Direction
    to_type: str


@dataclass(frozen=True)
class MetapathSpec:
    name: str
    description: str
    steps: tuple[MetapathStep, ...]
    keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class MetapathResult:
    metapath_name: str
    start_entity: str
    end_entity: str
    path: tuple[tuple[str, str, str], ...]
    evidence: str
    score: float

    @property
    def intermediate_entities(self) -> tuple[str, ...]:
        if not self.path:
            return ()
        return tuple(edge[2] for edge in self.path[:-1])


@dataclass(frozen=True)
class MetapathSelection:
    spec: MetapathSpec
    score: int
    matched_keywords: tuple[str, ...]
    reason: str
    fallback: bool = False

    def as_trace(self) -> dict[str, object]:
        return {
            "metapath": self.spec.name,
            "score": self.score,
            "matched_keywords": list(self.matched_keywords),
            "reason": self.reason,
            "fallback": self.fallback,
        }


FINANCIAL_METAPATHS: dict[str, MetapathSpec] = {
    "sector_exposure": MetapathSpec(
        name="sector_exposure",
        description="Find sectors represented by companies held by a fund.",
        keywords=("sector", "industry", "concentration", "exposure"),
        steps=(
            MetapathStep("Fund", "holds", "out", "Company"),
            MetapathStep("Company", "belongs_to", "out", "Sector"),
        ),
    ),
    "geographic_risk": MetapathSpec(
        name="geographic_risk",
        description="Find geographic regions represented by fund holdings.",
        keywords=("geographic", "geography", "region", "country", "location"),
        steps=(
            MetapathStep("Fund", "holds", "out", "Company"),
            MetapathStep("Company", "located_in", "out", "Region"),
        ),
    ),
    "supply_chain_risk": MetapathSpec(
        name="supply_chain_risk",
        description="Find supplier dependencies created by fund holdings.",
        keywords=("supplier", "supply", "vendor", "dependency", "dependencies"),
        steps=(
            MetapathStep("Fund", "holds", "out", "Company"),
            MetapathStep("Company", "depends_on", "out", "Supplier"),
        ),
    ),
    "technology_risk": MetapathSpec(
        name="technology_risk",
        description="Find technology dependencies created by fund holdings.",
        keywords=("technology", "platform", "cloud", "software", "infrastructure"),
        steps=(
            MetapathStep("Fund", "holds", "out", "Company"),
            MetapathStep("Company", "uses", "out", "Technology"),
        ),
    ),
    "shared_sector": MetapathSpec(
        name="shared_sector",
        description="Find companies connected through the same sector.",
        keywords=("shared sector", "share", "peer", "peers", "same sector", "similar companies"),
        steps=(
            MetapathStep("Company", "belongs_to", "out", "Sector"),
            MetapathStep("Sector", "belongs_to", "in", "Company"),
        ),
    ),
    "management_overlap": MetapathSpec(
        name="management_overlap",
        description="Find companies connected through shared people.",
        keywords=("management", "executive", "board", "director", "overlap"),
        steps=(
            MetapathStep("Company", "works_at", "in", "Person"),
            MetapathStep("Person", "works_at", "out", "Company"),
        ),
    ),
    "compliance_chain": MetapathSpec(
        name="compliance_chain",
        description="Find regulations that apply through fund holdings.",
        keywords=("regulation", "regulatory", "compliance", "basel", "sec", "subject to"),
        steps=(
            MetapathStep("Fund", "holds", "out", "Company"),
            MetapathStep("Company", "subject_to", "out", "Regulation"),
        ),
    ),
    "subsidiary_chain": MetapathSpec(
        name="subsidiary_chain",
        description="Find multi-level company ownership chains.",
        keywords=("subsidiary", "ownership", "owns", "owned", "parent"),
        steps=(
            MetapathStep("Company", "owns", "out", "Company"),
            MetapathStep("Company", "owns", "out", "Company"),
        ),
    ),
}


class MetapathValidationError(ValueError):
    """Raised when a metapath references unsupported graph vocabulary."""


def validate_metapath(spec: MetapathSpec) -> None:
    if not spec.name or not spec.steps:
        raise MetapathValidationError("Metapath requires a name and at least one step")
    for step in spec.steps:
        if step.from_type not in ENTITY_TYPES:
            raise MetapathValidationError(f"Unsupported source entity type: {step.from_type}")
        if step.to_type not in ENTITY_TYPES:
            raise MetapathValidationError(f"Unsupported target entity type: {step.to_type}")
        if step.relation not in RELATION_TYPES:
            raise MetapathValidationError(f"Unsupported relation type: {step.relation}")
        if step.direction not in ("out", "in"):
            raise MetapathValidationError(f"Unsupported direction: {step.direction}")


def validate_all_metapaths(specs: dict[str, MetapathSpec] | None = None) -> None:
    for spec in (specs or FINANCIAL_METAPATHS).values():
        validate_metapath(spec)


class CandidateMetapathGenerator:
    """Produce candidate metapaths before ranking.

    The current generator intentionally returns every validated finance metapath.
    This keeps recall high and makes the ranking stage the explicit replacement
    point for a learned router or HAN-style attention layer.
    """

    def __init__(self, specs: dict[str, MetapathSpec] | None = None) -> None:
        self.specs = specs or FINANCIAL_METAPATHS
        validate_all_metapaths(self.specs)

    def generate(self, query: str, entities: list[str] | None = None) -> list[MetapathSpec]:
        del query, entities
        return list(self.specs.values())


class RuleMetapathRanker:
    """Rank candidate metapaths with transparent keyword rules."""

    def __init__(self, fallback_names: tuple[str, ...] | None = None) -> None:
        self.fallback_names = fallback_names or ("sector_exposure", "geographic_risk", "compliance_chain")

    def rank(
        self,
        query: str,
        candidates: list[MetapathSpec],
        entities: list[str] | None = None,
        limit: int = 3,
    ) -> list[MetapathSelection]:
        lowered = query.lower()
        candidate_by_name = {candidate.name: candidate for candidate in candidates}
        scored: list[tuple[int, str, MetapathSelection]] = []
        for spec in candidates:
            matched_keywords = tuple(keyword for keyword in spec.keywords if self._keyword_matches(lowered, keyword))
            score = len(matched_keywords)
            if score:
                scored.append((
                    score,
                    spec.name,
                    MetapathSelection(
                        spec=spec,
                        score=score,
                        matched_keywords=matched_keywords,
                        reason=f"matched query terms: {', '.join(matched_keywords)}",
                    ),
                ))

        if not scored and entities:
            for name in self.fallback_names:
                fallback_spec = candidate_by_name.get(name)
                if fallback_spec is None:
                    continue
                scored.append((
                    1,
                    name,
                    MetapathSelection(
                        spec=fallback_spec,
                        score=1,
                        matched_keywords=(),
                        reason="fallback for linked financial entities without a more specific query term",
                        fallback=True,
                    ),
                ))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [selection for _, _, selection in scored[:limit]]

    @staticmethod
    def _keyword_matches(lowered_query: str, keyword: str) -> bool:
        escaped = re.escape(keyword.lower())
        if "\\ " in escaped:
            return re.search(rf"(?<!\w){escaped}(?!\w)", lowered_query) is not None
        return re.search(rf"(?<!\w){escaped}s?(?!\w)", lowered_query) is not None


class MetapathRouter:
    """Backward-compatible facade over candidate generation and ranking."""

    def __init__(
        self,
        specs: dict[str, MetapathSpec] | None = None,
        generator: CandidateMetapathGenerator | None = None,
        ranker: RuleMetapathRanker | None = None,
    ) -> None:
        self.generator = generator or CandidateMetapathGenerator(specs)
        self.ranker = ranker or RuleMetapathRanker()
        self.specs = self.generator.specs

    def select(self, query: str, entities: list[str] | None = None, limit: int = 3) -> list[MetapathSpec]:
        return [selection.spec for selection in self.select_with_trace(query, entities, limit)]

    def select_with_trace(
        self,
        query: str,
        entities: list[str] | None = None,
        limit: int = 3,
    ) -> list[MetapathSelection]:
        candidates = self.generator.generate(query, entities)
        return self.ranker.rank(query, candidates, entities, limit)
