"""
Finance-domain metapath definitions and rule-based routing.

Metapaths encode analyst-style graph patterns such as fund exposure,
sector concentration, and regulatory scope. They are explicit domain
knowledge, not learned model parameters.
"""

from __future__ import annotations

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


class MetapathRouter:
    """Select finance-domain metapaths using transparent query rules."""

    def __init__(self, specs: dict[str, MetapathSpec] | None = None) -> None:
        self.specs = specs or FINANCIAL_METAPATHS
        validate_all_metapaths(self.specs)

    def select(self, query: str, entities: list[str] | None = None, limit: int = 3) -> list[MetapathSpec]:
        lowered = query.lower()
        scored: list[tuple[int, str, MetapathSpec]] = []
        for name, spec in self.specs.items():
            score = sum(1 for keyword in spec.keywords if keyword in lowered)
            if score:
                scored.append((score, name, spec))

        if not scored and entities:
            for name in ("sector_exposure", "geographic_risk", "compliance_chain"):
                scored.append((1, name, self.specs[name]))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [spec for _, _, spec in scored[:limit]]
