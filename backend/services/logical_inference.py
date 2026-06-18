"""Rule-based multi-hop inference over typed financial graph paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from services.metapaths import FINANCIAL_METAPATHS, MetapathResult, MetapathRouter, MetapathSpec


class GraphTraversalService(Protocol):
    async def traverse_metapath(
        self,
        start_entities: list[str],
        metapath: MetapathSpec,
        limit: int = 20,
    ) -> list[MetapathResult]:
        """Return typed graph paths matching a metapath."""


@dataclass(frozen=True)
class InferenceRule:
    name: str
    description: str
    metapath_name: str
    conclusion_template: str
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class InferredFact:
    rule_name: str
    conclusion: str
    evidence: str
    confidence: float
    path: tuple[tuple[str, str, str], ...]
    start_entity: str
    end_entity: str


INFERENCE_RULES: tuple[InferenceRule, ...] = (
    InferenceRule(
        name="fund_sector_exposure",
        description="If a fund holds a company and that company belongs to a sector, infer fund exposure to that sector.",
        metapath_name="sector_exposure",
        conclusion_template="{start} has inferred sector exposure to {end}.",
        keywords=("sector", "industry", "exposure", "concentration", "infer"),
    ),
    InferenceRule(
        name="fund_geographic_exposure",
        description="If a fund holds a company and that company is located in a region, infer geographic exposure.",
        metapath_name="geographic_risk",
        conclusion_template="{start} has inferred geographic exposure to {end}.",
        keywords=("geographic", "geography", "region", "country", "location", "infer"),
    ),
    InferenceRule(
        name="fund_supplier_dependency",
        description="If a fund holds a company and that company depends on a supplier, infer indirect supplier exposure.",
        metapath_name="supply_chain_risk",
        conclusion_template="{start} has inferred supplier dependency exposure to {end}.",
        keywords=("supplier", "supply", "vendor", "dependency", "dependencies", "infer"),
    ),
    InferenceRule(
        name="fund_technology_dependency",
        description="If a fund holds a company and that company uses a technology, infer technology dependency exposure.",
        metapath_name="technology_risk",
        conclusion_template="{start} has inferred technology dependency exposure to {end}.",
        keywords=("technology", "platform", "cloud", "software", "infrastructure", "infer"),
    ),
    InferenceRule(
        name="fund_regulatory_scope",
        description="If a fund holds a company and that company is subject to a regulation, infer regulatory scope.",
        metapath_name="compliance_chain",
        conclusion_template="{start} has inferred regulatory exposure to {end}.",
        keywords=("regulation", "regulatory", "compliance", "basel", "sec", "subject", "infer"),
    ),
    InferenceRule(
        name="sector_peer",
        description="If two companies belong to the same sector, infer they are sector peers.",
        metapath_name="shared_sector",
        conclusion_template="{start} and {end} are inferred sector peers.",
        keywords=("peer", "peers", "same sector", "shared sector", "similar", "infer"),
    ),
    InferenceRule(
        name="management_overlap",
        description="If two companies share a person through works_at relationships, infer management overlap.",
        metapath_name="management_overlap",
        conclusion_template="{start} and {end} have inferred management overlap.",
        keywords=("management", "executive", "board", "director", "overlap", "infer"),
    ),
    InferenceRule(
        name="transitive_ownership",
        description="If company A owns company B and B owns company C, infer indirect ownership exposure from A to C.",
        metapath_name="subsidiary_chain",
        conclusion_template="{start} has inferred indirect ownership exposure to {end}.",
        keywords=("subsidiary", "ownership", "owns", "owned", "parent", "indirect", "infer"),
    ),
)


class LogicalInferenceEngine:
    """Derive explicit facts from typed multi-hop graph paths."""

    def __init__(self, rules: tuple[InferenceRule, ...] = INFERENCE_RULES) -> None:
        self.rules = rules
        self.metapath_router = MetapathRouter()

    async def infer(
        self,
        query: str,
        start_entities: list[str],
        graph: GraphTraversalService,
        limit: int = 10,
    ) -> list[InferredFact]:
        if not start_entities:
            return []

        selected_rules = self.select_rules(query, start_entities)
        inferred: list[InferredFact] = []
        for rule in selected_rules:
            metapath = FINANCIAL_METAPATHS[rule.metapath_name]
            paths = await graph.traverse_metapath(start_entities, metapath, limit=limit)
            for path in paths:
                inferred.append(self._fact_from_path(rule, path))
                if len(inferred) >= limit:
                    return inferred
        return inferred

    def select_rules(self, query: str, start_entities: list[str], limit: int = 4) -> list[InferenceRule]:
        lowered = query.lower()
        scored: list[tuple[int, str, InferenceRule]] = []
        for rule in self.rules:
            score = sum(1 for keyword in rule.keywords if keyword in lowered)
            if score:
                scored.append((score, rule.name, rule))

        if not scored and start_entities:
            routed_metapaths = {spec.name for spec in self.metapath_router.select(query, start_entities, limit=limit)}
            scored = [
                (1, rule.name, rule)
                for rule in self.rules
                if rule.metapath_name in routed_metapaths
            ]

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [rule for _, _, rule in scored[:limit]]

    @staticmethod
    def _fact_from_path(rule: InferenceRule, path: MetapathResult) -> InferredFact:
        conclusion = rule.conclusion_template.format(start=path.start_entity, end=path.end_entity)
        evidence = (
            f"Inference rule {rule.name}: {rule.description} "
            f"Evidence path: {path.evidence}. Therefore: {conclusion}"
        )
        confidence = round(min(1.0, 0.55 + (0.15 * len(path.path))), 4)
        return InferredFact(
            rule_name=rule.name,
            conclusion=conclusion,
            evidence=evidence,
            confidence=confidence,
            path=path.path,
            start_entity=path.start_entity,
            end_entity=path.end_entity,
        )

    @staticmethod
    def as_metadata(fact: InferredFact) -> dict[str, Any]:
        return {
            "rule": fact.rule_name,
            "path": list(fact.path),
            "start_entity": fact.start_entity,
            "end_entity": fact.end_entity,
            "score_method": "rule_based_multi_hop_inference",
        }
