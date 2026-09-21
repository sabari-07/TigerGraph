"""Abstract graph backend interface.

Both the local in-memory backend and the TigerGraph Savanna backend implement
this, so pipelines and agents are backend-agnostic. The methods are the graph
"tools" the agents call.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Node:
    id: str
    type: str
    props: dict[str, Any] = field(default_factory=dict)


class GraphBackend(ABC):
    """Common query surface for graph traversal tools."""

    @abstractmethod
    def get_node(self, node_id: str) -> Optional[Node]:
        ...

    @abstractmethod
    def neighbors(
        self, node_id: str, edge_type: Optional[str] = None, direction: str = "out"
    ) -> list[Node]:
        """Return neighbor nodes along an edge type. direction: out|in|any."""

    @abstractmethod
    def find_nodes(self, node_type: str, **prop_filters: Any) -> list[Node]:
        """Find nodes of a type matching exact property filters."""

    @abstractmethod
    def events_in_games(self, games_id: str, sport: Optional[str] = None) -> list[Node]:
        """All Event nodes belonging to a Games edition, optionally by sport.

        This is the key aggregation/superlative primitive: gather the full set
        of events so we can count or find the max, rather than relying on
        top-k similarity which misses documents.
        """

    @abstractmethod
    def games_by_year_season(self, year: int, season: str) -> Optional[Node]:
        ...

    @abstractmethod
    def adjacent_games(self, games_id: str, direction: str) -> Optional[Node]:
        """direction: 'prev' | 'next' along the temporal chain."""

    @abstractmethod
    def events_at_venue(self, venue_name: str) -> list[Node]:
        ...

    def find_event_by_title(self, title: str) -> Optional[Node]:
        """Find an Event node by its exact title (fuzzy fallback allowed).

        Default returns None; backends override. Kept non-abstract so existing
        backends remain valid.
        """
        return None

    @abstractmethod
    def stats(self) -> dict[str, int]:
        ...
