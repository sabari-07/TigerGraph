"""The graph ontology for the Olympic-events corpus.

Node types
----------
Event   : one Olympic event page (the primary entity; maps 1:1 to a doc_id)
Games   : an Olympic edition, e.g. "2012 Summer" / "2010 Winter"
Venue   : a competition venue, e.g. "Eton Dorney"
Athlete : a medalist (gold/silver/bronze); label is the raw infobox value
NOC     : a National Olympic Committee code, e.g. "HUN", "USA"
Sport   : a sport discipline parsed from the title, e.g. "Canoeing"

Edge types
----------
Event  -PART_OF->   Games
Event  -HELD_AT->   Venue
Event  -HAS_SPORT-> Sport
Event  -WON_GOLD->  Athlete   (also WON_SILVER / WON_BRONZE)
Athlete-REPRESENTS->NOC
Games  -NEXT->      Games      (temporal chain; PREV is the reverse)

Event carries typed properties: competitors (int), nations (int),
date_raw (str), win_value (str), year (int), season ("Summer"/"Winter").

This mirrors a TigerGraph schema 1:1 so the local backend and the Savanna
backend expose the same traversals.
"""
from __future__ import annotations

from enum import Enum


class NodeType(str, Enum):
    EVENT = "Event"
    GAMES = "Games"
    VENUE = "Venue"
    ATHLETE = "Athlete"
    NOC = "NOC"
    SPORT = "Sport"


class EdgeType(str, Enum):
    PART_OF = "PART_OF"       # Event -> Games
    HELD_AT = "HELD_AT"       # Event -> Venue
    HAS_SPORT = "HAS_SPORT"   # Event -> Sport
    WON_GOLD = "WON_GOLD"     # Event -> Athlete
    WON_SILVER = "WON_SILVER"
    WON_BRONZE = "WON_BRONZE"
    REPRESENTS = "REPRESENTS"  # Athlete -> NOC
    NEXT = "NEXT"             # Games -> Games (chronological successor)
    PREV = "PREV"             # Games -> Games (chronological predecessor)


MEDAL_EDGE = {
    "gold": EdgeType.WON_GOLD,
    "silver": EdgeType.WON_SILVER,
    "bronze": EdgeType.WON_BRONZE,
}

# GSQL schema for TigerGraph Savanna. Applied by the TigerGraph backend.
GSQL_SCHEMA = """
CREATE VERTEX Event (
    PRIMARY_ID id STRING,
    title STRING, url STRING, event_name STRING,
    competitors INT, nations INT, year INT, season STRING,
    date_raw STRING, win_value STRING
) WITH primary_id_as_attribute="true"
CREATE VERTEX Games (PRIMARY_ID id STRING, label STRING, year INT, season STRING)
    WITH primary_id_as_attribute="true"
CREATE VERTEX Venue (PRIMARY_ID id STRING, name STRING) WITH primary_id_as_attribute="true"
CREATE VERTEX Athlete (PRIMARY_ID id STRING, name STRING) WITH primary_id_as_attribute="true"
CREATE VERTEX NOC (PRIMARY_ID id STRING, code STRING) WITH primary_id_as_attribute="true"
CREATE VERTEX Sport (PRIMARY_ID id STRING, name STRING) WITH primary_id_as_attribute="true"

CREATE DIRECTED EDGE PART_OF (FROM Event, TO Games)
CREATE DIRECTED EDGE HELD_AT (FROM Event, TO Venue)
CREATE DIRECTED EDGE HAS_SPORT (FROM Event, TO Sport)
CREATE DIRECTED EDGE WON_GOLD (FROM Event, TO Athlete)
CREATE DIRECTED EDGE WON_SILVER (FROM Event, TO Athlete)
CREATE DIRECTED EDGE WON_BRONZE (FROM Event, TO Athlete)
CREATE DIRECTED EDGE REPRESENTS (FROM Athlete, TO NOC)
CREATE DIRECTED EDGE NEXT (FROM Games, TO Games)
CREATE DIRECTED EDGE PREV (FROM Games, TO Games)
"""
