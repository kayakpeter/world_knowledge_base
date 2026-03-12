# KG + Open-Brain Inference Engine — Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:writing-plans to create the implementation plan from this design doc.

**Goal:** Add Neo4j as a persistent graph store alongside the existing NetworkX-based KnowledgeGraphBuilder, connect it to the open-brain semantic search layer, and wire Apollo's causal chain triggers into an AgentDirective bridge that fires messages to agent inboxes when scenario conditions are met.

**Architecture:** Additive (Option A) — NetworkX is kept for in-memory HMM/shock-propagation/Monte Carlo algorithms. Neo4j is added as the persistent, queryable source of truth. Open-brain (Supabase pgvector) receives text summaries of graph nodes as embeddings. An inference trigger engine evaluates rules after each ingest and dispatches directives to agents via the existing message bus.

**Tech Stack:** Neo4j (Python driver `neo4j` already installed), NetworkX (existing), Supabase pgvector via `brain_ingest` (existing), Polars for parquet handling, existing `~/.claude/agent-tools/send.py` message bus.

**Locked decisions (Peter, 2026-03-12):**
1. Neo4j is the graph backend
2. Inference triggers are event-driven (not polled)
3. KG is the primary store; open-brain is the semantic search layer over KG node summaries

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Data Sources                          │
│  News Parquets  │  Interpretation Parquets  │  Shared State│
└────────┬────────┴──────────┬────────────────┴──────┬─────┘
         │                   │                        │
         ▼                   ▼                        ▼
┌─────────────────────────────────────────────────────────┐
│               Ingestion Pipeline (kg_ingest.py)          │
│  reads parquets → upserts nodes/edges → fires triggers   │
└────────────────────────┬────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
  ┌─────────────┐  ┌──────────┐  ┌──────────────────┐
  │   Neo4j     │  │NetworkX  │  │  open-brain       │
  │  (persist)  │  │(runtime) │  │  (semantic search)│
  └──────┬──────┘  └────┬─────┘  └──────────────────┘
         │              │
         └──────┬───────┘
                ▼
      ┌──────────────────────┐
      │   Inference Engine    │
      │   event_triggers.py   │
      │   (Apollo causal rules)│
      └──────────┬───────────┘
                 │
                 ▼
      ┌──────────────────────┐
      │   AgentDirective      │
      │   Bridge              │
      │   → agent inboxes     │
      └──────────────────────┘
```

**Data flow per ingest cycle:**
1. New interpretation parquet arrives in `data/interpretations/`
2. `kg_ingest.py` parses rows → upserts Event/Stat/Actor nodes in Neo4j
3. Triggers evaluated → matching rules fire AgentDirective messages
4. Node summaries posted to open-brain for embedding
5. Affected NetworkX subgraph invalidated and reloaded from Neo4j

---

## Neo4j Schema — Three-Layer Ontology

### Layer A — Structural (years half-life)

```cypher
(:Country {iso3, name, region, gdp_usd, population})
(:Commodity {symbol, category, unit})
(:TradeRoute {id, from_iso3, to_iso3, commodity, daily_volume_bbl})
(:Infrastructure {id, name, type, country_iso3, risk_tier})
  // type: PORT | PIPELINE | REFINERY | NPP | MILITARY_BASE
```

Seeded once from existing `TRADE_CORRIDORS` dict in `knowledge_base/graph_builder.py`.

### Layer B — Political (weeks–months half-life)

```cypher
(:Actor {id, name, faction, country_iso3})
  // e.g. IRGC, IDF, USN, NATO, UK_FORCES
(:Scenario {id, name, probability, active, last_updated})
  // e.g. HORMUZ_CLOSURE, ABQAIQ_STRIKE, NUCLEAR_INCIDENT, RUSSIA_PROXY_WAR
(:Stat {country_iso3, stat_name, value, direction, estimated_at})
  // from interpretation parquet fields: affected_stats, stat_direction, estimated_magnitude
```

### Layer C — Signal (hours–days, TTL 72h)

```cypher
(:Event {id, headline, published_at, urgency, sentiment, causal_chain, confidence})
  // id = interpretation parquet item_id
(:Flag {name, value, set_at, set_by})
  // mirrors shared_state geopolitical flags
```

### Relationships

```cypher
(Country)-[:CONTROLS]->(Infrastructure)
(Country)-[:TRADES_WITH {commodity, volume}]->(Country)
(Actor)-[:OPERATES_IN]->(Country)
(Event)-[:AFFECTS {direction, magnitude}]->(Country|Commodity|Infrastructure)
(Event)-[:ESCALATES]->(Scenario)
(Scenario)-[:THREATENS]->(Infrastructure|TradeRoute)
(Stat)-[:MEASURES]->(Country)
(Flag)-[:MODIFIES]->(Scenario)
```

**Upsert semantics:** All writes use `MERGE` on the natural key (iso3, item_id, flag name, etc.). No duplicates possible.

**Concurrent write safety:** Neo4j transactions are ACID. The existing `fcntl` lock on `run_news_dispatch.py` prevents simultaneous ingestion runs at the process level.

---

## Ingestion Pipeline (`kg_ingest.py`)

Called from `run_news_dispatch.py` after each dispatch cycle (or independently for backfill).

```python
def ingest_parquet(path: Path, neo4j_driver: Driver, dry_run: bool = False) -> int:
    """Parse one interpretation parquet → upsert Neo4j nodes → fire triggers → sync open-brain."""
```

**Steps per row in the parquet:**
1. Upsert `Event` node (keyed on `item_id`)
2. Upsert `Stat` node per entry in `affected_stats` (keyed on `country_iso3 × stat_name`)
3. Create `AFFECTS` edges from Event → Country (from `country_iso3` + `cross_country_iso3`)
4. Derive `ESCALATES` edges: map `causal_chain` text to Scenario nodes via lookup table
5. After all rows: call `evaluate_triggers(driver)` → fires matching AgentDirective messages
6. Post node summaries to open-brain batch
7. Invalidate NetworkX snapshot for affected countries (lazy reload on next algorithm call)

**TTL cleanup** (`kg_ttl_cleanup.py`, runs nightly via cron):
```cypher
MATCH (e:Event)
WHERE e.published_at < datetime() - duration('PT72H')
  AND NOT (e)-[:ESCALATES]->(:Scenario {active: true})
DELETE e
```

---

## Inference Triggers (`event_triggers.py`)

Rule table evaluated after every ingest. Rules check Neo4j `Flag` nodes (synced from shared_state) and `Scenario` node states.

```python
TRIGGERS: list[dict] = [
    # Apollo scenario subgraph 1: Global Oil Shock
    {
        "id": "OIL_SHOCK_ACTIVATE",
        "conditions": {
            "flag:HORMUZ_STATUS": "EFFECTIVE_CLOSURE",
            "flag:TANKER_WAR_STATUS": "OPERATIONAL",
        },
        "action": "activate_scenario",
        "scenario_id": "GLOBAL_OIL_SHOCK",
        "directive": {
            "to": "prometheus",
            "type": "status",
            "subject": "KG TRIGGER: GLOBAL_OIL_SHOCK scenario activated",
            "body": "Hormuz closure + active tanker war confirmed in KG. Review energy_bonus config.",
        },
    },
    # Apollo scenario subgraph 2: Nuclear Threshold
    {
        "id": "BUSHEHR_WATCH",
        "conditions": {
            "flag:BUSHEHR_NPP_RISK": "CRITICAL",
            "flag:NUCLEAR_FLAG": "False",
        },
        "action": "alert",
        "directive": {
            "to": "apollo",
            "type": "task",
            "subject": "KG TRIGGER: Bushehr NPP risk CRITICAL — nuclear threshold watch",
            "body": "NUCLEAR_FLAG still False but BUSHEHR_NPP_RISK is CRITICAL. Any strike = immediate flag.",
        },
    },
    # Apollo scenario subgraph 3: Russia proxy escalation
    {
        "id": "RUSSIA_PROXY_WAR",
        "conditions": {
            "flag:RUSSIA_IRAN_MILITARY_COORDINATION": "CONFIRMED_UK_DEFMIN_LEVEL",
            "flag:NATO_KINETIC_ENGAGEMENT": "True",
        },
        "action": "activate_scenario",
        "scenario_id": "NATO_RUSSIA_PROXY_WAR",
        "directive": {
            "to": "apollo",
            "type": "task",
            "subject": "KG TRIGGER: NATO-Russia proxy war scenario activated",
            "body": "Russia drone tactics confirmed + NATO kinetic engagement = proxy war scenario active. Review CONFLICT_DURATION.",
        },
    },
    # Apollo scenario subgraph 4: Commodity cascade
    {
        "id": "COMMODITY_CASCADE",
        "conditions": {
            "flag:COMMODITY_CASCADE": "ACTIVE",
            "flag:OIL_SUPPLY_DISRUPTION": "HISTORIC_SCALE",
        },
        "action": "alert",
        "directive": {
            "to": "apollo",
            "type": "task",
            "subject": "KG TRIGGER: Commodity cascade at historic scale — EM contagion risk",
            "body": "Multi-commodity disruption at HISTORIC_SCALE. Food security + metals = second-order EM equity contagion. Review position sizing for EM-exposed names.",
        },
    },
    # Apollo scenario subgraph 5: Ceasefire pathway collapse
    {
        "id": "CEASEFIRE_COLLAPSE",
        "conditions": {
            "flag:QATAR_UNDER_ATTACK": "True",
            "flag:OMAN_SALALAH_STRUCK": "True",
        },
        "action": "alert",
        "directive": {
            "to": "apollo",
            "type": "task",
            "subject": "KG TRIGGER: Both ceasefire backchannels compromised — pathway collapse",
            "body": "Qatar ceasefire channel CLOSED + Oman Salalah struck. No viable backchannel. Oil long thesis reinforced.",
        },
    },
]
```

**Idempotency:** Each trigger fires at most once per scenario activation. State tracked via `(:Scenario {active: true, activated_at})` in Neo4j. A trigger that has already activated a scenario will not re-fire unless the scenario is explicitly reset.

**AgentDirective bridge:** `evaluate_triggers()` calls `send.py` via subprocess (same pattern as all other agent-tools calls) for each matching trigger. Uses `--no-reply-expected` for alerts, `--type task` for config-change directives.

---

## Morning Briefing (`kg_morning_brief.py`)

Runs at 06:00 ET via cron. Executes Apollo's 5 pre-encoded Cypher queries against Neo4j and dispatches the result to Apollo's inbox.

**Apollo's 5 briefing queries:**
```cypher
-- Q1: Highest-urgency events in last 24h
MATCH (e:Event) WHERE e.published_at > datetime() - duration('PT24H')
  AND e.urgency IN ['critical','high']
RETURN e ORDER BY e.urgency, e.published_at DESC LIMIT 20

-- Q2: Active scenarios with probability changes
MATCH (s:Scenario {active: true})
RETURN s.name, s.probability, s.last_updated ORDER BY s.probability DESC

-- Q3: Infrastructure at risk (Layer A nodes with threatening scenarios)
MATCH (s:Scenario {active: true})-[:THREATENS]->(i:Infrastructure)
RETURN s.name, i.name, i.type, i.country_iso3

-- Q4: Causal chain propagation (Event → Scenario → Infra chains)
MATCH path = (e:Event)-[:ESCALATES]->(s:Scenario)-[:THREATENS]->(i:Infrastructure)
WHERE e.published_at > datetime() - duration('PT24H')
RETURN e.headline, s.name, i.name LIMIT 10

-- Q5: Cross-country contagion (multi-hop AFFECTS)
MATCH (c1:Country)<-[:AFFECTS]-(e:Event)-[:AFFECTS]->(c2:Country)
WHERE e.published_at > datetime() - duration('PT24H')
  AND c1 <> c2
RETURN e.headline, c1.iso3, c2.iso3 LIMIT 15
```

---

## Open-Brain Bridge (`kg_to_openbrain.py`)

Syncs Neo4j → open-brain after each ingestion cycle.

**Summary templates:**
- `Event`: `"{headline} [{urgency}] — affects {countries}, chain: {causal_chain}, confidence: {confidence}%"`
- `Scenario`: `"Scenario {name}: probability={probability}% active={active} — threatens {infra_list}"`
- `Flag`: `"Geopolitical flag {name}={value} (set by {set_by} at {set_at})"`

**Tags on all thoughts:** `subsystem=geopolitical`, `agent=apollo`

**Batch pattern:** Calls existing `brain_ingest.post_batch_summary(BATCH_ID, rows, NARRATIVE)` at end of each sync. BATCH_ID derived from ingestion parquet filename.

---

## Error Handling

| Failure | Behaviour |
|---------|-----------|
| Neo4j unreachable at startup | Log error, skip Neo4j upsert, continue — parquet still processed |
| Neo4j transaction failure | Retry once with exponential backoff; log and skip row on second failure |
| Trigger dispatch failure | Log warning, continue — don't block ingestion for messaging failures |
| open-brain sync failure | Log warning, continue — KG ingest is independent of embedding sync |
| NetworkX reload failure | Log error, keep stale snapshot — algorithms still work with old data |

Neo4j is additive — if it's down, existing behaviour (NetworkX + open-brain) continues unchanged.

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `processing/kg_ingest.py` | Create | Main ingestion pipeline |
| `processing/event_triggers.py` | Create | Trigger rule table + evaluator |
| `processing/kg_to_openbrain.py` | Create | Neo4j → open-brain sync |
| `processing/kg_morning_brief.py` | Create | Morning briefing queries → Apollo inbox |
| `processing/kg_ttl_cleanup.py` | Create | Nightly Layer C TTL cleanup |
| `knowledge_base/neo4j_client.py` | Create | Neo4j driver wrapper (connection, upsert helpers) |
| `knowledge_base/graph_builder.py` | Modify | Add `load_from_neo4j()` to replace NetworkX seed from Neo4j |
| `run_news_dispatch.py` | Modify | Call `kg_ingest.py` after dispatch cycle |
| `run_daily_ingestion.sh` | Modify | Add cron entries for morning brief + TTL cleanup |
| `tests/test_kg_ingest.py` | Create | Unit tests for ingest + trigger logic |
| `tests/test_event_triggers.py` | Create | Unit tests for trigger evaluation |

---

## Out of Scope

- Neo4j server installation/configuration (infrastructure, not code)
- Migrating historical interpretation parquets (separate backfill script, future work)
- GraphQL/REST API over Neo4j (not needed; agents query via Python driver directly)
- Removing NetworkX (kept as runtime compute layer)

---

*Design approved by Peter, 2026-03-12. Inputs from Apollo (12 causal chains, 5 scenario subgraphs, 5 briefing queries) and Minerva (batch interpretations_minerva_20260312_190047) incorporated.*
