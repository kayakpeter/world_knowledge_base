# Extracting the SMB course into the playbook

Goal: turn the course into **testable setup specs** in `references/playbook/`
plus **hard risk rules**. Raw course files stay where they are and are never
committed. The playbook paraphrases rules; it never copies the material
verbatim.

## Step 0: check for an earlier pass (once)

Claude has read the course before. Before redoing work, search once for
earlier notes (qbb_trader `docs/`, `~/`, the SMB folder itself) for files that
mention SMB, playbook or setup names. Record what you find in
`data_locations.md` and import anything useful into the playbook using the
template below. Don't search again in later sessions.

## Step 1: inventory

Write `references/smb_inventory.md`: one row per course file (module, title,
type pdf/video/slides, path, status `unread | extracted | not-a-setup`).
Videos need transcripts first. Run Whisper locally on the GPU and save `.txt`
files next to the videos. Work through the inventory in order. Its status
column is how a later session knows where to resume.

## Step 2: extract per module

For each module, sort its content into one of three buckets:

1. **Setups** → one playbook file each, using the template below.
2. **Risk and process rules** (daily max loss, sizing, when to stop trading,
   review habits) → `references/playbook/_risk_rules.md`. These become hard
   constraints in paper trading, not hypotheses.
3. **Context and concepts** (reading tape, "in play", catalysts, relative
   strength) → `references/playbook/_concepts.md`. These are candidate
   features for the base-rate slices.

## Step 3: translate discretion into data

SMB teaches discretionary trading. For every cue such as "strong", "holding",
"in play" or "good level", write the closest measurable proxy and mark how
confident the translation is. Ambiguous cues are fine; list them. They become
hypotheses ("does proxy A or proxy B capture 'holding VWAP' better?").

## Playbook file template

```markdown
# <Setup name>
status: extracted | in-test | validated | dead
source: <module / video / timestamp>

## Idea (one sentence)
## Selection: which stocks/days (daily data)
## Trigger: exact entry condition (1-min / tick data)
## Stop: initial risk point
## Management: targets, partials, trail, time stop
## Invalidation: when SMB says skip or abort
## Discretionary cues → proxies
| Course language | Measurable proxy | Data needed | Confidence |
## Open questions
## Ledger links: H-xxx
```

When a playbook file is ready, add a hypothesis to the research ledger for its
**selection layer first**, then follow the normal loop.
