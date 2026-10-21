# Trading skills

`edge-discovery/` is a Claude Code skill. It lives here only because this was
the repo open in the session. Its real home is the trading repo.

## Install (once, on the workstation)

```bash
mkdir -p /media/peter/fast-storage/projects/qbb_trader/.claude/skills
cp -r /home/<you>/world_knowledge_base/trading_skills/edge-discovery \
      /media/peter/fast-storage/projects/qbb_trader/.claude/skills/
```

Commit it in qbb_trader. As a **project skill** it travels with the code, and
Claude can update `references/research_ledger.md` and `data_locations.md` in
place. That is what makes the loop self-improving. A skill uploaded to
claude.ai is a read-only copy, so ledger edits would not persist.

## Use

In a Claude Code session started in qbb_trader, say something like
"run edge-discovery" or "what should we test next?". The skill's description
triggers it. First run: Claude fills `data_locations.md`, then runs the two
scripts to produce the Day-2 base-rate table.
