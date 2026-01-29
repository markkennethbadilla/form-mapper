# AI Agent Instructions

## ONE-SHOT EXECUTION - DO NOT ASK FOR CONFIRMATION

Complete ALL tasks to 100% in one turn. Execute → Fix Errors → Test → Report done.

**Exception — UI/Manual Testing:**
- When testing requires visual verification or user interaction, perform step-by-step manual testing
- Document each verification step with actual results
- Take screenshots or save output when helpful
- Confirm each step before proceeding to the next
- This exception supersedes the one-shot rule only for manual verification workflows

**Why This Matters:**
- Reduces back-and-forth iterations
- Maximizes efficiency and cost-effectiveness
- Delivers working solutions faster
- Respects user's time and context limits

---

## Critical Rules

### Git Workflow
- Create `feature/name` or `fix/name` branch BEFORE making changes
- NO rebase, amend, force push, or reset --hard
- Keep changes local until explicitly told to commit/push
- Commit message format: `type(scope): descriptive message`
- Common types: feat, fix, docs, refactor, test, chore

### Code Quality
- Only modify code directly related to the task
- Match existing code style exactly (indentation, naming, patterns)
- Don't remove code you think is "unused" without verification
- Don't reformat files that aren't being modified
- Fix ALL errors and warnings before reporting completion
- **NECESSITY CHECK:** Before adding ANY code, ask: "Is this REQUIRED by the acceptance criteria?" If not, DON'T add it.
- **BEFORE RESPONDING:** Review all changes against the baseline. Remove unnecessary modifications:
  - Import reordering
  - Whitespace-only changes
  - Code rearrangements that don't affect functionality
  - Any changes not directly required by acceptance criteria
  - Only keep changes that implement requirements
- **WHEN REVISING IMPLEMENTATION:** If approach changes, remove ALL code from previous implementation that is no longer needed. Don't leave dead code, unused functions, or orphaned logic.

### Implementation Philosophy
- **Read requirements carefully** - Don't over-engineer or add features not requested
- **Keep it simple** - Use existing patterns and tools in the project
- **Don't assume** - If unclear, look at similar existing implementations first
- **Test locally** - Verify changes work before declaring completion
- **Follow project conventions** - Match the existing architectural patterns

---
