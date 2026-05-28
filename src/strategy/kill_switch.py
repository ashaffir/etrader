"""Daily-loss kill switch — bot-attributable drawdown only.

Three important properties:

- ``daily_loss_stop_usd <= 0`` **disables** the kill switch entirely.
  In that mode the function never sets ``halted_today`` and actively
  *clears* any previously-set halt (e.g. from a config change mid-day
  or an operator switching the cap off). This is the "always-on,
  always-checking" mode the operator can opt into when they'd rather
  let signal quality + per-trade stops manage risk.
- The kill switch only ever fires when the bot has skin in the game
  today (an open bot-owned position OR a successful BUY/CLOSE this
  session-day). Drawdown caused by the user's manual or mirror
  positions never halts the bot.
- The equity baseline rebases at UTC day rollover *and* whenever the
  bot has zero exposure and zero actions today, so a recovery after
  the user closes their own losing trade auto-clears a previously-set
  halt.

Extracted from :mod:`src.strategy.risk` so the evaluator file stays
under the 300-line cap. Pure function: takes a config + mutable
``BotState`` and returns whether the bot must halt this cycle.
"""

from __future__ import annotations

import datetime as dt

from ..config import GuardrailsConfig
from ..state import BotState


def update_and_check_kill_switch(
    *,
    cfg: GuardrailsConfig,
    state: BotState,
    current_equity: float | None,
    bot_owned_position_count: int = 0,
) -> bool:
    """Return ``True`` when the bot must halt this cycle.

    Side-effects ``state.session_baseline_equity``, ``halted_today``,
    ``halted_day``, ``baseline_day`` per the rules in the module doc.
    """
    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    bot_has_skin = (bot_owned_position_count > 0) or (state.bot_actions_today > 0)
    kill_switch_disabled = cfg.daily_loss_stop_usd <= 0.0

    if state.halted_day != today:
        # New UTC day: reset the daily counters and rebase the baseline.
        state.halted_today = False
        state.halted_day = today
        state.bot_actions_today = 0
        state.session_baseline_equity = current_equity
        state.baseline_day = today
    elif state.baseline_day != today:
        # Rare: persisted baseline carried in from yesterday. Rebase.
        state.session_baseline_equity = current_equity
        state.baseline_day = today

    if kill_switch_disabled:
        # Operator opted into "always-on" mode. Clear any sticky halt
        # left over from a previous config and skip the drawdown check.
        # The baseline is kept so re-enabling mid-day measures drawdown
        # from a real reference point rather than mid-flight equity.
        state.halted_today = False
        return False

    if state.halted_today and not bot_has_skin:
        # Stale halt — there's nothing the bot could have damaged today,
        # so a previous halt cannot be attributed to it. Auto-unstick
        # and rebase against current equity.
        state.halted_today = False
        state.session_baseline_equity = current_equity

    if state.halted_today:
        return True

    if current_equity is None or state.session_baseline_equity is None:
        if state.session_baseline_equity is None and current_equity is not None:
            state.session_baseline_equity = current_equity
            state.baseline_day = today
        return False

    if not bot_has_skin:
        # No bot exposure → drift in the user's own positions can't trip
        # the kill switch. Rebase the baseline so when the bot DOES start
        # trading today, drawdown is measured from that point forward.
        state.session_baseline_equity = current_equity
        return False

    drawdown = state.session_baseline_equity - current_equity
    if drawdown >= cfg.daily_loss_stop_usd:
        state.halted_today = True
        return True
    return False
