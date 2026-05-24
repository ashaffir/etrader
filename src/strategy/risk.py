"""Guardrails: per-trade cap, parallel trades, daily-loss kill switch, cooldown.

The risk layer never opens or closes positions; it only *vetoes* or
*amends* requested actions before the executor sees them. Its outputs
are deterministic given the same inputs, which makes the unit tests
trivial to write.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Iterable, Sequence

from ..config import GuardrailsConfig
from ..state import BotState


@dataclass(frozen=True)
class TradeRequest:
    """A trade request emitted by the decision engine."""

    instrument_id: int
    symbol: str
    action: str          # "BUY" | "CLOSE"
    amount_usd: float    # for BUY only; ignored for CLOSE
    position_id: int | None = None  # required for CLOSE


@dataclass(frozen=True)
class TradeVerdict:
    request: TradeRequest
    approved: bool
    reason: str
    amended_amount_usd: float | None = None  # populated when we cap a BUY


@dataclass
class RiskEvaluator:
    cfg: GuardrailsConfig

    def evaluate(
        self,
        *,
        requests: Sequence[TradeRequest],
        state: BotState,
        current_equity: float | None,
        bot_owned_position_count: int,
    ) -> list[TradeVerdict]:
        # Daily-loss kill switch first — this gates every BUY this cycle.
        kill_switch_active = self._update_and_check_kill_switch(
            state=state,
            current_equity=current_equity,
            bot_owned_position_count=bot_owned_position_count,
        )

        verdicts: list[TradeVerdict] = []
        new_open_count = 0
        for req in requests:
            if req.action == "BUY":
                v = self._evaluate_buy(
                    req=req,
                    state=state,
                    new_open_count=new_open_count,
                    bot_owned_count=bot_owned_position_count,
                    kill_switch_active=kill_switch_active,
                )
                if v.approved:
                    new_open_count += 1
            elif req.action == "CLOSE":
                v = self._evaluate_close(req=req, state=state)
            else:
                v = TradeVerdict(req, False, f"unknown action {req.action!r}")
            verdicts.append(v)
        return verdicts

    # -- per-action guards -----------------------------------------------

    def _evaluate_buy(
        self,
        *,
        req: TradeRequest,
        state: BotState,
        new_open_count: int,
        bot_owned_count: int,
        kill_switch_active: bool,
    ) -> TradeVerdict:
        if kill_switch_active:
            return TradeVerdict(req, False, "daily-loss kill switch active")
        if bot_owned_count + new_open_count >= self.cfg.max_parallel_trades:
            return TradeVerdict(
                req,
                False,
                f"max parallel trades reached ({self.cfg.max_parallel_trades})",
            )
        cooldown = self._cooldown_remaining(req.instrument_id, state)
        if cooldown > 0:
            return TradeVerdict(req, False, f"cooldown {cooldown:.0f}s remaining")
        if req.amount_usd <= 0:
            return TradeVerdict(req, False, "amount_usd must be positive")
        amended: float | None = None
        amount = req.amount_usd
        if amount > self.cfg.max_per_trade_usd:
            amended = float(self.cfg.max_per_trade_usd)
            amount = amended
        return TradeVerdict(
            req,
            True,
            f"approved (amount=${amount:.2f})"
            + ("" if amended is None else f", capped from ${req.amount_usd:.2f}"),
            amended_amount_usd=amended,
        )

    def _evaluate_close(self, *, req: TradeRequest, state: BotState) -> TradeVerdict:
        if req.position_id is None:
            return TradeVerdict(req, False, "CLOSE requires a position_id")
        if req.position_id not in state.bot_owned_positions:
            return TradeVerdict(req, False, "position not bot-owned (refusing to close)")
        return TradeVerdict(req, True, "approved")

    # -- helpers ---------------------------------------------------------

    def _cooldown_remaining(self, instrument_id: int, state: BotState) -> float:
        seconds_since = state.seconds_since_action(instrument_id)
        if seconds_since is None:
            return 0.0
        cooldown_s = self.cfg.per_instrument_cooldown_min * 60
        return max(0.0, cooldown_s - seconds_since)

    def _update_and_check_kill_switch(
        self,
        *,
        state: BotState,
        current_equity: float | None,
        bot_owned_position_count: int = 0,
    ) -> bool:
        """Daily-loss kill switch — bot-attributable drawdown only.

        Two important properties:

        - The kill switch only ever fires when the bot has skin in the
          game today (an open bot-owned position OR a successful BUY/CLOSE
          this session-day). Drawdown caused by the user's manual or
          mirror positions never halts the bot.
        - The equity baseline rebases at UTC day rollover *and* whenever
          the bot has zero exposure and zero actions today, so a recovery
          after the user closes their own losing trade auto-clears a
          previously-set halt.
        """
        today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
        bot_has_skin = (bot_owned_position_count > 0) or (state.bot_actions_today > 0)

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
        if drawdown >= self.cfg.daily_loss_stop_usd:
            state.halted_today = True
            return True
        return False


# ---------------------------------------------------------------------------
# Stop-loss / take-profit price helpers
# ---------------------------------------------------------------------------

def compute_stop_loss_take_profit(
    *,
    entry_price: float,
    is_buy: bool,
    stop_loss_pct: float,
    take_profit_pct: float,
) -> tuple[float, float]:
    """Return ``(stop_loss_rate, take_profit_rate)`` aligned with eToro semantics.

    For a BUY (long): SL is below entry, TP is above entry.
    For a SELL (short): SL is above entry, TP is below entry.
    """
    if entry_price <= 0:
        raise ValueError("entry_price must be positive")
    if is_buy:
        sl = entry_price * (1.0 - stop_loss_pct / 100.0)
        tp = entry_price * (1.0 + take_profit_pct / 100.0)
    else:
        sl = entry_price * (1.0 + stop_loss_pct / 100.0)
        tp = entry_price * (1.0 - take_profit_pct / 100.0)
    sl = max(sl, 1e-4)
    tp = max(tp, 1e-4)
    return round(sl, 4), round(tp, 4)


def aggregate_summary(verdicts: Iterable[TradeVerdict]) -> dict[str, int]:
    approved = denied = capped = 0
    for v in verdicts:
        if v.approved:
            approved += 1
            if v.amended_amount_usd is not None:
                capped += 1
        else:
            denied += 1
    return {"approved": approved, "denied": denied, "capped": capped}
