"""Guardrails: per-trade cap, parallel trades, daily-loss kill switch, cooldown.

The risk layer never opens or closes positions; it only *vetoes* or
*amends* requested actions before the executor sees them. Its outputs
are deterministic given the same inputs, which makes the unit tests
trivial to write.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from ..config import GuardrailsConfig
from ..state import BotState
from .directives import Directives
from .kill_switch import update_and_check_kill_switch


@dataclass(frozen=True)
class TradeRequest:
    """A trade request emitted by the decision engine.

    ``action`` ∈ ``{"BUY", "CLOSE", "MODIFY_STOPS"}``.

    Per-action fields:

    - ``BUY``: ``amount_usd`` is the cash to commit.
    - ``CLOSE``: ``position_id`` is required. ``close_fraction``
      (optional, in ``(0, 1]``) closes only that fraction of units.
      Defaults to ``None`` (full close).
    - ``MODIFY_STOPS``: ``position_id`` is required. At least one of
      ``stop_loss_pct``, ``take_profit_pct``, ``trailing_stop_pct``
      must be set. The risk layer validates ranges; the executor
      applies them to the :class:`DynamicStopsStore` synthetically —
      eToro's API has no modify-position endpoint, so this never
      hits the broker.
    """

    instrument_id: int
    symbol: str
    action: str
    amount_usd: float
    position_id: int | None = None
    # CLOSE-only: fraction of units to deduct (0,1]; None = full close.
    # Resolved at the cycle layer (which has access to the live position
    # units) into ``close_units`` before the request reaches the executor.
    close_fraction: float | None = None
    close_units: float | None = None
    # MODIFY_STOPS-only: any field that is ``None`` is left unchanged.
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    trailing_stop_pct: float | None = None
    rationale: str = ""  # free-text reason from the LLM, surfaced in alerts


@dataclass(frozen=True)
class TradeVerdict:
    request: TradeRequest
    approved: bool
    reason: str
    amended_amount_usd: float | None = None  # populated when we cap a BUY


@dataclass
class RiskEvaluator:
    cfg: GuardrailsConfig
    # Optional resolver for the operator-directives snapshot. Kept as
    # a callable (instead of a stored ``Directives``) so the evaluator
    # always reads the *latest* directives even when the operator
    # edits them between cycles. The default returns a no-op snapshot
    # so legacy call sites (and tests) keep working unchanged.
    directives_provider: Callable[[], Directives] | None = None

    def evaluate(
        self,
        *,
        requests: Sequence[TradeRequest],
        state: BotState,
        current_equity: float | None,
        bot_owned_position_count: int,
        bot_invested_total_usd: float = 0.0,
        account_invested_total_usd: float = 0.0,
    ) -> list[TradeVerdict]:
        # Daily-loss kill switch first — this gates every BUY this cycle.
        kill_switch_active = self._update_and_check_kill_switch(
            state=state,
            current_equity=current_equity,
            bot_owned_position_count=bot_owned_position_count,
        )

        directives = (
            self.directives_provider() if self.directives_provider is not None
            else Directives()
        )
        verdicts: list[TradeVerdict] = []
        new_open_count = 0
        invested_this_cycle: float = 0.0
        for req in requests:
            if req.action == "BUY":
                v = self._evaluate_buy(
                    req=req,
                    state=state,
                    new_open_count=new_open_count,
                    bot_owned_count=bot_owned_position_count,
                    bot_invested_total_usd=bot_invested_total_usd,
                    invested_this_cycle=invested_this_cycle,
                    kill_switch_active=kill_switch_active,
                    directives=directives,
                    account_invested_total_usd=account_invested_total_usd,
                )
                if v.approved:
                    new_open_count += 1
                    invested_this_cycle += float(
                        v.amended_amount_usd if v.amended_amount_usd is not None
                        else req.amount_usd
                    )
            elif req.action == "CLOSE":
                v = self._evaluate_close(req=req, state=state)
            elif req.action == "MODIFY_STOPS":
                v = self._evaluate_modify_stops(req=req, state=state)
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
        bot_invested_total_usd: float,
        invested_this_cycle: float,
        kill_switch_active: bool,
        directives: Directives,
        account_invested_total_usd: float,
    ) -> TradeVerdict:
        if kill_switch_active:
            return TradeVerdict(req, False, "daily-loss kill switch active")
        if directives.is_symbol_blocked(req.symbol):
            return TradeVerdict(
                req, False,
                f"directive blocked_symbols: {req.symbol}",
            )
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

        # First clamp: per-trade cap (existing behaviour).
        amended: float | None = None
        amount = req.amount_usd
        if amount > self.cfg.max_per_trade_usd:
            amended = float(self.cfg.max_per_trade_usd)
            amount = amended

        # Second clamp: bot-wide total-invested budget. The headroom
        # accounts for everything the bot already has on the broker
        # PLUS BUYs we've approved earlier in this same cycle (so a
        # cycle that emits multiple BUYs can't collectively bust the
        # cap). A cap value of 0 means "disabled".
        amount, amended, budget_reason = self._apply_budget_cap(
            request_amount=req.amount_usd,
            amount=amount,
            amended=amended,
            bot_invested_total_usd=bot_invested_total_usd,
            invested_this_cycle=invested_this_cycle,
        )
        if budget_reason is not None:
            return TradeVerdict(req, False, budget_reason)

        # Third clamp: account-wide total-invested directive (bot +
        # manual + mirror). Refuses new buys (does NOT close manual
        # positions). 0 = disabled.
        amount, amended, account_reason = self._apply_account_total_cap(
            directives=directives,
            account_invested_total_usd=account_invested_total_usd,
            invested_this_cycle=invested_this_cycle,
            amount=amount,
            amended=amended,
        )
        if account_reason is not None:
            return TradeVerdict(req, False, account_reason)

        return TradeVerdict(
            req,
            True,
            f"approved (amount=${amount:.2f})"
            + ("" if amended is None else f", capped from ${req.amount_usd:.2f}"),
            amended_amount_usd=amended,
        )

    def _apply_account_total_cap(
        self,
        *,
        directives: Directives,
        account_invested_total_usd: float,
        invested_this_cycle: float,
        amount: float,
        amended: float | None,
    ) -> tuple[float, float | None, str | None]:
        """Apply the operator's ``max_total_account_invested_usd`` directive.

        Unlike ``max_bot_invested_usd``, this includes the user's
        manual + mirror positions. The intent is to act as a TOTAL
        portfolio brake on new BUYs — the bot still refuses to close
        manual positions, but it won't push the account further past
        a ceiling the operator has set. 0 = disabled.
        """
        cap = float(getattr(directives, "max_total_account_invested_usd", 0.0))
        if cap <= 0:
            return amount, amended, None
        already_committed = float(account_invested_total_usd) + float(invested_this_cycle)
        headroom = cap - already_committed
        if headroom <= 0:
            return amount, amended, (
                f"directive max_total_account_invested_usd exhausted "
                f"(${already_committed:.2f} / ${cap:.2f})"
            )
        if amount <= headroom:
            return amount, amended, None
        floor = float(self.cfg.min_amend_remainder_usd)
        if headroom < floor:
            return amount, amended, (
                f"directive max_total_account_invested_usd would leave only "
                f"${headroom:.2f} headroom (< ${floor:.2f} amend floor); rejecting"
            )
        return headroom, headroom, None

    def _apply_budget_cap(
        self,
        *,
        request_amount: float,
        amount: float,
        amended: float | None,
        bot_invested_total_usd: float,
        invested_this_cycle: float,
    ) -> tuple[float, float | None, str | None]:
        """Apply the total-invested cap.

        Returns ``(final_amount, final_amended, reject_reason)``. When
        ``reject_reason`` is not None the caller should refuse the BUY.
        """
        budget_cap = float(self.cfg.max_bot_invested_usd)
        if budget_cap <= 0:
            return amount, amended, None
        already_committed = bot_invested_total_usd + invested_this_cycle
        headroom = budget_cap - already_committed
        if headroom <= 0:
            return amount, amended, (
                f"bot budget exhausted "
                f"(${already_committed:.2f} / ${budget_cap:.2f})"
            )
        if amount <= headroom:
            return amount, amended, None
        # Need to amend down. Floor: don't post a tiny trade.
        floor = float(self.cfg.min_amend_remainder_usd)
        if headroom < floor:
            return amount, amended, (
                f"bot budget would leave only ${headroom:.2f} headroom "
                f"(< ${floor:.2f} amend floor); rejecting"
            )
        return headroom, headroom, None

    def _evaluate_close(self, *, req: TradeRequest, state: BotState) -> TradeVerdict:
        if req.position_id is None:
            return TradeVerdict(req, False, "CLOSE requires a position_id")
        if req.position_id not in state.bot_owned_positions:
            return TradeVerdict(req, False, "position not bot-owned (refusing to close)")
        if req.close_fraction is not None:
            frac = float(req.close_fraction)
            if not (0.0 < frac <= 1.0):
                return TradeVerdict(
                    req, False,
                    f"close_fraction must be in (0,1]; got {frac:.4f}",
                )
        return TradeVerdict(
            req, True,
            "approved (partial close)" if (req.close_fraction or 1.0) < 1.0
            else "approved",
        )

    def _evaluate_modify_stops(
        self, *, req: TradeRequest, state: BotState,
    ) -> TradeVerdict:
        """Validate the LLM's MODIFY_STOPS request.

        Guards:
        - position must be bot-owned (refuse to touch user's positions)
        - at least one of SL / TP / trailing must be supplied
        - any supplied % must be in a sensible band (0, 50] — wider is
          almost certainly a mistake the LLM is making (e.g. emitting
          a price instead of a percentage)
        """
        if req.position_id is None:
            return TradeVerdict(req, False, "MODIFY_STOPS requires a position_id")
        if req.position_id not in state.bot_owned_positions:
            return TradeVerdict(
                req, False,
                "MODIFY_STOPS refused: position not bot-owned",
            )
        if (
            req.stop_loss_pct is None
            and req.take_profit_pct is None
            and req.trailing_stop_pct is None
        ):
            return TradeVerdict(
                req, False,
                "MODIFY_STOPS requires at least one of stop_loss_pct, "
                "take_profit_pct, trailing_stop_pct",
            )
        for label, value in (
            ("stop_loss_pct", req.stop_loss_pct),
            ("take_profit_pct", req.take_profit_pct),
            ("trailing_stop_pct", req.trailing_stop_pct),
        ):
            if value is None:
                continue
            try:
                f = float(value)
            except (TypeError, ValueError):
                return TradeVerdict(req, False, f"{label} not numeric")
            if f <= 0.0 or f > 50.0:
                return TradeVerdict(
                    req, False,
                    f"{label}={f} out of allowed band (0, 50]",
                )
        return TradeVerdict(req, True, "approved (MODIFY_STOPS)")

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
        return update_and_check_kill_switch(
            cfg=self.cfg,
            state=state,
            current_equity=current_equity,
            bot_owned_position_count=bot_owned_position_count,
        )


# Back-compat re-exports — the pricing helpers used to live here. They
# moved to ``risk_pricing.py`` so this file stays under the line cap.
# Existing imports (tests + executor) keep working.
from .risk_pricing import (  # noqa: E402  (intentional late import)
    aggregate_summary,
    compute_stop_loss_take_profit,
)

__all__ = [
    "TradeRequest",
    "TradeVerdict",
    "RiskEvaluator",
    "aggregate_summary",
    "compute_stop_loss_take_profit",
]
