"""Convert PHH/PHHS poker histories into one cleaned row per player-hand."""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = [
    "hand_id",
    "player",
    "opponent",
    "position",
    "num_bets",
    "num_calls",
    "num_raises",
    "num_folds",
    "num_checks",
    "total_bet",
    "avg_bet",
    "max_bet",
    "bet_to_stack_ratio",
    "raise_rate",
    "call_rate",
    "fold_rate",
    "aggression_rate",
    "starting_stack",
    "pot_size",
    "num_betting_rounds",
    "went_to_showdown",
    "hand_profit",
    "previous_hand_profit",
    "profit_last_5",
    "profit_last_10",
    "consecutive_losses",
    "baseline_bet",
    "baseline_raise_rate",
    "baseline_call_rate",
    "baseline_fold_rate",
    "baseline_aggression_rate",
    "bet_deviation",
    "raise_deviation",
    "call_deviation",
    "fold_deviation",
    "aggression_deviation",
]


def load_clean_phhs(path: str | Path) -> list[dict]:
    """Load a PHH or multi-hand PHHS file while preserving hand order."""
    with Path(path).open("rb") as stream:
        data = tomllib.load(stream)

    if "actions" in data:
        return [data]

    def sort_key(value: str) -> tuple[int, int | str]:
        return (0, int(value)) if value.isdigit() else (1, value)

    return [data[key] for key in sorted(data, key=sort_key)]


def _position_labels(blinds: list[float], player_count: int) -> list[str]:
    positive = [float(value) for value in blinds if value and float(value) > 0]
    if not positive:
        return [f"seat_{index + 1}" for index in range(player_count)]

    small_blind = min(positive)
    big_blind = max(positive)
    labels = []
    for index in range(player_count):
        blind = float(blinds[index]) if index < len(blinds) else 0.0
        if blind == big_blind:
            labels.append("BB")
        elif blind == small_blind:
            labels.append("SB")
        else:
            labels.append(f"seat_{index + 1}")
    return labels


def _empty_stats() -> dict:
    return {
        "num_bets": 0,
        "num_calls": 0,
        "num_raises": 0,
        "num_folds": 0,
        "num_checks": 0,
        "bet_amounts": [],
    }


def parse_hand_summary(hand: dict, hand_order: int) -> list[dict]:
    """Replay one hand and return one summary row for each player."""
    stacks = [float(value) for value in hand.get("starting_stacks", [])]
    player_count = len(stacks)
    if player_count == 0:
        return []

    players = hand.get("players") or [f"player_{index + 1}" for index in range(player_count)]
    blinds = [float(value or 0) for value in hand.get("blinds_or_straddles", [])]
    antes = [float(value or 0) for value in hand.get("antes", [])]
    positions = _position_labels(blinds, player_count)

    contributions = [0.0] * player_count
    street_commit = [0.0] * player_count
    folded = [False] * player_count
    stats = [_empty_stats() for _ in range(player_count)]

    for index in range(player_count):
        ante = antes[index] if index < len(antes) else 0.0
        blind = blinds[index] if index < len(blinds) else 0.0
        contributions[index] = ante + blind
        street_commit[index] = blind

    current_bet = max(street_commit, default=0.0)
    betting_rounds = 1
    went_to_showdown = False

    for action in hand.get("actions", []):
        tokens = action.split()
        if len(tokens) < 2:
            continue

        if tokens[0] == "d":
            if tokens[1] == "db":
                betting_rounds += 1
                street_commit = [0.0] * player_count
                current_bet = 0.0
            continue

        match = re.fullmatch(r"p(\d+)", tokens[0])
        if not match:
            continue
        player_index = int(match.group(1)) - 1
        if not 0 <= player_index < player_count:
            continue

        verb = tokens[1]
        player_stats = stats[player_index]

        if verb == "f":
            player_stats["num_folds"] += 1
            folded[player_index] = True
        elif verb == "cc":
            amount_to_call = max(0.0, current_bet - street_commit[player_index])
            if amount_to_call > 0:
                player_stats["num_calls"] += 1
                contributions[player_index] += amount_to_call
                street_commit[player_index] += amount_to_call
            else:
                player_stats["num_checks"] += 1
        elif verb == "cbr" and len(tokens) >= 3:
            target_amount = float(tokens[2])
            added_amount = max(0.0, target_amount - street_commit[player_index])
            if current_bet > 0:
                player_stats["num_raises"] += 1
            else:
                player_stats["num_bets"] += 1
            player_stats["bet_amounts"].append(added_amount)
            contributions[player_index] += added_amount
            street_commit[player_index] = target_amount
            current_bet = max(current_bet, target_amount)
        elif verb == "sm":
            went_to_showdown = True

    final_pot = float(sum(contributions))
    hand_id = hand.get("hand", hand_order)
    rows = []

    recorded_results = hand.get("_results") or hand.get("results") or []
    if len(recorded_results) >= player_count:
        resolved_profits = [float(value) for value in recorded_results[:player_count]]
    elif not went_to_showdown and sum(not is_folded for is_folded in folded) == 1:
        # When all other players fold, the remaining player is the winner.
        # This is gross profit reconstructed from the action stream and does
        # not attempt to model rake.
        winner = folded.index(False)
        resolved_profits = [-value for value in contributions]
        resolved_profits[winner] = final_pot - contributions[winner]
    else:
        resolved_profits = None

    for index, player in enumerate(players):
        player_stats = stats[index]
        action_count = sum(
            player_stats[key]
            for key in ("num_bets", "num_calls", "num_raises", "num_folds", "num_checks")
        )
        aggressive_count = player_stats["num_bets"] + player_stats["num_raises"]
        bet_amounts = player_stats["bet_amounts"]
        total_bet = float(sum(bet_amounts))

        opponents = [str(name) for other_index, name in enumerate(players) if other_index != index]
        hand_profit = resolved_profits[index] if resolved_profits is not None else np.nan

        rows.append(
            {
                "_hand_order": hand_order,
                "hand_id": hand_id,
                "player": str(player),
                "opponent": " | ".join(opponents),
                "position": positions[index],
                "num_bets": player_stats["num_bets"],
                "num_calls": player_stats["num_calls"],
                "num_raises": player_stats["num_raises"],
                "num_folds": player_stats["num_folds"],
                "num_checks": player_stats["num_checks"],
                "total_bet": total_bet,
                "avg_bet": float(np.mean(bet_amounts)) if bet_amounts else 0.0,
                "max_bet": float(max(bet_amounts)) if bet_amounts else 0.0,
                "bet_to_stack_ratio": total_bet / stacks[index] if stacks[index] else np.nan,
                "raise_rate": player_stats["num_raises"] / action_count if action_count else 0.0,
                "call_rate": player_stats["num_calls"] / action_count if action_count else 0.0,
                "fold_rate": player_stats["num_folds"] / action_count if action_count else 0.0,
                "aggression_rate": aggressive_count / action_count if action_count else 0.0,
                "starting_stack": stacks[index],
                "pot_size": final_pot,
                "num_betting_rounds": betting_rounds,
                "went_to_showdown": went_to_showdown,
                "hand_profit": hand_profit,
            }
        )

    return rows


def _prior_loss_streak(values: pd.Series) -> pd.Series:
    streak = 0
    output = []
    for value in values:
        output.append(streak)
        if pd.isna(value):
            streak = 0
        elif value < 0:
            streak += 1
        else:
            streak = 0
    return pd.Series(output, index=values.index, dtype="int64")


def add_history_and_baselines(frame: pd.DataFrame) -> pd.DataFrame:
    """Add pre-hand history, rolling baselines, and current-minus-baseline deviations."""
    cleaned = frame.sort_values(["player", "_hand_order"], kind="stable").copy()
    grouped = cleaned.groupby("player", sort=False, group_keys=False)

    cleaned["previous_hand_profit"] = grouped["hand_profit"].shift(1)
    cleaned["profit_last_5"] = grouped["hand_profit"].transform(
        lambda values: values.shift(1).rolling(5, min_periods=1).sum()
    )
    cleaned["profit_last_10"] = grouped["hand_profit"].transform(
        lambda values: values.shift(1).rolling(10, min_periods=1).sum()
    )
    cleaned["consecutive_losses"] = grouped["hand_profit"].transform(_prior_loss_streak)

    baseline_sources = {
        "baseline_bet": "avg_bet",
        "baseline_raise_rate": "raise_rate",
        "baseline_call_rate": "call_rate",
        "baseline_fold_rate": "fold_rate",
        "baseline_aggression_rate": "aggression_rate",
    }
    for baseline_column, source_column in baseline_sources.items():
        cleaned[baseline_column] = grouped[source_column].transform(
            lambda values: values.shift(1).expanding(min_periods=1).mean()
        )

    cleaned["bet_deviation"] = cleaned["avg_bet"] - cleaned["baseline_bet"]
    cleaned["raise_deviation"] = cleaned["raise_rate"] - cleaned["baseline_raise_rate"]
    cleaned["call_deviation"] = cleaned["call_rate"] - cleaned["baseline_call_rate"]
    cleaned["fold_deviation"] = cleaned["fold_rate"] - cleaned["baseline_fold_rate"]
    cleaned["aggression_deviation"] = (
        cleaned["aggression_rate"] - cleaned["baseline_aggression_rate"]
    )

    return cleaned.sort_values("_hand_order", kind="stable")[OUTPUT_COLUMNS].reset_index(drop=True)


def parse_phhs_to_dataframe(path: str | Path) -> pd.DataFrame:
    """Parse a PHH/PHHS file into the complete cleaned modeling table."""
    rows = []
    for hand_order, hand in enumerate(load_clean_phhs(path)):
        rows.extend(parse_hand_summary(hand, hand_order))
    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return add_history_and_baselines(pd.DataFrame(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input .phh or .phhs file")
    parser.add_argument("-o", "--output", type=Path, default=Path("poker_tilt_cleaned.csv"))
    arguments = parser.parse_args()

    cleaned = parse_phhs_to_dataframe(arguments.input)
    cleaned.to_csv(arguments.output, index=False)
    print(f"Parsed {cleaned['hand_id'].nunique():,} hands into {len(cleaned):,} player-hand rows")
    print(f"Saved {len(cleaned.columns)} columns to {arguments.output}")


if __name__ == "__main__":
    main()
