"""Exact shared-input projection grouping policy for transformer blocks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BlockProjection:
    name: str
    input_features: int
    output_features: int
    base_k_tiles_per_phase: int


def plan_exact_shared_input_groups(
    projections: tuple[BlockProjection, ...],
) -> tuple[tuple[int, ...], ...]:
    """Group compatible projections, with a silicon-accepted maximum of two.

    Equal-width pairs are preferred. This gives Llama gate/up and K/V groups,
    while Pythia's already-combined QKV remains a single member. Three-wide
    native execution is retained only as a fast numerical experiment because
    it does not reproduce the generic graph bit-for-bit on P150a.
    """

    if not projections:
        return ()
    unassigned = set(range(len(projections)))
    result: list[tuple[int, ...]] = []

    def compatible(left: int, right: int) -> bool:
        a = projections[left]
        b = projections[right]
        return (
            a.input_features == b.input_features
            and a.base_k_tiles_per_phase == b.base_k_tiles_per_phase
        )

    # Prefer natural symmetric pairs such as K/V and gate/up.
    for left in range(len(projections)):
        if left not in unassigned:
            continue
        for right in range(left + 1, len(projections)):
            if (
                right in unassigned
                and compatible(left, right)
                and projections[left].output_features
                == projections[right].output_features
            ):
                result.append((left, right))
                unassigned.remove(left)
                unassigned.remove(right)
                break

    # Then pair any remaining compatible projections.
    remaining = sorted(unassigned)
    while remaining:
        left = remaining.pop(0)
        partner = next(
            (
                right
                for right in remaining
                if compatible(left, right)
            ),
            None,
        )
        if partner is None:
            result.append((left,))
        else:
            remaining.remove(partner)
            result.append((left, partner))
    return tuple(sorted(result, key=lambda group: min(group)))
