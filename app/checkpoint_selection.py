"""Checkpoint selection rules for a single fine-tuning run."""
import math

def should_save_checkpoint(current_dice: float, best_dice: float) -> bool:
    """Save finite improvements independently of pretrained metadata."""
    current = float(current_dice)
    if not math.isfinite(current) or not 0.0 <= current <= 1.0:
        raise ValueError("Validation Dice must be finite and between 0 and 1.")
    return current > float(best_dice)
