from pathlib import Path

# Every region's outputs are written to a folder of its own in here, so runs
# for different regions never write over each other.
OUTPUT_DIR = Path("output")


def region_dir(las: list[str]) -> Path:
    """The folder a region's outputs are written to: its local authorities'
    names, as the register spells them, joined by "+" in alphabetical order.

    Args:
        las (list[str]): The region's local authorities.

    Returns:
        Path: A folder in OUTPUT_DIR.
    """
    return OUTPUT_DIR / "+".join(sorted(las))
