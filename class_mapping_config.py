"""
Class mapping configuration for MMS point cloud classification
Maps detailed CloudCompare labels to 5 main categories
"""

# Your original detailed classification labels
ORIGINAL_CLASSES = {
    1: "Unclassified",
    2: "Ground",
    3: "Low vegetation",
    5: "High vegetation",
    6: "Building",
    7: "Noise (falling snow/isolated points)",
    9: "Water",
    11: "Road surface dry",
    22: "Road surface moist",
    23: "Road surface water",
    24: "Pavement markings",
    31: "Snow (slush)",
    32: "Snow",
    35: "Sign pole",
    36: "Utility pole",
    37: "Overhead road sign",
    38: "Guardrail",
    41: "Fence",
    42: "Wire",
    43: "Overpass bridge",
    51: "Vehicle (oncoming lane)",
    52: "Vehicle (following)",
    53: "Vehicle (parallel/adjacent lane)",
    54: "Vehicle (other)",
}

# Target 5 classes for RandLA-Net
TARGET_CLASSES = {
    0: "Road",
    1: "Snow",
    2: "Vehicle",
    3: "Vegetation",
    4: "Others"
}

# Mapping from original classes to target classes
CLASS_MAPPING = {
    # Unclassified -> Others
    1: 4,

    # Ground -> Road
    2: 0,

    # Vegetation -> Vegetation
    3: 3,  # Low veg
    5: 3,  # High veg

    # Building -> Others
    6: 4,

    # Noise (falling snow) -> Snow
    7: 1,

    # Water -> Others
    9: 4,

    # All Road surfaces -> Road
    11: 0,  # Dry
    22: 0,  # Moist
    23: 0,  # Water
    24: 0,  # Pavement markings

    # Snow -> Snow
    31: 1,  # Slush
    32: 1,  # Snow

    # Infrastructure -> Others
    35: 4,  # Sign pole
    36: 4,  # Utility pole
    37: 4,  # Overhead road sign
    38: 4,  # Guardrail
    41: 4,  # Fence
    42: 4,  # Wire
    43: 0,  # Overpass bridge -> Road (it's a road structure)

    # All Vehicles -> Vehicle
    51: 2,  # Oncoming
    52: 2,  # Following
    53: 2,  # Parallel/adjacent
    54: 2,  # Other
}

def get_class_distribution_summary(original_classes, original_counts):
    """
    Summarize how original classes map to target classes

    Args:
        original_classes: Array of original class IDs
        original_counts: Array of counts for each class

    Returns:
        Dictionary with target class distributions
    """
    target_distribution = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    for orig_cls, count in zip(original_classes, original_counts):
        target_cls = CLASS_MAPPING.get(int(orig_cls), 4)  # Default to Others
        target_distribution[target_cls] += count

    return target_distribution


def print_mapping_summary():
    """Print a summary of the class mapping"""
    print("=" * 80)
    print("CLASS MAPPING SUMMARY")
    print("=" * 80)

    for target_id, target_name in TARGET_CLASSES.items():
        print(f"\n{target_id}. {target_name.upper()}:")
        for orig_id, mapped_id in CLASS_MAPPING.items():
            if mapped_id == target_id:
                orig_name = ORIGINAL_CLASSES.get(orig_id, f"Class {orig_id}")
                print(f"   ← Class {orig_id:2d}: {orig_name}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    print_mapping_summary()
