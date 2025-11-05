"""
Read CloudCompare .bin format files
CloudCompare binary format has a specific header structure
"""

import numpy as np
import struct
import laspy
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_cloudcompare_bin(bin_path: str):
    """
    Read CloudCompare .bin file format

    CloudCompare .bin format structure:
    - Header with magic number "CCB2" or similar
    - Point count
    - Field descriptions
    - Point data
    """
    logger.info(f"Reading CloudCompare .bin file: {bin_path}")

    with open(bin_path, 'rb') as f:
        # Read magic number (4 bytes)
        magic = f.read(4).decode('latin-1', errors='ignore')
        logger.info(f"Magic number: {magic}")

        if magic not in ['CCB2', 'CCSF']:
            logger.warning(f"Unknown magic number: {magic} - attempting to parse anyway")

        # Skip header bytes - CloudCompare header is complex
        # Let's try to find the actual point data
        f.seek(0)
        all_data = f.read()

    # Strategy: The file likely contains XYZ + RGB + Classification
    # Common CloudCompare format: XYZ (3x float32=12) + RGB (3x uint8=3) + Scalar field (float32=4) = 19 bytes
    # Or: XYZ (3x float64=24) + RGB (3x uint8=3) + Classification (uint8=1) = 28 bytes

    file_size = len(all_data)
    logger.info(f"Total file size: {file_size:,} bytes")

    # Try to find repeating pattern by analyzing file
    # Most CloudCompare files have a header, then pure binary point data

    # Common point sizes to try
    point_sizes = [
        (28, 'XYZ(double) + RGB(uint8) + Class(uint8)'),
        (32, 'XYZ(double) + RGB(uint8) + Scalar(float32) + Class(uint8)'),
        (19, 'XYZ(float32) + RGB(uint8) + Scalar(float32)'),
        (25, 'XYZ(float32) + RGB(float32) + Class(uint8)'),
        (16, 'XYZ(float32) + RGB(uint8) + Class(uint8)'),
    ]

    # Try different header sizes
    for header_size in [0, 100, 200, 300, 400, 500, 1000, 2000]:
        remaining_size = file_size - header_size

        for point_size, description in point_sizes:
            if remaining_size % point_size == 0:
                num_points = remaining_size // point_size
                if num_points > 100 and num_points < 100000000:  # Reasonable point count
                    logger.info(f"Possible: Header={header_size}, PointSize={point_size}, Points={num_points:,} - {description}")

                    # Try to parse
                    try:
                        points = parse_points(all_data[header_size:], num_points, point_size, description)
                        if points is not None and len(points) > 0:
                            # Validate: check if XYZ values seem reasonable
                            xyz_vals = points[:, :3]
                            if np.all(np.isfinite(xyz_vals)) and np.std(xyz_vals) > 0.001:
                                logger.info(f"✓ Successfully parsed with header={header_size}, point_size={point_size}")
                                return points, description
                    except Exception as e:
                        continue

    raise ValueError("Could not parse CloudCompare .bin file")


def parse_points(data: bytes, num_points: int, point_size: int, description: str):
    """Parse binary point data based on description"""

    points_list = []

    for i in range(num_points):
        offset = i * point_size
        point_bytes = data[offset:offset + point_size]

        if len(point_bytes) < point_size:
            break

        try:
            if 'XYZ(double)' in description:
                if point_size == 28:  # XYZ(double) + RGB(uint8) + Class(uint8)
                    x, y, z = struct.unpack('ddd', point_bytes[:24])
                    r, g, b, c = struct.unpack('BBBB', point_bytes[24:28])
                    points_list.append([x, y, z, r/255.0, g/255.0, b/255.0, c])
                elif point_size == 32:  # XYZ(double) + RGB(uint8) + Scalar + Class
                    x, y, z = struct.unpack('ddd', point_bytes[:24])
                    r, g, b = struct.unpack('BBB', point_bytes[24:27])
                    scalar = struct.unpack('f', point_bytes[27:31])[0]
                    c = struct.unpack('B', point_bytes[31:32])[0]
                    points_list.append([x, y, z, r/255.0, g/255.0, b/255.0, scalar, c])

            elif 'XYZ(float32)' in description:
                if point_size == 16:  # XYZ(float32) + RGB(uint8) + Class(uint8)
                    x, y, z = struct.unpack('fff', point_bytes[:12])
                    r, g, b, c = struct.unpack('BBBB', point_bytes[12:16])
                    points_list.append([x, y, z, r/255.0, g/255.0, b/255.0, c])
                elif point_size == 19:  # XYZ(float32) + RGB(uint8) + Scalar(float32)
                    x, y, z = struct.unpack('fff', point_bytes[:12])
                    r, g, b = struct.unpack('BBB', point_bytes[12:15])
                    scalar = struct.unpack('f', point_bytes[15:19])[0]
                    points_list.append([x, y, z, r/255.0, g/255.0, b/255.0, scalar])
                elif point_size == 25:  # XYZ(float32) + RGB(float32) + Class(uint8)
                    vals = struct.unpack('fffffffB', point_bytes)
                    points_list.append(list(vals))

        except struct.error:
            continue

        # Sample every 1000th point for validation
        if i > 0 and i % 1000 == 0 and len(points_list) > 0:
            # Check if coordinates seem reasonable
            sample = np.array(points_list[-1])
            if not np.all(np.isfinite(sample[:3])):
                return None

    if len(points_list) == 0:
        return None

    return np.array(points_list)


def convert_cc_bin_to_las(bin_path: str, las_path: str):
    """Convert CloudCompare .bin to .las"""

    points_data, description = read_cloudcompare_bin(bin_path)

    logger.info(f"Parsed {len(points_data):,} points")
    logger.info(f"Format: {description}")
    logger.info(f"Data shape: {points_data.shape}")

    # Create LAS file
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = np.min(points_data[:, :3], axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])

    las = laspy.LasData(header)
    las.x = points_data[:, 0]
    las.y = points_data[:, 1]
    las.z = points_data[:, 2]

    # RGB
    if points_data.shape[1] >= 6:
        las.red = (points_data[:, 3] * 65535).astype(np.uint16)
        las.green = (points_data[:, 4] * 65535).astype(np.uint16)
        las.blue = (points_data[:, 5] * 65535).astype(np.uint16)

    # Classification
    if points_data.shape[1] >= 7:
        class_idx = 6 if points_data.shape[1] == 7 else 7
        las.classification = points_data[:, class_idx].astype(np.uint8)

        unique_classes = np.unique(points_data[:, class_idx])
        logger.info(f"Unique classification values: {unique_classes}")

        for cls in unique_classes:
            count = np.sum(points_data[:, class_idx] == cls)
            pct = (count / len(points_data)) * 100
            logger.info(f"  Class {int(cls)}: {count:,} points ({pct:.1f}%)")

    las.write(las_path)
    logger.info(f"Saved LAS file: {las_path}")

    return las_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python read_cloudcompare_bin.py <input.bin> [output.las]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.bin', '.las')

    convert_cc_bin_to_las(input_file, output_file)
