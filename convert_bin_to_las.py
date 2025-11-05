"""
Convert .bin point cloud files to .las format
Supports various .bin formats from point cloud software
"""

import numpy as np
import laspy
import struct
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_bin_file(bin_path: str):
    """
    Read binary point cloud file

    Common formats:
    - XYZIRGB (XYZ + Intensity + RGB)
    - XYZRGB (XYZ + RGB)
    - XYZIC (XYZ + Intensity + Classification)
    - XYZRGBC (XYZ + RGB + Classification)
    """
    logger.info(f"Reading binary file: {bin_path}")

    with open(bin_path, 'rb') as f:
        data = f.read()

    file_size = len(data)
    logger.info(f"File size: {file_size:,} bytes")

    # Try different format interpretations
    formats = {
        # Format: (struct_format, point_size, description)
        'xyzrgbc_float32': ('ffffffB', 25, 'XYZ(float32) + RGB(float32) + Class(uint8)'),
        'xyzrgbc_uint8': ('fffBBBB', 16, 'XYZ(float32) + RGB(uint8) + Class(uint8)'),
        'xyzirgbc': ('ffffBBBB', 19, 'XYZI(float32) + RGB(uint8) + Class(uint8)'),
        'xyzic': ('ffffB', 17, 'XYZI(float32) + Class(uint8)'),
        'xyzrgb': ('fffBBB', 15, 'XYZ(float32) + RGB(uint8)'),
        'xyz_double': ('ddd', 24, 'XYZ(double)'),
        'xyz_float': ('fff', 12, 'XYZ(float32)'),
    }

    # Auto-detect format
    detected_format = None
    for format_name, (fmt, point_size, desc) in formats.items():
        if file_size % point_size == 0:
            num_points = file_size // point_size
            logger.info(f"Possible format: {format_name} - {desc} ({num_points:,} points)")
            if detected_format is None:
                detected_format = (format_name, fmt, point_size, desc, num_points)

    if detected_format is None:
        # Calculate possible point sizes
        logger.error("Could not detect format. File size doesn't match known formats.")
        logger.info("Trying to find divisors...")
        for size in range(12, 50):
            if file_size % size == 0:
                logger.info(f"  Possible point size: {size} bytes ({file_size // size:,} points)")
        raise ValueError("Unknown binary format")

    format_name, struct_fmt, point_size, desc, num_points = detected_format
    logger.info(f"Using format: {format_name} - {desc}")
    logger.info(f"Number of points: {num_points:,}")

    # Parse binary data
    points_data = []

    for i in range(num_points):
        offset = i * point_size
        point_bytes = data[offset:offset + point_size]

        try:
            if format_name in ['xyzrgbc_float32']:
                x, y, z, r, g, b, c = struct.unpack(struct_fmt, point_bytes)
                points_data.append([x, y, z, r, g, b, c])
            elif format_name in ['xyzrgbc_uint8']:
                x, y, z, r, g, b, c = struct.unpack(struct_fmt, point_bytes)
                points_data.append([x, y, z, r/255.0, g/255.0, b/255.0, c])
            elif format_name == 'xyzirgbc':
                x, y, z, i, r, g, b, c = struct.unpack(struct_fmt, point_bytes)
                points_data.append([x, y, z, r/255.0, g/255.0, b/255.0, i, c])
            elif format_name == 'xyzic':
                x, y, z, i, c = struct.unpack(struct_fmt, point_bytes)
                points_data.append([x, y, z, i, c])
            elif format_name == 'xyzrgb':
                x, y, z, r, g, b = struct.unpack(struct_fmt, point_bytes)
                points_data.append([x, y, z, r/255.0, g/255.0, b/255.0])
            elif format_name in ['xyz_double', 'xyz_float']:
                x, y, z = struct.unpack(struct_fmt, point_bytes)
                points_data.append([x, y, z])
        except struct.error:
            continue

    logger.info(f"Successfully parsed {len(points_data):,} points")

    return np.array(points_data), format_name


def convert_bin_to_las(bin_path: str, las_path: str):
    """Convert .bin file to .las format"""

    # Read binary file
    points_data, format_name = read_bin_file(bin_path)

    # Create LAS file
    logger.info(f"Creating LAS file: {las_path}")

    # Create header
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = np.min(points_data[:, :3], axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])

    # Create LAS data
    las = laspy.LasData(header)

    # Set XYZ
    las.x = points_data[:, 0]
    las.y = points_data[:, 1]
    las.z = points_data[:, 2]

    # Set RGB if available
    if points_data.shape[1] >= 6:
        las.red = (points_data[:, 3] * 65535).astype(np.uint16)
        las.green = (points_data[:, 4] * 65535).astype(np.uint16)
        las.blue = (points_data[:, 5] * 65535).astype(np.uint16)
        logger.info("RGB data included")

    # Set classification if available
    if format_name in ['xyzrgbc_float32', 'xyzrgbc_uint8']:
        las.classification = points_data[:, 6].astype(np.uint8)
        logger.info("Classification data included")
        unique_classes = np.unique(points_data[:, 6])
        logger.info(f"Unique classes: {unique_classes}")
    elif format_name == 'xyzirgbc':
        las.classification = points_data[:, 7].astype(np.uint8)
        if points_data.shape[1] > 6:
            las.intensity = (points_data[:, 6] * 65535).astype(np.uint16)
        logger.info("Classification and intensity data included")
        unique_classes = np.unique(points_data[:, 7])
        logger.info(f"Unique classes: {unique_classes}")
    elif format_name == 'xyzic':
        las.intensity = (points_data[:, 3] * 65535).astype(np.uint16)
        las.classification = points_data[:, 4].astype(np.uint8)
        logger.info("Intensity and classification data included")
        unique_classes = np.unique(points_data[:, 4])
        logger.info(f"Unique classes: {unique_classes}")

    # Write LAS file
    las.write(las_path)
    logger.info(f"Successfully wrote LAS file with {len(las.points):,} points")

    return las_path


def main():
    parser = argparse.ArgumentParser(description='Convert .bin point cloud to .las format')
    parser.add_argument('--input', type=str, required=True, help='Input .bin file path')
    parser.add_argument('--output', type=str, help='Output .las file path (optional)')

    args = parser.parse_args()

    input_path = Path(args.input)

    if args.output:
        output_path = args.output
    else:
        output_path = input_path.with_suffix('.las')

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    try:
        convert_bin_to_las(str(input_path), str(output_path))
        logger.info(f"\nConversion complete!")
        logger.info(f"Input:  {input_path}")
        logger.info(f"Output: {output_path}")
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
