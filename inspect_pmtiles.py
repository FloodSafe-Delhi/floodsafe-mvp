#!/usr/bin/env python3
"""
Inspect and compare PMTiles files for Bangalore and Delhi flood data
"""
import struct
import json

def read_pmtiles_header(filepath):
    """Read PMTiles v3 header"""
    with open(filepath, 'rb') as f:
        # PMTiles v3 header is 127 bytes
        magic = f.read(7)
        if magic != b'PMTiles':
            return None

        version = struct.unpack('<B', f.read(1))[0]

        # Read header fields
        root_dir_offset = struct.unpack('<Q', f.read(8))[0]
        root_dir_length = struct.unpack('<Q', f.read(8))[0]
        metadata_offset = struct.unpack('<Q', f.read(8))[0]
        metadata_length = struct.unpack('<Q', f.read(8))[0]
        leaf_dirs_offset = struct.unpack('<Q', f.read(8))[0]
        leaf_dirs_length = struct.unpack('<Q', f.read(8))[0]
        tile_data_offset = struct.unpack('<Q', f.read(8))[0]
        tile_data_length = struct.unpack('<Q', f.read(8))[0]
        addressed_tiles = struct.unpack('<Q', f.read(8))[0]
        tile_entries = struct.unpack('<Q', f.read(8))[0]
        tile_contents_entries = struct.unpack('<Q', f.read(8))[0]

        # Skip to metadata if present
        metadata_json = None
        if metadata_offset > 0 and metadata_length > 0:
            f.seek(metadata_offset)
            metadata_bytes = f.read(metadata_length)
            try:
                # Try to decompress if needed (gzip)
                import gzip
                metadata_json = json.loads(gzip.decompress(metadata_bytes))
            except:
                try:
                    metadata_json = json.loads(metadata_bytes)
                except:
                    pass

        return {
            'version': version,
            'root_dir_offset': root_dir_offset,
            'root_dir_length': root_dir_length,
            'metadata_offset': metadata_offset,
            'metadata_length': metadata_length,
            'tile_data_offset': tile_data_offset,
            'tile_data_length': tile_data_length,
            'addressed_tiles': addressed_tiles,
            'tile_entries': tile_entries,
            'tile_contents_entries': tile_contents_entries,
            'metadata': metadata_json
        }

def format_bytes(bytes_val):
    """Format bytes to human readable"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"

def inspect_file(filepath, name):
    """Inspect a PMTiles file"""
    import os

    print(f"\n{'='*60}")
    print(f" {name} FLOOD TILES")
    print(f"{'='*60}")

    file_size = os.path.getsize(filepath)
    print(f"File Size: {format_bytes(file_size)}")

    header = read_pmtiles_header(filepath)
    if not header:
        print("ERROR: Not a valid PMTiles file")
        return

    print(f"\nPMTiles Version: {header['version']}")
    print(f"Tile Entries: {header['tile_entries']:,}")
    print(f"Addressed Tiles: {header['addressed_tiles']:,}")
    print(f"Tile Contents Entries: {header['tile_contents_entries']:,}")
    print(f"\nData Segments:")
    print(f"  Root Directory: {format_bytes(header['root_dir_length'])}")
    print(f"  Metadata: {format_bytes(header['metadata_length'])}")
    print(f"  Tile Data: {format_bytes(header['tile_data_length'])}")

    if header['metadata']:
        print(f"\nMetadata:")
        for key, value in header['metadata'].items():
            if key in ['minzoom', 'maxzoom', 'center', 'bounds', 'type', 'format']:
                print(f"  {key}: {value}")

    return header

if __name__ == '__main__':
    blr_header = inspect_file('/home/user/floodsafe-mvp/apps/frontend/public/tiles.pmtiles', 'BANGALORE')
    del_header = inspect_file('/home/user/floodsafe-mvp/apps/frontend/public/delhi-tiles.pmtiles', 'DELHI')

    print(f"\n{'='*60}")
    print(" COMPARISON SUMMARY")
    print(f"{'='*60}")

    if blr_header and del_header:
        import os
        blr_size = os.path.getsize('/home/user/floodsafe-mvp/apps/frontend/public/tiles.pmtiles')
        del_size = os.path.getsize('/home/user/floodsafe-mvp/apps/frontend/public/delhi-tiles.pmtiles')

        print(f"\nFile Size:")
        print(f"  Bangalore: {format_bytes(blr_size)}")
        print(f"  Delhi: {format_bytes(del_size)}")
        print(f"  Ratio: {blr_size/del_size:.1f}x larger (Bangalore)")

        print(f"\nTile Entries:")
        print(f"  Bangalore: {blr_header['tile_entries']:,}")
        print(f"  Delhi: {del_header['tile_entries']:,}")
        ratio = blr_header['tile_entries'] / max(del_header['tile_entries'], 1)
        print(f"  Ratio: {ratio:.1f}x more tiles (Bangalore)")

        print(f"\nData Complexity:")
        blr_avg = blr_size / max(blr_header['tile_entries'], 1)
        del_avg = del_size / max(del_header['tile_entries'], 1)
        print(f"  Bangalore: {format_bytes(blr_avg)} per tile")
        print(f"  Delhi: {format_bytes(del_avg)} per tile")
