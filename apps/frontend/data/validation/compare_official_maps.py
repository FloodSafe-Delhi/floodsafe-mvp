"""
Compare our flood zones with official DDA/BDA flood hazard maps
"""
import geopandas as gpd
from pathlib import Path
import json

def compare_with_official_maps(city: str, our_zones_path: str, official_map_path: str):
    """
    Load official flood hazard maps and compare with our zones

    Official sources:
    - Delhi: DDA Master Plan 2041 flood hazard zones
    - Bangalore: BDA Development Plan flood-prone areas
    """
    print(f"\n{city} Official Map Comparison:")
    print("-" * 60)

    # Check if our zones exist
    our_zones = Path(our_zones_path)
    if not our_zones.exists():
        print(f"[WARNING] Our flood zones not found at: {our_zones}")
        return None

    # Check if official map exists
    official_map = Path(official_map_path)
    if not official_map.exists():
        print(f"[INFO] Official map not found at: {official_map}")
        print(f"[INFO] Download official flood hazard map and place at this location")
        return None

    # Load both datasets
    print(f"[OK] Loading our flood zones from: {our_zones.name}")
    our_gdf = gpd.read_file(str(our_zones))

    print(f"[OK] Loading official map from: {official_map.name}")
    official_gdf = gpd.read_file(str(official_map))

    # Ensure same CRS
    if our_gdf.crs != official_gdf.crs:
        print(f"[INFO] Reprojecting to common CRS: {official_gdf.crs}")
        our_gdf = our_gdf.to_crs(official_gdf.crs)

    # Calculate spatial overlap
    print("\n[INFO] Calculating spatial overlap...")

    # Method 1: Intersection area
    intersection = gpd.overlay(our_gdf, official_gdf, how='intersection')
    intersection_area = intersection.geometry.area.sum()
    our_total_area = our_gdf.geometry.area.sum()
    official_total_area = official_gdf.geometry.area.sum()

    overlap_percentage = (intersection_area / our_total_area * 100) if our_total_area > 0 else 0
    official_coverage = (intersection_area / official_total_area * 100) if official_total_area > 0 else 0

    # Method 2: Areas we mark but official doesn't (potential false positives)
    our_only = gpd.overlay(our_gdf, official_gdf, how='difference')
    our_only_area = our_only.geometry.area.sum()
    false_positive_percentage = (our_only_area / our_total_area * 100) if our_total_area > 0 else 0

    # Method 3: Areas official marks but we don't (potential false negatives)
    official_only = gpd.overlay(official_gdf, our_gdf, how='difference')
    official_only_area = official_only.geometry.area.sum()
    false_negative_percentage = (official_only_area / official_total_area * 100) if official_total_area > 0 else 0

    results = {
        'city': city,
        'our_total_area_sq_km': float(our_total_area / 1_000_000),  # Convert to sq km
        'official_total_area_sq_km': float(official_total_area / 1_000_000),
        'intersection_area_sq_km': float(intersection_area / 1_000_000),
        'overlap_with_official_percentage': float(overlap_percentage),
        'coverage_of_official_percentage': float(official_coverage),
        'false_positive_percentage': float(false_positive_percentage),
        'false_negative_percentage': float(false_negative_percentage),
        'agreement_score': float(100 - abs(overlap_percentage - 100))
    }

    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"Our flood zones area: {results['our_total_area_sq_km']:.2f} sq km")
    print(f"Official flood zones area: {results['official_total_area_sq_km']:.2f} sq km")
    print(f"Intersection area: {results['intersection_area_sq_km']:.2f} sq km")
    print(f"\nOverlap with official map: {results['overlap_with_official_percentage']:.1f}%")
    print(f"Coverage of official zones: {results['coverage_of_official_percentage']:.1f}%")
    print(f"False positives (we mark, official doesn't): {results['false_positive_percentage']:.1f}%")
    print(f"False negatives (official marks, we don't): {results['false_negative_percentage']:.1f}%")
    print(f"Agreement score: {results['agreement_score']:.1f}/100")

    # Save results
    output_file = f"{city.lower()}-official-comparison.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[SUCCESS] Comparison results saved to: {output_file}")

    return results

def main():
    print("="*70)
    print("OFFICIAL FLOOD MAP COMPARISON")
    print("="*70)

    official_map_sources = {
        'Delhi': [
            'DDA Master Plan 2041 - Yamuna Floodplain zones',
            'Central Water Commission flood inundation maps',
            'IMD flood hazard assessment'
        ],
        'Bangalore': [
            'BDA Revised Master Plan 2015 - Flood prone areas',
            'KSNDMC flood vulnerability zones',
            'Lakes and Rajakaluves (storm water drains) buffer zones'
        ]
    }

    print("\nOfficial Sources to Compare Against:")
    for city, sources in official_map_sources.items():
        print(f"\n{city}:")
        for source in sources:
            print(f"  - {source}")

    print("\n" + "="*70)
    print("COMPARISON ANALYSIS")
    print("="*70)

    # Delhi comparison
    delhi_results = compare_with_official_maps(
        "Delhi",
        "../delhi/dem/tiles/stream_influence_reclass.geojson",  # Need to convert TIF to GeoJSON first
        "official-maps/delhi-dda-flood-zones.geojson"
    )

    # Bangalore comparison
    bangalore_results = compare_with_official_maps(
        "Bangalore",
        "../bangalore/dem/tiles/stream_influence_reclass.geojson",
        "official-maps/bangalore-bda-flood-zones.geojson"
    )

    print("\n" + "="*70)
    print("ACTION REQUIRED")
    print("="*70)
    print("To complete official map comparison:")
    print("\n1. Download official flood hazard maps from:")
    print("   Delhi:")
    print("     - Visit: https://dda.gov.in/")
    print("     - Search for: 'Master Plan 2041 Flood Zones' or 'Yamuna Floodplain'")
    print("   Bangalore:")
    print("     - Visit: https://bbmp.gov.in/ or https://bda.gov.in/")
    print("     - Search for: 'Flood Prone Areas' or 'Storm Water Drain Maps'")
    print("\n2. Convert official maps to GeoJSON format:")
    print("   - If PDF: Georeference and digitize using QGIS")
    print("   - If Shapefile: Use ogr2ogr to convert to GeoJSON")
    print("   - If KML/KMZ: Convert using GDAL/OGR tools")
    print("\n3. Place converted GeoJSON files in:")
    print("   - validation/official-maps/delhi-dda-flood-zones.geojson")
    print("   - validation/official-maps/bangalore-bda-flood-zones.geojson")
    print("\n4. Convert our raster flood zones to GeoJSON:")
    print("   Example using gdal_polygonize.py:")
    print("   gdal_polygonize.py ../delhi/dem/tiles/stream_influence_reclass.tif \\")
    print("     ../delhi/dem/tiles/stream_influence_reclass.geojson")
    print("\n5. Re-run this script to perform the comparison")

    print("\n" + "="*70)
    print("INTERPRETATION GUIDE")
    print("="*70)
    print("Overlap with official map:")
    print("  >80%: Excellent agreement - high confidence")
    print("  60-80%: Good agreement - moderate confidence")
    print("  40-60%: Fair agreement - needs investigation")
    print("  <40%: Poor agreement - major discrepancies")
    print("\nFalse positives (we mark, official doesn't):")
    print("  - Could indicate our model is too conservative")
    print("  - Or official maps are outdated")
    print("  - Investigate specific areas")
    print("\nFalse negatives (official marks, we don't):")
    print("  - Could indicate missing flood-prone areas")
    print("  - Or our DEM resolution insufficient")
    print("  - Critical to investigate for safety")

if __name__ == "__main__":
    main()
