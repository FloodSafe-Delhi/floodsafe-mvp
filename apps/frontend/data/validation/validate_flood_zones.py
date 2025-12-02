"""
Validates DEM-based flood zones against historical flood data
Calculates overlap percentage and generates validation report
"""
import json
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import Point, shape
import numpy as np
from pathlib import Path

def load_historical_floods(json_path: str) -> gpd.GeoDataFrame:
    """Load historical flood points from JSON"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    gdf = gpd.GeoDataFrame.from_features(data['features'])
    gdf['date'] = pd.to_datetime([f['properties']['date'] for f in data['features']])
    return gdf

def load_flood_risk_zones(raster_path: str, city: str) -> gpd.GeoDataFrame:
    """
    Extract flood risk polygons from raster tiles
    Zones: 1=Low (Yellow), 2=Medium (Green), 3=High (Teal), 4=Very High (Blue)
    """
    with rasterio.open(raster_path) as src:
        image = src.read(1)
        mask = image != src.nodata if src.nodata else np.ones_like(image, dtype=bool)

        # Extract shapes for each risk level
        results = []
        for geom, value in shapes(image, mask=mask, transform=src.transform):
            if value in [1, 2, 3, 4]:  # Valid risk levels
                results.append({
                    'geometry': shape(geom),
                    'risk_level': int(value),
                    'risk_category': {
                        1: 'Low',
                        2: 'Medium',
                        3: 'High',
                        4: 'Very High'
                    }[int(value)]
                })

        gdf = gpd.GeoDataFrame(results, crs=src.crs)
        return gdf

def calculate_overlap(historical_floods: gpd.GeoDataFrame,
                     flood_zones: gpd.GeoDataFrame) -> dict:
    """
    Calculate what percentage of historical floods fall within high-risk zones
    Target: >70% in blue (4) or teal (3) zones
    """
    # Ensure same CRS
    if historical_floods.crs != flood_zones.crs:
        historical_floods = historical_floods.to_crs(flood_zones.crs)

    # Spatial join to find which zone each historical flood is in
    joined = gpd.sjoin(historical_floods, flood_zones, how='left', predicate='within')

    # Calculate statistics
    total_floods = len(historical_floods)
    floods_in_high_risk = len(joined[joined['risk_level'].isin([3, 4])])
    floods_in_very_high_risk = len(joined[joined['risk_level'] == 4])
    floods_in_medium_risk = len(joined[joined['risk_level'] == 2])
    floods_in_low_risk = len(joined[joined['risk_level'] == 1])
    floods_outside_zones = len(joined[joined['risk_level'].isna()])

    results = {
        'total_historical_floods': total_floods,
        'floods_in_very_high_risk_zones': floods_in_very_high_risk,
        'floods_in_high_risk_zones': floods_in_high_risk - floods_in_very_high_risk,
        'floods_in_medium_risk_zones': floods_in_medium_risk,
        'floods_in_low_risk_zones': floods_in_low_risk,
        'floods_outside_any_zone': floods_outside_zones,
        'percentage_in_high_or_very_high': float((floods_in_high_risk / total_floods * 100)) if total_floods > 0 else 0.0,
        'percentage_in_very_high_only': float((floods_in_very_high_risk / total_floods * 100)) if total_floods > 0 else 0.0,
        'validation_status': 'PASS' if total_floods > 0 and (floods_in_high_risk / total_floods * 100) >= 70 else 'FAIL',
        'confidence_score': float(min(100, (floods_in_high_risk / total_floods * 100) * 1.2)) if total_floods > 0 else 0.0  # Weighted score
    }

    return results, joined

def generate_validation_report(city: str,
                               dem_metrics: dict,
                               overlap_results: dict,
                               output_path: str):
    """Generate comprehensive validation report"""
    report = {
        'city': city,
        'validation_date': '2024-12-02',
        'dem_quality': dem_metrics,
        'historical_flood_validation': overlap_results,
        'overall_assessment': {
            'dem_grade': dem_metrics.get('quality_grade', 'N/A'),
            'validation_status': overlap_results['validation_status'],
            'confidence_score': overlap_results['confidence_score'],
            'recommendation': (
                'APPROVED: Flood zones validated against historical data'
                if overlap_results['validation_status'] == 'PASS' and
                   dem_metrics.get('quality_score', 0) >= 60
                else 'NEEDS IMPROVEMENT: Validation criteria not met'
            )
        }
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    return report

def main():
    print("="*70)
    print("FLOOD ZONE VALIDATION - HISTORICAL DATA OVERLAY")
    print("="*70)

    # Bangalore validation
    print("\n" + "-"*70)
    print("BANGALORE FLOOD ZONE VALIDATION")
    print("-"*70)

    bangalore_historical = Path("historical-floods-bangalore.json")
    if bangalore_historical.exists():
        historical_floods = load_historical_floods(str(bangalore_historical))
        print(f"[OK] Loaded {len(historical_floods)} historical flood incidents")

        # TODO: Load flood zones (need raster path)
        print("[INFO] Bangalore flood zone analysis not yet implemented")
        print("[INFO] Waiting for: ../bangalore/dem/tiles/stream_influence_reclass.tif")
    else:
        print("[INFO] Historical flood data not collected yet")
        print("[INFO] Need to create: historical-floods-bangalore.json")

    print("\n" + "-"*70)
    print("DELHI FLOOD ZONE VALIDATION")
    print("-"*70)

    delhi_historical = Path("historical-floods-delhi.json")
    if delhi_historical.exists():
        historical_floods = load_historical_floods(str(delhi_historical))
        print(f"[OK] Loaded {len(historical_floods)} historical flood incidents")

        # Load Delhi flood zones
        delhi_flood_raster = Path("../delhi/dem/tiles/stream_influence_reclass.tif")
        if delhi_flood_raster.exists():
            print(f"[OK] Loading flood risk zones from {delhi_flood_raster.name}")
            flood_zones = load_flood_risk_zones(str(delhi_flood_raster), "Delhi")
            print(f"[OK] Extracted {len(flood_zones)} flood risk zones")

            # Calculate overlap
            overlap_results, joined = calculate_overlap(historical_floods, flood_zones)

            print("\n" + "="*70)
            print("VALIDATION RESULTS - DELHI")
            print("="*70)
            print(f"Total Historical Floods: {overlap_results['total_historical_floods']}")
            print(f"  In Very High Risk (Blue) Zones: {overlap_results['floods_in_very_high_risk_zones']}")
            print(f"  In High Risk (Teal) Zones: {overlap_results['floods_in_high_risk_zones']}")
            print(f"  In Medium Risk (Green) Zones: {overlap_results['floods_in_medium_risk_zones']}")
            print(f"  In Low Risk (Yellow) Zones: {overlap_results['floods_in_low_risk_zones']}")
            print(f"  Outside Any Zone: {overlap_results['floods_outside_any_zone']}")
            print(f"\nHigh/Very High Risk Overlap: {overlap_results['percentage_in_high_or_very_high']:.1f}%")
            print(f"Status: {overlap_results['validation_status']} (target: >=70%)")
            print(f"Confidence Score: {overlap_results['confidence_score']:.1f}/100")

            # Load DEM metrics
            dem_report_path = Path("dem-quality-report.json")
            if dem_report_path.exists():
                with open(dem_report_path, 'r') as f:
                    dem_report = json.load(f)

                if 'delhi' in dem_report:
                    report = generate_validation_report(
                        "Delhi",
                        dem_report['delhi'],
                        overlap_results,
                        "delhi-validation-report.json"
                    )
                    print(f"\n[SUCCESS] Validation report saved to: delhi-validation-report.json")
                    print(f"Overall Recommendation: {report['overall_assessment']['recommendation']}")
        else:
            print(f"[WARNING] Flood raster not found at: {delhi_flood_raster}")
            print("[INFO] Run DEM processing pipeline first to generate flood risk zones")
    else:
        print("[INFO] Historical flood data not collected yet")
        print("[INFO] Need to create: historical-floods-delhi.json")
        print("[INFO] Expected format: GeoJSON FeatureCollection with Point geometries")
        print("[INFO] Each feature should have properties: location, date, severity, source")

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Collect historical flood data:")
    print("   - Research 50-100 flood incidents from news, reports, social media")
    print("   - Create historical-floods-bangalore.json")
    print("   - Create historical-floods-delhi.json")
    print("2. Re-run this script to calculate validation metrics")
    print("3. Review validation reports and adjust if <70% overlap")

if __name__ == "__main__":
    main()
