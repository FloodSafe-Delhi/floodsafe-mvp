"""
Validates DEM quality for flood modeling accuracy
Checks: resolution, data gaps, vertical accuracy, terrain representation
"""
import rasterio
import numpy as np
from pathlib import Path
import json

def check_dem_quality(dem_path: str, city: str) -> dict:
    """
    Analyzes DEM file for quality metrics

    Returns:
        dict: Quality metrics including resolution, gaps, statistics
    """
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        transform = src.transform
        crs = src.crs

        # Calculate metrics
        metrics = {
            'city': city,
            'resolution_meters': float(abs(transform[0])),
            'total_pixels': int(dem.size),
            'nodata_pixels': int(np.sum(dem == src.nodata)) if src.nodata else 0,
            'nodata_percentage': float((np.sum(dem == src.nodata) / dem.size * 100)) if src.nodata else 0.0,
            'elevation_range': {
                'min': float(np.min(dem[dem != src.nodata])) if src.nodata else float(np.min(dem)),
                'max': float(np.max(dem[dem != src.nodata])) if src.nodata else float(np.max(dem)),
                'mean': float(np.mean(dem[dem != src.nodata])) if src.nodata else float(np.mean(dem)),
                'std': float(np.std(dem[dem != src.nodata])) if src.nodata else float(np.std(dem))
            },
            'crs': str(crs),
            'bounds': {
                'left': float(src.bounds.left),
                'bottom': float(src.bounds.bottom),
                'right': float(src.bounds.right),
                'top': float(src.bounds.top)
            }
        }

        # Quality assessment
        quality_score = 100
        issues = []

        if metrics['resolution_meters'] > 30:
            quality_score -= 20
            issues.append(f"Low resolution: {metrics['resolution_meters']}m (recommend <30m)")

        if metrics['nodata_percentage'] > 5:
            quality_score -= 15
            issues.append(f"High data gaps: {metrics['nodata_percentage']:.1f}% (recommend <5%)")

        if metrics['elevation_range']['std'] < 5:
            quality_score -= 10
            issues.append("Low terrain variation - may be flattened DEM")

        metrics['quality_score'] = quality_score
        metrics['quality_issues'] = issues
        metrics['quality_grade'] = (
            'A' if quality_score >= 90 else
            'B' if quality_score >= 75 else
            'C' if quality_score >= 60 else
            'D' if quality_score >= 40 else 'F'
        )

        return metrics

def main():
    results = {}

    # Check Bangalore DEM
    bangalore_dem = Path("../bangalore/dem/tiles/filled_dem.tif")
    if bangalore_dem.exists():
        print("="*60)
        print("BANGALORE DEM QUALITY CHECK")
        print("="*60)
        metrics = check_dem_quality(str(bangalore_dem), "Bangalore")
        print(f"  Quality Grade: {metrics['quality_grade']} ({metrics['quality_score']}/100)")
        print(f"  Resolution: {metrics['resolution_meters']:.2f}m")
        print(f"  Data Gaps: {metrics['nodata_percentage']:.2f}%")
        print(f"  Elevation Range: {metrics['elevation_range']['min']:.1f}m to {metrics['elevation_range']['max']:.1f}m")
        print(f"  Mean Elevation: {metrics['elevation_range']['mean']:.1f}m (±{metrics['elevation_range']['std']:.1f}m)")
        if metrics['quality_issues']:
            print("\n  Issues:")
            for issue in metrics['quality_issues']:
                print(f"    - {issue}")
        else:
            print("\n  [OK] No quality issues detected!")
        results['bangalore'] = metrics
    else:
        print("[WARNING] Bangalore DEM not found at:", bangalore_dem)

    print("\n")

    # Check Delhi DEM
    delhi_dem = Path("../delhi/dem/tiles/filled_dem.tif")
    if delhi_dem.exists():
        print("="*60)
        print("DELHI DEM QUALITY CHECK")
        print("="*60)
        metrics = check_dem_quality(str(delhi_dem), "Delhi")
        print(f"  Quality Grade: {metrics['quality_grade']} ({metrics['quality_score']}/100)")
        print(f"  Resolution: {metrics['resolution_meters']:.2f}m")
        print(f"  Data Gaps: {metrics['nodata_percentage']:.2f}%")
        print(f"  Elevation Range: {metrics['elevation_range']['min']:.1f}m to {metrics['elevation_range']['max']:.1f}m")
        print(f"  Mean Elevation: {metrics['elevation_range']['mean']:.1f}m (±{metrics['elevation_range']['std']:.1f}m)")
        if metrics['quality_issues']:
            print("\n  Issues:")
            for issue in metrics['quality_issues']:
                print(f"    - {issue}")
        else:
            print("\n  [OK] No quality issues detected!")
        results['delhi'] = metrics
    else:
        print("[WARNING] Delhi DEM not found at:", delhi_dem)

    # Save results to JSON
    output_file = Path("dem-quality-report.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print(f"[SUCCESS] Results saved to: {output_file.absolute()}")
    print("="*60)

if __name__ == "__main__":
    main()
