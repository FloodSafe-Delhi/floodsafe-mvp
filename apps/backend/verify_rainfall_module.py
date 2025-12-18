"""
Verify rainfall module is correctly implemented.
This script checks that the module compiles and has all required components.
"""
import sys
import importlib.util

def verify_rainfall_module():
    """Verify the rainfall module structure."""
    print("=" * 60)
    print("RAINFALL MODULE VERIFICATION")
    print("=" * 60)

    checks = []

    # 1. Check module imports
    print("\n1. Checking module imports...")
    try:
        from src.api import rainfall
        print("   SUCCESS: Module imports correctly")
        checks.append(("Import", True))
    except Exception as e:
        print(f"   FAIL: {e}")
        checks.append(("Import", False))
        return checks

    # 2. Check router exists
    print("\n2. Checking router...")
    try:
        assert hasattr(rainfall, 'router')
        print("   SUCCESS: Router defined")
        checks.append(("Router", True))
    except Exception as e:
        print(f"   FAIL: {e}")
        checks.append(("Router", False))

    # 3. Check response models
    print("\n3. Checking response models...")
    try:
        assert hasattr(rainfall, 'RainfallForecastResponse')
        assert hasattr(rainfall, 'RainfallGridResponse')
        print("   SUCCESS: Response models defined")
        checks.append(("Models", True))
    except Exception as e:
        print(f"   FAIL: {e}")
        checks.append(("Models", False))

    # 4. Check IMD classification function
    print("\n4. Checking IMD classification...")
    try:
        assert hasattr(rainfall, '_classify_intensity')

        # Test classifications
        assert rainfall._classify_intensity(5.0) == "light"
        assert rainfall._classify_intensity(20.0) == "moderate"
        assert rainfall._classify_intensity(50.0) == "heavy"
        assert rainfall._classify_intensity(100.0) == "very_heavy"
        assert rainfall._classify_intensity(150.0) == "extremely_heavy"

        print("   SUCCESS: IMD classification working correctly")
        checks.append(("IMD Classification", True))
    except Exception as e:
        print(f"   FAIL: {e}")
        checks.append(("IMD Classification", False))

    # 5. Check cache functions
    print("\n5. Checking cache functions...")
    try:
        assert hasattr(rainfall, '_get_cache_key')
        assert hasattr(rainfall, '_is_cache_valid')
        assert hasattr(rainfall, '_cleanup_cache')
        print("   SUCCESS: Cache functions defined")
        checks.append(("Cache Functions", True))
    except Exception as e:
        print(f"   FAIL: {e}")
        checks.append(("Cache Functions", False))

    # 6. Check helper functions
    print("\n6. Checking helper functions...")
    try:
        assert hasattr(rainfall, '_fetch_open_meteo_forecast')
        assert hasattr(rainfall, '_process_forecast_data')
        print("   SUCCESS: Helper functions defined")
        checks.append(("Helper Functions", True))
    except Exception as e:
        print(f"   FAIL: {e}")
        checks.append(("Helper Functions", False))

    # 7. Check endpoints
    print("\n7. Checking endpoint routes...")
    try:
        routes = [route.path for route in rainfall.router.routes]

        assert any('/forecast' in r and '/grid' not in r for r in routes), "Missing /forecast endpoint"
        assert any('/forecast/grid' in r for r in routes), "Missing /forecast/grid endpoint"
        assert any('/health' in r for r in routes), "Missing /health endpoint"

        print(f"   SUCCESS: All endpoints defined")
        print(f"   Routes: {routes}")
        checks.append(("Endpoints", True))
    except Exception as e:
        print(f"   FAIL: {e}")
        checks.append(("Endpoints", False))

    return checks

def main():
    """Run verification."""
    checks = verify_rainfall_module()

    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"{status:8} - {name}")

    passed = sum(1 for _, p in checks if p)
    total = len(checks)
    print(f"\nTotal: {passed}/{total} checks passed")

    if passed == total:
        print("\nSUCCESS: Rainfall module is correctly implemented!")
        sys.exit(0)
    else:
        print("\nFAILURE: Some checks failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
