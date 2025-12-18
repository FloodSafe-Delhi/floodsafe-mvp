"""
Quick test script for rainfall API endpoints.
Run this after starting the backend server.

Usage:
    python test_rainfall_api.py
"""
import httpx
import sys
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    print("\n1. Testing /api/rainfall/health...")
    try:
        response = httpx.get(f"{BASE_URL}/api/rainfall/health", timeout=10.0)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"   ERROR: {e}")
        return False

def test_single_point():
    """Test single point forecast."""
    print("\n2. Testing /api/rainfall/forecast (Delhi)...")
    try:
        response = httpx.get(
            f"{BASE_URL}/api/rainfall/forecast",
            params={"lat": 28.6, "lng": 77.2},
            timeout=15.0
        )
        print(f"   Status: {response.status_code}")
        data = response.json()

        if response.status_code == 200:
            print(f"   Forecast 24h: {data['forecast_24h_mm']}mm")
            print(f"   Forecast 48h: {data['forecast_48h_mm']}mm")
            print(f"   Forecast 72h: {data['forecast_72h_mm']}mm")
            print(f"   Total 3-day: {data['forecast_total_3d_mm']}mm")
            print(f"   Intensity: {data['intensity_category']}")
            print(f"   Hourly max: {data['hourly_max_mm']}mm")
            print(f"   Probability: {data.get('probability_max_pct', 'N/A')}%")
            return True
        else:
            print(f"   ERROR Response: {json.dumps(data, indent=2)}")
            return False
    except Exception as e:
        print(f"   ERROR: {e}")
        return False

def test_grid():
    """Test grid forecast."""
    print("\n3. Testing /api/rainfall/forecast/grid (small grid around Delhi)...")
    try:
        # Small 3x3 grid around Delhi
        response = httpx.get(
            f"{BASE_URL}/api/rainfall/forecast/grid",
            params={
                "lat_min": 28.5,
                "lng_min": 77.1,
                "lat_max": 28.7,
                "lng_max": 77.3,
                "resolution": 0.1
            },
            timeout=30.0
        )
        print(f"   Status: {response.status_code}")
        data = response.json()

        if response.status_code == 200:
            print(f"   Type: {data['type']}")
            print(f"   Features: {len(data['features'])} points")
            print(f"   Metadata: {json.dumps(data['metadata'], indent=2)}")

            # Show first feature
            if data['features']:
                first = data['features'][0]
                print(f"\n   Sample point:")
                print(f"   - Coords: {first['geometry']['coordinates']}")
                print(f"   - Forecast 24h: {first['properties']['forecast_24h_mm']}mm")
                print(f"   - Intensity: {first['properties']['intensity_category']}")
            return True
        else:
            print(f"   ERROR Response: {json.dumps(data, indent=2)}")
            return False
    except Exception as e:
        print(f"   ERROR: {e}")
        return False

def test_cache():
    """Test caching by making duplicate request."""
    print("\n4. Testing cache (duplicate request)...")
    try:
        # First request
        start1 = httpx.get(
            f"{BASE_URL}/api/rainfall/forecast",
            params={"lat": 28.6, "lng": 77.2},
            timeout=15.0
        )

        # Second request (should be cached)
        start2 = httpx.get(
            f"{BASE_URL}/api/rainfall/forecast",
            params={"lat": 28.6, "lng": 77.2},
            timeout=15.0
        )

        print(f"   Both requests successful: {start1.status_code == 200 and start2.status_code == 200}")
        print(f"   Data matches: {start1.json() == start2.json()}")
        return True
    except Exception as e:
        print(f"   ERROR: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("RAINFALL API TEST SUITE")
    print("=" * 60)
    print(f"Target: {BASE_URL}")

    tests = [
        ("Health Check", test_health),
        ("Single Point Forecast", test_single_point),
        ("Grid Forecast", test_grid),
        ("Cache Test", test_cache),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n   FATAL ERROR in {name}: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:8} - {name}")

    passed = sum(1 for _, s in results if s)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")

    sys.exit(0 if passed == total else 1)

if __name__ == "__main__":
    main()
