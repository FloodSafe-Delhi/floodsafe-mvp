# FloodSafe Flood Zone Validation Framework

## Overview

This directory contains validation scripts to verify the accuracy and legitimacy of DEM-based flood risk zones for Bangalore and Delhi.

## Validation Scripts

### 1. DEM Quality Checker (`check_dem_quality.py`)

Validates the quality of Digital Elevation Model (DEM) data used for flood modeling.

**Checks:**
- Resolution (target: <30m)
- Data gaps (target: <5%)
- Terrain variation
- Elevation statistics

**Usage:**
```bash
python check_dem_quality.py
```

**Output:**
- Console report with quality grades (A-F)
- `dem-quality-report.json` with detailed metrics

**Current Results:**
- **Delhi**: Grade A (100/100) - Resolution ~31m, 0% gaps, 191-306m elevation

### 2. Historical Flood Overlay Validator (`validate_flood_zones.py`)

Validates flood risk zones against 50-100 historical flood incidents from news, reports, and government data.

**Target:** >70% of historical floods should fall within high-risk (blue/teal) zones

**Usage:**
```bash
python validate_flood_zones.py
```

**Prerequisites:**
- Create `historical-floods-bangalore.json` (50-100 verified incidents)
- Create `historical-floods-delhi.json` (50-100 verified incidents)

**Expected Format:**
```json
{
  "type": "FeatureCollection",
  "metadata": {
    "city": "Delhi",
    "total_incidents": 75,
    "date_range": "2015-01-01 to 2024-12-01",
    "sources": ["The Hindu", "DDA Reports", "IIT Delhi Studies"]
  },
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [77.2090, 28.6139]
      },
      "properties": {
        "location": "Yamuna floodplain near ITO",
        "date": "2023-07-13",
        "severity": "very_high",
        "water_depth_cm": 150,
        "source": "The Hindu",
        "source_url": "https://...",
        "verified": true
      }
    }
  ]
}
```

### 3. Official Map Comparison (`compare_official_maps.py`)

Compares our DEM-based flood zones with official government flood hazard maps.

**Sources:**
- **Delhi**: DDA Master Plan 2041 Yamuna Floodplain zones, CWC flood maps
- **Bangalore**: BDA Revised Master Plan 2015 flood-prone areas, KSNDMC zones

**Usage:**
```bash
python compare_official_maps.py
```

**Prerequisites:**
- Download official flood maps from DDA/BDA websites
- Convert to GeoJSON format
- Place in `official-maps/` directory

**Interpretation:**
- >80% overlap: Excellent agreement (high confidence)
- 60-80%: Good agreement (moderate confidence)
- 40-60%: Fair agreement (needs investigation)
- <40%: Poor agreement (major discrepancies)

## Data Collection Guide

### Historical Flood Data Sources

#### Delhi (2015-2024)
**News Archives:**
- The Hindu Delhi, Times of India, Hindustan Times, Indian Express

**Government Reports:**
- Delhi Development Authority (DDA) flood reports
- Delhi Jal Board drainage reports
- Central Water Commission flood bulletins
- IMD Delhi flood warnings

**Academic Studies:**
- IIT Delhi flood research
- TERI urban flood studies
- Search: "Delhi urban flooding Yamuna 2015-2024"

**Key Events:**
- July 2023: Yamuna flooding (record levels)
- September 2022 floods
- 2019 monsoon flooding

**Recurring Areas:**
- Yamuna floodplains
- Najafgarh drain areas
- Dwarka low-lying zones

#### Bangalore (2015-2024)
**News Archives:**
- The Hindu Bangalore, Times of India, Deccan Herald, Indian Express

**Government Reports:**
- BBMP flood reports
- Karnataka State Natural Disaster Monitoring Centre (KSNDMC)
- Bangalore Development Authority (BDA) studies

**Academic Studies:**
- IISc Bangalore flood research
- Journal of Hydrology case studies
- Search: "Bangalore urban flooding 2015-2024"

**Key Events:**
- September 2022 floods (major event)
- August 2017 floods
- 2015 monsoon flooding

**Recurring Areas:**
- Yemalur, Bellandur, HSR Layout, Whitefield

## Success Criteria

| Metric | Target | Status |
|--------|--------|--------|
| DEM quality grade (Delhi) | B or higher | ✅ A (100/100) |
| DEM quality grade (Bangalore) | B or higher | ⏳ Pending (no DEM data yet) |
| Historical flood data (Delhi) | 50-100 incidents | ⏳ Pending |
| Historical flood data (Bangalore) | 50-100 incidents | ⏳ Pending |
| Overlap with high-risk zones (Delhi) | ≥70% | ⏳ Pending |
| Overlap with high-risk zones (Bangalore) | ≥70% | ⏳ Pending |
| Official map comparison | Completed | ⏳ Pending |

## Current Status

### Completed ✅
1. Validation directory structure created
2. Three validation scripts implemented:
   - DEM quality checker
   - Historical flood overlay validator
   - Official map comparison
3. Delhi DEM quality validated: **Grade A (100/100)**

### In Progress ⏳
1. Historical flood data collection (manual research required)
2. Official map acquisition and conversion
3. Running full validation suite

### Next Steps
1. **Collect Historical Flood Data** (manual, 2-3 days per city):
   - Research news archives, government reports, academic papers
   - Verify coordinates for 50-100 incidents per city
   - Document sources with URLs
   - Create JSON files with verified data

2. **Run Historical Validation**:
   - Execute `validate_flood_zones.py`
   - Verify >70% overlap target
   - Investigate false positives/negatives

3. **Official Map Comparison**:
   - Download DDA/BDA flood maps
   - Convert to GeoJSON
   - Run `compare_official_maps.py`
   - Document agreement/disagreement areas

4. **Generate Final Report**:
   - Compile all validation metrics
   - Assign confidence scores
   - Document limitations and disclaimers
   - Prepare for public disclosure

## Limitations

The DEM-based flood risk model has inherent limitations:
- **No drainage infrastructure modeling** (storm drains, culverts, pumps)
- **No rainfall intensity integration** (static risk, not event-based)
- **No soil permeability modeling** (assumes uniform infiltration)
- **Temporal changes not captured** (urban development, land use changes)
- **Climate change impacts uncertain** (future rainfall patterns)

**These zones should be used for general awareness and urban planning guidance, not as the sole basis for emergency decisions or property assessments.**

## Files

```
validation/
├── README.md                              # This file
├── check_dem_quality.py                   # DEM quality validation
├── validate_flood_zones.py                # Historical flood overlay
├── compare_official_maps.py               # Official map comparison
├── dem-quality-report.json                # DEM quality results
├── historical-floods-bangalore.json       # (To be created)
├── historical-floods-delhi.json           # (To be created)
├── delhi-validation-report.json           # (Generated after validation)
├── bangalore-validation-report.json       # (Generated after validation)
└── official-maps/
    ├── delhi-dda-flood-zones.geojson     # (To be downloaded)
    └── bangalore-bda-flood-zones.geojson # (To be downloaded)
```

## Dependencies

```bash
pip install rasterio geopandas shapely numpy pandas
```

## Contact

For questions or issues with validation methodology, refer to the main plan document:
`C:\Users\Anirudh Mohan\.claude\plans\fuzzy-singing-eclipse.md`
