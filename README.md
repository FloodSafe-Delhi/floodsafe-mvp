# 📘 FloodSafe Project Guide

Welcome! This guide is designed to help you understand the **FloodSafe** codebase, its architecture, and the vision behind it. Whether you are a student or a contributor, this document will walk you through everything from the folder structure to the future roadmap.

---

## ✅ System Status: OPERATIONAL

**All services are running and ready for development**:
- ✅ **Backend API**: http://localhost:8000 (FastAPI with Clean Architecture)
- ✅ **API Documentation**: http://localhost:8000/docs (Swagger UI)
- ✅ **Frontend**: http://localhost:5175 (React + Vite + MapLibre GL)
- ✅ **Database**: PostgreSQL + PostGIS on port 5432 (seeded with Bangalore data)
- ✅ **IoT Service**: http://localhost:8001 (High-throughput ingestion)

**To start the entire system**:
```powershell
docker compose up -d
```

---

## 🏗️ 1. Understanding the Codebase

We use a **Monorepo** structure. This means all our different projects (Backend, Frontend, IoT) live in one single repository. This makes it easier to share code and run everything together.

### 📂 Directory Structure

```
FloodSafe/
├── apps/                  # Where the actual code lives
│   ├── backend/           # The Brain (Python/FastAPI)
│   ├── frontend/          # The Face (React/Vite)
│   └── iot-ingestion/     # The Ears (Python/IoT)
├── docker-compose.yml     # The Orchestra Conductor
└── pnpm-workspace.yaml    # The Glue (links everything)
```

### 🧠 The Backend (`apps/backend`)
**Tech**: Python, FastAPI, SQLAlchemy, Pydantic V2, PostGIS.
**Pattern**: **Clean Architecture**.
*   **Why?** We separate the "Business Logic" (Domain) from the "Database" (Infrastructure). This means if we change the database later, our core logic doesn't break.
*   **Key Files**:
    *   `src/domain/models.py`: Defines what a "User" or "Report" *is* (Pydantic models).
    *   `src/domain/services/interfaces.py`: Abstract interfaces for AI, Notifications, and Routing (future extensibility).
    *   `src/infrastructure/models.py`: SQLAlchemy ORM models mapping to PostgreSQL + PostGIS.
    *   `src/api/reports.py`: Handles flood report uploads with geotagging verification.
    *   `src/api/webhook.py`: WhatsApp SOS webhook endpoint.
    *   `src/core/utils.py`: EXIF/GPS extraction utilities.
    *   `src/scripts/seed_db.py`: Database seeding script with Bangalore flood zones.

### 🎨 The Frontend (`apps/frontend`)
**Tech**: React, TypeScript, Vite, TanStack Query, MapLibre GL, PMTiles.
**Key Feature**: **DEM-based Vector Flood Maps**.
*   **Why?** Instead of loading heavy images for the map, we use "Vector Tiles" (`.pmtiles`) generated from scientifically-processed DEM data. This makes the map fast, zoomable, and scientifically accurate without needing a Google Maps API key.
*   **Key Files**:
    *   `src/lib/map/useMap.ts`: The hook that loads the map, handles layers, and renders graduated flood risk visualization
    *   `src/lib/map/cityConfigs.ts`: Multi-city configuration with bounds, PMTiles sources, and display settings
    *   `src/components/MapComponent.tsx`: Main map display component with city selector
    *   `data/delhi/dem/`: DEM processing pipeline (WhiteboxTools + Docker GDAL)
    *   `data/validation/`: Quality assurance tools for flood data accuracy

### 📡 IoT Ingestion (`apps/iot-ingestion`)
**Tech**: Python (FastAPI).
**Purpose**: To listen to thousands of sensors.
*   **Why separate?** If 10,000 sensors send data at once, we don't want the main website to slow down. This service handles the noise.

---

## 🚀 2. Current Features (Implemented)

### Advanced Flood Visualization
1.  **DEM-based Flood Risk Mapping**:
    *   **Hydrological Analysis**: Uses Digital Elevation Model (DEM) processing with WhiteboxTools for scientific flood prediction
    *   **Stream Influence Zones**: Calculates flow direction, flow accumulation, and stream networks
    *   **4-Level Risk Classification**: Graduated color scheme showing flood risk from low (yellow) to critical (dark blue)
    *   **Validated Against Official Data**: Includes validation tools to compare with government flood hazard maps

2.  **High-Performance Map Infrastructure**:
    *   **Multi-City Support**: Delhi and Bangalore with city selector
    *   **Enhanced Basemaps**: Zoom level 15 support with detailed buildings, POIs, and landuse data
    *   **Comprehensive PMTiles**:
        - Basemap: 33MB with high-detail OpenMapTiles
        - Flood zones: 21MB with complete DEM-processed risk data
    *   **Efficient Range Requests**: Vite configured for HTTP range requests and caching
    *   **City-Specific Configuration**: Each city has custom bounds, center, and zoom levels

3.  **Geotagged Photo Reporting**:
    *   Users can upload photos with flood reports.
    *   **Smart EXIF Verification**: Backend extracts GPS coordinates from photo metadata and verifies location accuracy.
    *   Location mismatch flagging for suspicious reports.
    
3.  **Real-time Sensor Integration**:
    *   System receives and stores water level data from IoT sensors.
    *   Sensor status tracking (active, warning, inactive).
    
4.  **WhatsApp SOS Emergency System**:
    *   Dedicated webhook endpoint (`/api/webhooks/whatsapp`) for receiving location pins.
    *   High-priority report creation for emergency situations.

### Gamification System
5.  **User Profiles & Leaderboards**:
    *   **Points System**: Users earn points for submitting verified reports.
    *   **Levels**: Progressive leveling system based on contributions.
    *   **Badges**: Achievement tracking ("First Reporter", "Flood Guardian", etc.).
    *   **Stats Tracking**: 
        - Total reports count
        - Verified reports count
        - User reputation score

### Data Processing & Validation
6.  **DEM Processing Pipeline**:
    *   **Automated Workflow**: Python scripts for DEM-to-flood-zones processing
    *   **Hydrological Processing**:
        - Depression filling for accurate flow modeling
        - D8 flow direction and accumulation calculation
        - Stream network extraction with configurable thresholds
        - Gaussian filtering for stream influence zones
    *   **PMTiles Generation**: Docker-based workflow for cross-platform tile generation
    *   **Quality Assurance**:
        - DEM quality validation (completeness, anomalies, statistics)
        - Flood zone geometry validation
        - Cross-validation with official flood maps
        - Automated quality reports in JSON format

### Analytics Foundation
7.  **Area-Based Analytics**:
    *   Flood zone risk levels (1-4 graduated scale based on DEM analysis).
    *   Report aggregation by location.
    *   Upvote/verification score tracking.
    *   Foundation for dashboards showing:
        - Most flooded areas
        - Report trends over time
        - Sensor coverage maps
        - Stream influence and flow patterns

### Technical Infrastructure
8.  **Production-Ready Configuration**:
    *   **Environment Variables**: VITE_API_URL for flexible API endpoint configuration
    *   **CORS Support**: Network IP addresses allowed for local network testing
    *   **PMTiles Optimization**:
        - Vite configured with Accept-Ranges headers for efficient tile streaming
        - Asset handling optimized for large binary files
        - HMR (Hot Module Reload) configured for development efficiency
    *   **React Strict Mode Compatible**: Protocol registration handles double-mounting correctly

---

## 📱 3. The Mobile App Path

We designed this system to be **Mobile-First**. Here is how we turn this into an app:

### Strategy: "Capacitor" (The Fast Way)
We don't need to rewrite the code. We can use a tool called **Capacitor** to wrap our React website into a native Android/iOS app.
1.  **Install Capacitor** in `apps/frontend`.
2.  **Build**: It takes the web code and puts it inside a native container.
3.  **Deploy**: You get an `.apk` file to install on your phone.

**Why this is cool**:
*   The **Backend** doesn't care if it's a web or mobile app. It just serves JSON.
*   We can access native features like **Push Notifications** and **Background Geolocation** using Capacitor plugins.

---

## 🔮 4. Future Features (Architected & Ready to Build)

These features have complete interface definitions and database schema support but need implementation:

### Production Gaps

1.  **Video Reporting (Deferred for MVP)**
    *   **The Goal**: Allow users to upload videos.
    *   **The Gap**: Requires cloud storage (AWS S3/Google Cloud).
    *   **Code Prep**: `Report` model has `media_type="video"` field ready.

2.  **AI Flood Prediction (Prophet)**
    *   **The Goal**: Predict flood probability 1-2 hours in advance.
    *   **The Gap**: Requires historical data collection and model training.
    *   **Code Prep**: `IPredictionService` interface defined in `src/domain/services/interfaces.py`.

3.  **Safe Route Navigation**
    *   **The Goal**: Suggest routes avoiding flooded areas.
    *   **The Gap**: Integration with OSRM or Google Routes API.
    *   **Code Prep**: `IRoutingService` interface ready.

4.  **Omnichannel Alerts (SMS/WhatsApp/Push)**
    *   **The Goal**: Auto-send alerts when sensors hit critical levels.
    *   **The Gap**: Twilio account setup and WhatsApp Business approval.
    *   **Code Prep**: `INotificationService` interface and webhook endpoint exist.

---

## 🛠️ Getting Started

### Prerequisites
- Docker Desktop (for Windows)
- Git
- Node.js 18+ (for frontend development outside Docker)
- Python 3.11+ (for backend development outside Docker)
- Python packages for DEM processing: `whitebox`, `gdal` (via Docker)

### Quick Start
```powershell
# Clone the repository
git clone https://github.com/FloodSafe-Delhi/floodsafe-mvp.git
cd floodsafe-mvp

# Start all services
docker compose up -d

# Seed the database (first time only)
python apps/backend/src/scripts/seed_db.py

# Access the application
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
# Frontend: http://localhost:5175 or http://192.168.210.102:5175 (network access)
```

### DEM Data Processing (Optional)
If you need to process new DEM data for additional cities:

```powershell
# Navigate to DEM processing directory
cd apps/frontend/data/delhi/dem

# Process DEM to flood zones (requires WhiteboxTools)
python process_simple.py

# Generate PMTiles from processed GeoJSON
.\generate-pmtiles-docker.ps1

# Validate DEM quality
cd ../../validation
python check_dem_quality.py

# Compare with official flood maps (if available)
python compare_official_maps.py
```

---

## 📚 Documentation

### Core Documentation
- **Engineering Handbook**: `engineering_handbook.md` - Development standards and best practices
- **Architecture Guide**: `architecture.md` - System design and data flow
- **Project Assessment**: `project_assessment.md` - Risk analysis and roadmap
- **Roadmap**: `roadmap_and_status.md` - Development phases and timelines

### Data Processing Documentation
- **Validation Guide**: `apps/frontend/data/validation/README.md` - DEM quality assurance workflows
- **DEM Processing**: Scripts in `apps/frontend/data/delhi/dem/` for hydrological analysis
- **City Configurations**: `apps/frontend/src/lib/map/cityConfigs.ts` - Multi-city setup guide

---

## 🎓 For Contributors

*   **Start Small**: Run `docker compose up -d` and verify all services are healthy.
*   **Read the Handbook**: Check `engineering_handbook.md` for coding standards.
*   **Explore**: Look at `apps/backend/src/domain/models.py`. Try adding a new field to the `User` model.
*   **Test Locally**: Use Swagger UI at http://localhost:8000/docs to test API endpoints.
*   **Break Things**: It's the best way to learn. If the map stops loading, check the browser console!

---

## 🤝 Contributing

We follow the **Vertical Slice** development pattern:
1. Domain (Pydantic models)
2. Infrastructure (SQLAlchemy models)
3. API (FastAPI endpoints)
4. UI (React components)

See `engineering_handbook.md` for detailed contribution guidelines.

---

Happy Coding! 🚀
