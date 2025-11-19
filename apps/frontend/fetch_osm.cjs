const https = require('https');
const fs = require('fs');

// Updated query to fetch relations with geometry AND the referenced nodes to get their tags (names)
const url = 'https://overpass-api.de/api/interpreter?data=%5Bout%3Ajson%5D%3Brelation%5B%22network%22%3D%22Namma%20Metro%22%5D%5B%22type%22%3D%22route%22%5D%3Bout%20geom%3Bnode%28r%29%3Bout%20body%3B';

https.get(url, (res) => {
    let data = '';
    res.on('data', (chunk) => {
        data += chunk;
    });
    res.on('end', () => {
        try {
            const osmData = JSON.parse(data);
            const lines = {
                "Purple": { coordinates: [], color: "#e542de" },
                "Green": { coordinates: [], color: "#009933" },
                "Yellow": { coordinates: [], color: "#ffff00" }
            };
            const stations = [];
            const processedLines = new Set();

            // Create a lookup for node details (names)
            const nodeLookup = {};
            osmData.elements.forEach(element => {
                if (element.type === 'node' && element.tags) {
                    nodeLookup[element.id] = element.tags.name || element.tags['name:en'] || "Unknown Station";
                }
            });

            osmData.elements.forEach(element => {
                if (element.type === 'relation') {
                    const color = element.tags.colour || element.tags.color;
                    const ref = element.tags.ref;
                    let lineKey = null;

                    if (ref && ref.includes('Purple')) lineKey = 'Purple';
                    else if (ref && ref.includes('Green')) lineKey = 'Green';
                    else if (ref && ref.includes('Yellow')) lineKey = 'Yellow';

                    if (!lineKey && element.tags.name) {
                        if (element.tags.name.includes('Purple')) lineKey = 'Purple';
                        else if (element.tags.name.includes('Green')) lineKey = 'Green';
                        else if (element.tags.name.includes('Yellow')) lineKey = 'Yellow';
                    }

                    // Only process the first relation we find for each line to avoid double tracks
                    if (lineKey && !processedLines.has(lineKey)) {
                        processedLines.add(lineKey);

                        // Extract geometry for lines
                        const segments = [];
                        element.members.forEach(member => {
                            if (member.type === 'way' && member.geometry) {
                                const segment = member.geometry.map(pt => [pt.lon, pt.lat]);
                                segments.push(segment);
                            }
                        });

                        lines[lineKey].coordinates = segments;
                    }

                    // Collect stations from the chosen relation (or all? let's do chosen to match line)
                    if (lineKey && processedLines.has(lineKey)) {
                        element.members.forEach(member => {
                            if (member.type === 'node' && member.role === 'stop') {
                                const stationName = nodeLookup[member.ref] || "Unknown Station";
                                stations.push({
                                    type: "Feature",
                                    properties: {
                                        name: stationName,
                                        line: lineKey,
                                        color: lines[lineKey].color
                                    },
                                    geometry: {
                                        type: "Point",
                                        coordinates: [member.lon, member.lat]
                                    }
                                });
                            }
                        });
                    }
                }
            });

            // Construct Line GeoJSON
            const lineFeatures = Object.keys(lines).map(key => {
                return {
                    type: "Feature",
                    properties: {
                        name: key + " Line",
                        ref: key,
                        colour: lines[key].color,
                        network: "Namma Metro"
                    },
                    geometry: {
                        type: "MultiLineString",
                        coordinates: lines[key].coordinates
                    }
                };
            });

            const linesGeoJSON = {
                type: "FeatureCollection",
                features: lineFeatures
            };

            // Construct Stations GeoJSON
            const uniqueStations = [];
            const seenCoords = new Set();
            stations.forEach(st => {
                const key = st.geometry.coordinates.join(',');
                if (!seenCoords.has(key)) {
                    seenCoords.add(key);
                    uniqueStations.push(st);
                }
            });

            const stationsGeoJSON = {
                type: "FeatureCollection",
                features: uniqueStations
            };

            fs.writeFileSync('metro-lines-detailed.geojson', JSON.stringify(linesGeoJSON, null, 2));
            fs.writeFileSync('metro-stations.geojson', JSON.stringify(stationsGeoJSON, null, 2));
            console.log('Successfully wrote metro-lines-detailed.geojson and metro-stations.geojson');

        } catch (e) {
            console.error('Error parsing JSON:', e);
        }
    });
}).on('error', (e) => {
    console.error('Error fetching data:', e);
});
