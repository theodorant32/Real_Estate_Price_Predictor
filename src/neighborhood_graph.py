"""
Graph-Based Neighborhood Modeling

Models neighborhoods as graphs to capture spatial relationships:
- Nodes: Properties, amenities, schools, transit stations
- Edges: Distance, similarity, connectivity
- Features: Centrality measures, cluster coefficients, PageRank

Enables:
- Walkability scoring
- Amenity proximity analysis
- Neighborhood similarity detection
- Spatial autocorrelation modeling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import networkx as nx


@dataclass
class Amenity:
    """Represents a neighborhood amenity."""
    id: str
    type: str  # school, park, transit, shopping, healthcare, restaurant
    name: str
    latitude: float
    longitude: float
    rating: float = 0.0  # 0-5 scale
    sub_type: str = ""  # e.g., "elementary", "highschool", "bus", "skytrain"


@dataclass
class Property:
    """Represents a property node in the graph."""
    id: str
    latitude: float
    longitude: float
    property_type: str
    price: float
    bedrooms: int = 0
    bathrooms: int = 0
    area_sqft: float = 0.0
    neighborhood: str = ""
    city: str = ""


@dataclass
class Neighborhood:
    """Represents a neighborhood cluster."""
    id: str
    name: str
    city: str
    centroid_lat: float
    centroid_lng: float
    properties: List[str] = field(default_factory=list)
    amenities: Dict[str, int] = field(default_factory=dict)
    walkability_score: float = 0.0
    transit_score: float = 0.0
    bike_score: float = 0.0


class NeighborhoodGraph:
    """
    Graph-based neighborhood modeling.

    Creates a multi-layer graph where:
    - Property nodes connect to nearby amenities
    - Properties connect to similar properties
    - Amenities connect to related amenities
    - Neighborhoods emerge as clusters
    """

    def __init__(self):
        self.graph = nx.MultiGraph()
        self.properties: Dict[str, Property] = {}
        self.amenities: Dict[str, Amenity] = {}
        self.neighborhoods: Dict[str, Neighborhood] = {}

        # Amenity weights for scoring
        self.amenity_weights = {
            "school": 1.0,
            "park": 0.8,
            "transit": 1.2,
            "shopping": 0.7,
            "healthcare": 0.9,
            "restaurant": 0.5,
            "grocery": 1.0,
            "gym": 0.4,
        }

        # Distance decay parameter (km)
        self.distance_decay = 1.0

    def add_property(self, property: Property):
        """Add a property node to the graph."""
        self.properties[property.id] = property
        self.graph.add_node(
            property.id,
            type="property",
            lat=property.latitude,
            lng=property.longitude,
            price=property.price,
            property_type=property.property_type,
            neighborhood=property.neighborhood
        )

    def add_amenity(self, amenity: Amenity):
        """Add an amenity node to the graph."""
        self.amenities[amenity.id] = amenity
        self.graph.add_node(
            amenity.id,
            type="amenity",
            amenity_type=amenity.type,
            sub_type=amenity.sub_type,
            lat=amenity.latitude,
            lng=amenity.longitude,
            rating=amenity.rating
        )

    def add_neighborhood(self, neighborhood: Neighborhood):
        """Add a neighborhood cluster."""
        self.neighborhoods[neighborhood.id] = neighborhood
        self.graph.add_node(
            f"neighborhood_{neighborhood.id}",
            type="neighborhood",
            name=neighborhood.name,
            city=neighborhood.city,
            lat=neighborhood.centroid_lat,
            lng=neighborhood.centroid_lng
        )

    def _haversine_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate distance between two points in kilometers."""
        R = 6371  # Earth's radius in km

        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lng = np.radians(lng2 - lng1)

        a = (np.sin(delta_lat / 2) ** 2 +
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lng / 2) ** 2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return R * c

    def connect_properties_to_amenities(self, max_distance_km: float = 1.5):
        """Connect properties to nearby amenities with distance-weighted edges."""
        for prop_id, prop in self.properties.items():
            for amenity_id, amenity in self.amenities.items():
                distance = self._haversine_distance(
                    prop.latitude, prop.longitude,
                    amenity.latitude, amenity.longitude
                )

                if distance <= max_distance_km:
                    # Weight decays with distance
                    weight = np.exp(-distance / self.distance_decay)
                    weight *= self.amenity_weights.get(amenity.type, 0.5)
                    weight *= (1 + amenity.rating / 5)  # Higher rated = more weight

                    self.graph.add_edge(
                        prop_id, amenity_id,
                        type="proximity",
                        distance=distance,
                        weight=weight
                    )

    def connect_similar_properties(self, k_neighbors: int = 10, max_distance_km: float = 2.0):
        """Connect properties to similar properties (k-NN graph)."""
        prop_ids = list(self.properties.keys())
        n = len(prop_ids)

        if n == 0:
            return

        # Build distance matrix
        distances = np.zeros((n, n))
        for i, id1 in enumerate(prop_ids):
            for j, id2 in enumerate(prop_ids):
                if i < j:
                    distances[i, j] = self._haversine_distance(
                        self.properties[id1].latitude, self.properties[id1].longitude,
                        self.properties[id2].latitude, self.properties[id2].longitude
                    )
                    distances[j, i] = distances[i, j]

        # Connect k-nearest neighbors
        for i, prop_id in enumerate(prop_ids):
            nearest_indices = np.argsort(distances[i])[1:k_neighbors + 1]

            for j in nearest_indices:
                if distances[i, j] <= max_distance_km:
                    neighbor_id = prop_ids[j]
                    similarity = 1.0 / (1.0 + distances[i, j])

                    self.graph.add_edge(
                        prop_id, neighbor_id,
                        type="similarity",
                        distance=distances[i, j],
                        weight=similarity
                    )

    def connect_neighborhoods_to_properties(self, max_distance_km: float = 3.0):
        """Connect neighborhood centroids to properties within range."""
        for neigh_id, neigh in self.neighborhoods.items():
            node_id = f"neighborhood_{neigh_id}"

            for prop_id, prop in self.properties.items():
                distance = self._haversine_distance(
                    neigh.centroid_lat, neigh.centroid_lng,
                    prop.latitude, prop.longitude
                )

                if distance <= max_distance_km:
                    weight = np.exp(-distance / self.distance_decay)
                    self.graph.add_edge(
                        node_id, prop_id,
                        type="contains",
                        distance=distance,
                        weight=weight
                    )

    def compute_centrality_metrics(self) -> Dict[str, Dict]:
        """Compute graph centrality metrics for all nodes."""
        metrics = {}

        # Degree centrality
        degree_cent = nx.degree_centrality(self.graph)
        metrics["degree_centrality"] = degree_cent

        # PageRank (identifies important nodes)
        if len(self.graph) > 0:
            pagerank = nx.pagerank(self.graph, weight="weight")
            metrics["pagerank"] = pagerank

        # Betweenness (identifies bridges)
        betweenness = nx.betweenness_centrality(self.graph, weight="weight")
        metrics["betweenness"] = betweenness

        return metrics

    def compute_property_scores(self) -> Dict[str, Dict]:
        """
        Compute comprehensive scores for each property.

        Returns:
            Dict mapping property_id to scores including:
            - walkability_score
            - amenity_score
            - connectivity_score
            - neighborhood_quality
        """
        scores = {}
        centrality = self.compute_centrality_metrics()

        for prop_id, prop in self.properties.items():
            # Get all amenity edges
            amenity_edges = [
                (neighbor, data)
                for neighbor, data in self.graph[prop_id].items()
                if isinstance(data, dict) and data.get("type") == "proximity"
            ]

            # Walkability score based on nearby amenities
            walkability = 0.0
            amenity_counts = defaultdict(int)

            for neighbor, edge_data in amenity_edges:
                if neighbor in self.amenities:
                    amenity = self.amenities[neighbor]
                    amenity_counts[amenity.type] += 1
                    walkability += edge_data.get("weight", 0)

            # Normalize walkability (0-100)
            walkability = min(100, walkability * 10)

            # Transit score (based on transit amenities)
            transit_score = 0.0
            for neighbor, edge_data in amenity_edges:
                if neighbor in self.amenities:
                    amenity = self.amenities[neighbor]
                    if amenity.type == "transit":
                        transit_score += edge_data.get("weight", 0)
            transit_score = min(100, transit_score * 15)

            # Connectivity score (graph centrality)
            connectivity = centrality.get("degree_centrality", {}).get(prop_id, 0) * 100
            pr_score = centrality.get("pagerank", {}).get(prop_id, 0) * 1000

            # Neighborhood quality
            neighborhood_id = prop.neighborhood
            neigh_quality = 50.0  # Default
            if neighborhood_id in self.neighborhoods:
                neigh = self.neighborhoods[neighborhood_id]
                neigh_quality = neigh.walkability_score

            scores[prop_id] = {
                "walkability_score": round(walkability, 1),
                "transit_score": round(transit_score, 1),
                "connectivity_score": round(connectivity, 2),
                "pagerank_score": round(pr_score, 4),
                "neighborhood_quality": round(neigh_quality, 1),
                "amenity_counts": dict(amenity_counts),
                "overall_location_score": round(
                    walkability * 0.4 +
                    transit_score * 0.3 +
                    connectivity * 100 +
                    neigh_quality * 0.3,
                    1
                )
            }

        return scores

    def find_similar_neighborhoods(self, neighborhood_id: str, n_similar: int = 5) -> List[Tuple[str, float]]:
        """Find neighborhoods similar to the given one based on graph structure."""
        if neighborhood_id not in self.neighborhoods:
            return []

        target = self.neighborhoods[neighborhood_id]

        similarities = []
        for other_id, other in self.neighborhoods.items():
            if other_id == neighborhood_id:
                continue

            # Compare amenity profiles
            target_amenities = set(target.amenities.keys())
            other_amenities = set(other.amenities.keys())
            jaccard = len(target_amenities & other_amenities) / len(target_amenities | other_amenities) if target_amenities | other_amenities else 0

            # Compare walkability
            walk_diff = abs(target.walkability_score - other.walkability_score) / 100

            # Combine
            similarity = jaccard * 0.5 + (1 - walk_diff) * 0.5
            similarities.append((other_id, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n_similar]

    def detect_neighborhood_clusters(self) -> List[Set[str]]:
        """Detect natural neighborhood clusters using community detection."""
        if len(self.graph) == 0:
            return []

        # Use Louvain community detection
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(self.graph, weight="weight")

            # Group properties by community
            communities = defaultdict(set)
            for node, comm_id in partition.items():
                if node in self.properties:
                    communities[comm_id].add(node)

            return list(communities.values())
        except ImportError:
            # Fallback to connected components
            return list(nx.connected_components(self.graph))

    def get_neighborhood_summary(self, neighborhood_id: str) -> Dict:
        """Get comprehensive summary for a neighborhood."""
        if neighborhood_id not in self.neighborhoods:
            return {}

        neigh = self.neighborhoods[neighborhood_id]

        # Get properties in neighborhood
        prop_ids = neigh.properties
        properties = [self.properties[p] for p in prop_ids if p in self.properties]

        if not properties:
            return {"name": neigh.name, "error": "No properties found"}

        # Calculate statistics
        prices = [p.price for p in properties]
        avg_price = np.mean(prices)
        median_price = np.median(prices)
        price_std = np.std(prices)

        # Get property scores
        prop_scores = self.compute_property_scores()
        location_scores = [prop_scores[p.id]["overall_location_score"] for p in properties if p.id in prop_scores]
        avg_location_score = np.mean(location_scores) if location_scores else 50

        return {
            "name": neigh.name,
            "city": neigh.city,
            "property_count": len(properties),
            "avg_price": round(avg_price, 0),
            "median_price": round(median_price, 0),
            "price_volatility": round(price_std / avg_price * 100, 1) if avg_price > 0 else 0,
            "walkability_score": round(neigh.walkability_score, 1),
            "transit_score": round(neigh.transit_score, 1),
            "avg_location_score": round(avg_location_score, 1),
            "amenity_profile": neigh.amenities,
            "similar_neighborhoods": self.find_similar_neighborhoods(neighborhood_id)
        }


def create_sample_graph() -> NeighborhoodGraph:
    """Create a sample neighborhood graph for demonstration."""

    graph = NeighborhoodGraph()

    # Vancouver neighborhoods
    neighborhoods = [
        Neighborhood("dt", "Downtown", "Vancouver", 49.2827, -123.1207, walkability_score=95, transit_score=90),
        Neighborhood("yw", "Yaletown", "Vancouver", 49.2750, -123.1210, walkability_score=92, transit_score=85),
        Neighborhood("kw", "Kitsilano", "Vancouver", 49.2650, -123.1550, walkability_score=85, transit_score=70),
        Neighborhood("mt", "Mount Pleasant", "Vancouver", 49.2620, -123.1000, walkability_score=88, transit_score=80),
        Neighborhood("be", "Brentwood", "Burnaby", 49.2650, -123.0000, walkability_score=75, transit_score=85),
    ]

    for neigh in neighborhoods:
        graph.add_neighborhood(neigh)

    # Add sample properties
    sample_properties = [
        Property("p1", 49.2830, -123.1210, "condo", 850000, 2, 2, 900, "dt", "Vancouver"),
        Property("p2", 49.2820, -123.1200, "condo", 750000, 1, 1, 650, "dt", "Vancouver"),
        Property("p3", 49.2760, -123.1220, "condo", 920000, 2, 2, 1100, "yw", "Vancouver"),
        Property("p4", 49.2660, -123.1560, "townhouse", 1200000, 3, 2, 1400, "kw", "Vancouver"),
        Property("p5", 49.2630, -123.1010, "condo", 680000, 1, 1, 600, "mt", "Vancouver"),
        Property("p6", 49.2655, -123.0010, "condo", 620000, 2, 1, 800, "be", "Burnaby"),
        Property("p7", 49.2640, -123.0020, "townhouse", 780000, 3, 2, 1200, "be", "Burnaby"),
    ]

    for prop in sample_properties:
        graph.add_property(prop)
        graph.neighborhoods[prop.neighborhood[:2]].properties.append(prop.id)

    # Add amenities
    amenities = [
        Amenity("a1", "transit", "Waterfront Station", 49.2857, -123.1115, rating=4.5, sub_type="skytrain"),
        Amenity("a2", "transit", "Granville Station", 49.2835, -123.1145, rating=4.0, sub_type="skytrain"),
        Amenity("a3", "school", "Lord Roberts Elementary", 49.2780, -123.1280, rating=4.2, sub_type="elementary"),
        Amenity("a4", "park", "David Lam Park", 49.2730, -123.1200, rating=4.6),
        Amenity("a5", "shopping", "Pacific Centre", 49.2820, -123.1175, rating=4.3),
        Amenity("a6", "healthcare", "VGH Hospital", 49.2630, -123.1220, rating=4.0),
        Amenity("a7", "grocery", "Whole Foods", 49.2750, -123.1200, rating=4.2),
        Amenity("a8", "restaurant", "Restaurant Row", 49.2760, -123.1220, rating=4.5),
        Amenity("a9", "park", "Kitsilano Beach", 49.2750, -123.1550, rating=4.8),
        Amenity("a10", "transit", "Broadway Station", 49.2635, -123.0950, rating=4.2, sub_type="skytrain"),
        Amenity("a11", "transit", "Brentwood Station", 49.2655, -123.0015, rating=4.3, sub_type="skytrain"),
        Amenity("a12", "shopping", "Brentwood Mall", 49.2660, -123.0000, rating=4.1),
    ]

    for amenity in amenities:
        graph.add_amenity(amenity)

    # Build graph connections
    graph.connect_properties_to_amenities(max_distance_km=1.5)
    graph.connect_similar_properties(k_neighbors=5, max_distance_km=2.0)
    graph.connect_neighborhoods_to_properties(max_distance_km=3.0)

    return graph


def main():
    print("=" * 70)
    print("NEIGHBORHOOD GRAPH ANALYSIS")
    print("=" * 70)

    # Create sample graph
    graph = create_sample_graph()

    print(f"\nGraph Statistics:")
    print(f"  Properties: {len(graph.properties)}")
    print(f"  Amenities: {len(graph.amenities)}")
    print(f"  Neighborhoods: {len(graph.neighborhoods)}")
    print(f"  Total Nodes: {graph.graph.number_of_nodes()}")
    print(f"  Total Edges: {graph.graph.number_of_edges()}")

    # Compute property scores
    print("\n" + "=" * 70)
    print("PROPERTY LOCATION SCORES")
    print("=" * 70)

    scores = graph.compute_property_scores()

    for prop_id, prop_scores in scores.items():
        prop = graph.properties[prop_id]
        print(f"\n{prop.neighborhood} - {prop.property_type} (${prop.price:,})")
        print(f"  Walkability: {prop_scores['walkability_score']}")
        print(f"  Transit: {prop_scores['transit_score']}")
        print(f"  Location Score: {prop_scores['overall_location_score']}")

    # Neighborhood summaries
    print("\n" + "=" * 70)
    print("NEIGHBORHOOD SUMMARIES")
    print("=" * 70)

    for neigh_id, neigh in graph.neighborhoods.items():
        summary = graph.get_neighborhood_summary(neigh_id)
        if "error" not in summary:
            print(f"\n{summary['name']} ({summary['city']})")
            print(f"  Properties: {summary['property_count']}")
            print(f"  Avg Price: ${summary['avg_price']:,.0f}")
            print(f"  Walkability: {summary['walkability_score']}")
            print(f"  Location Score: {summary['avg_location_score']}")

            if summary["similar_neighborhoods"]:
                similar = summary["similar_neighborhoods"][0]
                print(f"  Similar to: {similar[0]} ({similar[1]:.0%} similar)")


if __name__ == "__main__":
    main()
