import click
import geopandas as gpd
import logging
import networkx as nx
import pandas as pd
import sys
from itertools import product
from operator import attrgetter, itemgetter
from pathlib import Path
from shapely import Point
from tqdm import tqdm

# Set logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
logger.addHandler(handler)


class LeastCostPaths:
    """Defines the LeastCostPaths class."""

    def __init__(self, src_nodes: Path, field_index_source: str, field_index_target: str, field_cost: str,
                 src_pts: Path, layer_pts_start: str, layer_pts_destination: str, field_pt_index: str,
                 field_pt_group: str, src_index_pt_lookup: Path, field_lookup_index: str, field_lookup_x: str,
                 field_lookup_y: str) -> None:
        """Initializes the LeastCostPaths class."""

        self.crs = "EPSG:3043"
        self.dst = src_nodes / "least_cost_paths.gpkg"
        self.dst_layer = "least_cost_paths"
        self.results = pd.DataFrame()
        self.pt_pairs = pd.DataFrame()
        self.graph = nx.DiGraph()

        # Define variables for nodes-cost source.
        self.src_nodes = pd.DataFrame()
        self.field_index_source = field_index_source
        self.field_index_target = field_index_target
        self.field_cost = field_cost

        # Define variables for points source.
        self.src_pts_start = gpd.GeoDataFrame()
        self.src_pts_destination = gpd.GeoDataFrame()
        self.field_pt_index = field_pt_index
        self.field_pt_group = field_pt_group

        # Define variables for index-pt lookup source.
        self.src_index_pt_lookup = pd.DataFrame()
        self.field_lookup_index = field_lookup_index
        self.field_lookup_x = field_lookup_x
        self.field_lookup_y = field_lookup_y

        # Compile source data - node index-cost pairs.
        logger.info(f"Compiling source data - node index-cost pairs: {src_nodes}.")
        self.src_nodes = pd.read_csv(src_nodes, sep=",", header=True)
        logger.info(f"Successfully loaded {len(self.src_nodes)} records.")

        # Compile source data - start points.
        logger.info(f"Compiling source data - start points: {src_pts}, layer={layer_pts_start}.")
        self.src_pts_start = gpd.read_file(src_pts, layer=layer_pts_start)
        logger.info(f"Successfully loaded {len(self.src_pts_start)} records.")

        # Compile source data - destination points.
        logger.info(f"Compiling source data - destination points: {src_pts}, layer={layer_pts_destination}.")
        self.src_pts_destination = gpd.read_file(src_pts, layer=layer_pts_destination)
        logger.info(f"Successfully loaded {len(self.src_pts_destination)} records.")

        # Compile source data - index-pt lookup.
        logger.info(f"Compiling source data - index-pt lookup: {src_index_pt_lookup}.")
        self.src_index_pt_lookup = pd.read_csv(src_index_pt_lookup, sep=",", header=True)
        logger.info(f"Successfully loaded {len(self.src_index_pt_lookup)} records.")

    def __call__(self) -> None:
        """Executes the LeastCostPaths class."""

        self.permute_pt_pairs()
        self.create_graph()
        self.calculate_least_cost_paths()
        self.export()

    def calculate_least_cost_paths(self) -> None:
        """Calculates least-cost paths for each point pair."""

        logger.info("Calculating least-cost paths.")

        # TODO - calculate lcps - syntax = nx.shortest_path(self.graph, source=?, target=?, weight="weight", method="dijkstra")

    def create_graph(self) -> None:
        """Creates a NetworkX DiGraph from a collection of node indexes and cost values."""

        logger.info(f"Creating NetworkX DiGraph.")

        # Iterate node data in chunks.
        chunksize = 50000
        fields = [self.field_index_source, self.field_index_target, self.field_cost]
        for idx in tqdm(range(int(len(self.src_nodes) / chunksize) + 1)):

            # Add node data to graph as edges.
            _ = self.src_nodes.loc[(self.src_nodes.index >= (idx * chunksize)) &
                                   (self.src_nodes.index < ((idx + 1) * chunksize)), fields].apply(dict, axis=1)\
                .map(lambda row: self.graph.add_edge(u_of_edge=row[self.field_index_source],
                                                     v_of_edge=row[self.field_index_target],
                                                     weight=row[self.field_cost]))

        logger.info(f"Successfully created DiGraph with {self.graph.number_of_nodes()} nodes and "
                    f"{self.graph.number_of_edges()} edges.")

    def export(self) -> None:
        """Construct and export output dataset."""

        logger.info("Compiling geometries associated with each collection of least-cost path node indexes.")

        # TODO - export least cost paths as start pt (idx), destination pt (idx), geometry (linestring), total cost.

        # Export to GeoPackage.
        logger.info(f"Exporting results to: {self.dst}, layer={self.dst_layer}.")
        self.results.to_file(self.dst, layer=self.dst_layer)
        logger.info(f"Successfully exported results to: {self.dst}, layer={self.dst_layer}.")

    def permute_pt_pairs(self) -> None:
        """Permutes each start / destination point pair within each named group."""

        logger.info("Permuting start / destination point pairs.")
        pt_pairs = list()

        # Iterate named groups.
        for group in set(self.src_pts_start[self.field_pt_group]):

            # Compile start / destination indexes.
            pts_start = set(self.src_pts_start.loc[self.src_pts_start[self.field_pt_group] == group,
                                                   self.field_pt_index])
            pts_destination = set(self.src_pts_destination.loc[self.src_pts_destination[self.field_pt_group] == group,
                                                               self.field_pt_index])

            # Compile permutations as DataFrame.
            pts_start_, pts_destination_ = zip(*product(pts_start, pts_destination))
            pt_pairs.append(pd.DataFrame({"group": group, "source": pts_start_, "target": pts_destination_}))

            logger.info(f"Compiled {len(self.pt_pairs[group])} start / destination pairs for group: {group}.")

        # Concatenate all permutations into single DataFrame.
        self.pt_pairs = pd.concat(pt_pairs, axis=0, ignore_index=True)


@click.command()
@click.argument("src_nodes",
                type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path))
@click.argument("field_index_source", type=click.STRING)
@click.argument("field_index_target", type=click.STRING)
@click.argument("field_cost", type=click.STRING)
@click.argument("src_pts",
                type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path))
@click.argument("layer_pts_start", type=click.STRING)
@click.argument("layer_pts_destination", type=click.STRING)
@click.argument("field_pt_index", type=click.STRING)
@click.argument("field_pt_group", type=click.STRING)
@click.argument("src_index_pt_lookup",
                type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path))
@click.argument("field_lookup_index", type=click.STRING)
@click.argument("field_lookup_x", type=click.STRING)
@click.argument("field_lookup_y", type=click.STRING)
def main(src_nodes: Path, field_index_source: str, field_index_target: str, field_cost: str, src_pts: Path,
         layer_pts_start: str, layer_pts_destination: str, field_pt_index: str, field_pt_group: str,
         src_index_pt_lookup: Path, field_lookup_index: str, field_lookup_x: str, field_lookup_y: str) -> None:
    """
    \b
    Description: Creates a NetworkX DiGraph from a set of node index pairs as edges, with associated cost values. For
    each permutation of start and destination points within each named group of start and destination points,
    calculates the least-cost path along the DiGraph, using the cost values as the weight. Outputs a GeoPackage,
    'least_cost_paths.gpkg' | layer='least_cost_paths', within the same directory as `src_nodes` input containing for
    each least-cost path: least-cost path geometry (LineString), start point index, destination point index, total
    cost, and start / destination point group name.

    \b
    Assumptions:
        - All spatial data is in the same, meter-based projection.
        - All start / destination points match node coordinates added to the DiGraph.
        - All start / destination points' named groups exist in each file.
        - The collection of node pairs added to the DiGraph as edges does not form any subgraphs (disconnected areas).
        - The collection of node pairs added to the DiGraph as edges does not contain any negative cost values.
        - All input CSVs have headers as the first row and comma (,) as the delimiter.

    \b
    :param Path src_nodes: CSV (.csv) containing each pair of source (from) and target (to) node indexes and the
        associated cost value to be added to a NetworkX DiGraph as edges.
    :param str field_index_source: CSV field containing source node indexes.
    :param str field_index_target: CSV field containing target node indexes.
    :param str field_cost: CSV field containing cost values for each node index pair.
    :param Path src_pts: GeoPackage (.gpkg) containing start and destination point layers.
    :param str layer_pts_start: GeoPackage layer containing start point geometries, associated node indexes, and names
        used to group sets of points.
    :param str layer_pts_destination: GeoPackage layer containing destination point geometries, associated node
        indexes, and names used to group sets of points.
    :param str field_pt_index: Field containing geometry-node indexes for start / destination point layers.
    :param str field_pt_group: Field containing point group names for start / destination point layers.
    :param Path src_index_pt_lookup: CSV (.csv) containing lookup data for node indexes and their geometry coordinates.
    :param str field_lookup_index: Lookup field containing node indexes.
    :param str field_lookup_x: Lookup field containing node longitude (x) values.
    :param str field_lookup_y: Lookup field containing node latitude (y) values.
    """

    try:

        least_cost_paths = LeastCostPaths(src_nodes, field_index_source, field_index_target, field_cost, src_pts,
                                          layer_pts_start, layer_pts_destination, field_pt_index, field_pt_group,
                                          src_index_pt_lookup, field_lookup_index, field_lookup_x, field_lookup_y)
        least_cost_paths()

    except KeyboardInterrupt:
        logger.exception("KeyboardInterrupt: Exiting program.")
        sys.exit(1)


if __name__ == "__main__":
    main()
