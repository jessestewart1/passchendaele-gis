import click
import geopandas as gpd
import logging
import pandas as pd
import sys
from igraph import Graph
from itertools import chain, product
from pathlib import Path
from shapely import LineString, Point
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
                 src_pts: Path, layer_pts_source: str, layer_pts_target: str, field_pt_index: str, field_pt_group: str,
                 src_index_pt_lookup: Path, field_lookup_index: str, field_lookup_x: str, field_lookup_y: str,
                 dst_name: str, pts_index_start: int = -1, pts_index_end: int = -1) -> None:
        """Initializes the LeastCostPaths class."""

        self.crs = "EPSG:3043"
        self.dst = src_nodes.parent / "least_cost_paths.gpkg"
        self.dst_layer = dst_name
        self.results = gpd.GeoDataFrame(geometry=gpd.GeoSeries(), crs=self.crs)
        self.pt_pairs = pd.DataFrame()
        self.pts_index_start = pts_index_start
        self.pts_index_end = pts_index_end
        self.graph = Graph(directed=True)

        # Define variables for nodes-cost source.
        self.src_nodes = pd.DataFrame()
        self.field_index_source = field_index_source
        self.field_index_target = field_index_target
        self.field_cost = field_cost

        # Define variables for points source.
        self.src_pts_source = gpd.GeoDataFrame()
        self.src_pts_target = gpd.GeoDataFrame()
        self.field_pt_index = field_pt_index
        self.field_pt_group = field_pt_group

        # Define variables for index-pt lookup source.
        self.src_index_pt_lookup = pd.DataFrame()
        self.field_lookup_index = field_lookup_index
        self.field_lookup_x = field_lookup_x
        self.field_lookup_y = field_lookup_y

        # Compile source data - node index-cost pairs.
        logger.info(f"Compiling source data - node index-cost pairs: {src_nodes}.")
        self.src_nodes = pd.read_csv(src_nodes, sep=",", header=0)
        logger.info(f"Successfully loaded {len(self.src_nodes):,} records.")

        # Compile source data - source points.
        logger.info(f"Compiling source data - source points: {src_pts}, layer={layer_pts_source}.")
        self.src_pts_source = gpd.read_file(src_pts, layer=layer_pts_source)
        logger.info(f"Successfully loaded {len(self.src_pts_source):,} records.")

        # Compile source data - target points.
        logger.info(f"Compiling source data - target points: {src_pts}, layer={layer_pts_target}.")
        self.src_pts_target = gpd.read_file(src_pts, layer=layer_pts_target)
        logger.info(f"Successfully loaded {len(self.src_pts_target):,} records.")

        # Compile source data - index-pt lookup.
        logger.info(f"Compiling source data - index-pt lookup: {src_index_pt_lookup}.")
        self.src_index_pt_lookup = pd.read_csv(src_index_pt_lookup, sep=",", header=0)
        logger.info(f"Successfully loaded {len(self.src_index_pt_lookup):,} records.")

    def __call__(self) -> None:
        """Executes the LeastCostPaths class."""

        self.permute_pt_pairs()
        self.create_graph()
        self.calculate_least_cost_paths()
        self.export()

    def calculate_least_cost_paths(self) -> None:
        """Calculates least-cost paths for each point pair."""

        logger.info("Calculating least-cost paths.")

        # Populate results GeoDataFrame with source and target indexes and group names from point pairs.
        for field in ("source", "target", "group"):
            self.results[field] = self.pt_pairs[field]

        self.results["indexes"] = None

        # Iteratively calculate least-cost paths.
        # Note: Vectorization not supported for igraph.
        for idx in tqdm(range(len(self.results))):

            # Compile input values.
            index_source = self.results.at[idx, "source"]
            index_target = self.results.at[idx, "target"]

            # Calculate least-cost paths using Dijkstra's algorithm.
            self.results.at[idx, "indexes"] = self.graph.get_shortest_path(
                v=index_source, to=index_target, weights="weight", mode="out", output="vpath", algorithm="dijkstra")

    def create_graph(self) -> None:
        """Creates a directed Graph from a collection of node indexes and cost values."""

        logger.info(f"Creating Graph - Adding edges.")

        # Add vertex count to Graph.
        self.graph.add_vertices(len(set(self.src_index_pt_lookup[self.field_lookup_index])))

        # Add node index pairs as Graph edges.
        self.graph.add_edges(zip(self.src_nodes[self.field_index_source], self.src_nodes[self.field_index_target]))

        logger.info(f"Creating Graph - Adding weights.")

        # Add weights to Graph.
        self.graph.es["weight"] = self.src_nodes[self.field_cost]

        logger.info(f"Successfully created Graph of size: nodes={self.graph.vcount():,}, "
                    f"edges={self.graph.ecount():,}.")

    def export(self) -> None:
        """Construct and export output dataset."""

        logger.info("Constructing output dataset - Compiling geometries.")

        # Create node index - point lookup dict using only relevant indexes (those encountered from least-cost paths).
        indexes = set(chain.from_iterable(self.results["indexes"]))
        lookup_df = self.src_index_pt_lookup.loc[self.src_index_pt_lookup[self.field_lookup_index].isin(indexes)]
        lookup = dict(zip(lookup_df[self.field_lookup_index],
                          lookup_df[[self.field_lookup_x, self.field_lookup_y]].apply(dict, axis=1)
                          .map(lambda row: Point(row[self.field_lookup_x], row[self.field_lookup_y]))))

        # Compile geometries associated with each least-cost path index collection.
        self.results["geometry"] = (self.results["indexes"]
                                    .map(lambda idxs: map(lambda idx: lookup[idx], idxs))
                                    .map(LineString))

        logger.info("Constructing output dataset - Compiling cost totals.")

        # Compile total costs from edges using edge IDs.
        self.results["cost"] = pd.Series(self.results["indexes"]
                                         .map(lambda idxs: zip(idxs[:-1], idxs[1:]))
                                         .map(lambda idxs: map(lambda idxs_: self.graph.get_eid(*idxs_), idxs))
                                         .map(lambda eids: map(lambda eid: self.graph.es[eid]["weight"], eids))
                                         .map(sum)).round(6)

        logger.info("Constructing output dataset - Compiling distance totals.")

        # Compile total distance from geometry lengths.
        self.results["distance"] = self.results.length.round(6)

        # Export to GeoPackage.
        logger.info(f"Exporting results to: {self.dst}, layer={self.dst_layer}.")
        self.results[["group", "source", "target", "cost", "distance", "geometry"]]\
            .to_file(self.dst, layer=self.dst_layer)
        logger.info(f"Successfully exported results to: {self.dst}, layer={self.dst_layer}.")

    def permute_pt_pairs(self) -> None:
        """Permutes each source - target point pair within each named group."""

        logger.info("Permuting source - target point pairs.")
        pt_pairs = list()

        # Iterate named groups.
        for group in set(self.src_pts_source[self.field_pt_group]):

            # Compile source and target indexes.
            pts_source = set(self.src_pts_source.loc[self.src_pts_source[self.field_pt_group] == group,
                                                     self.field_pt_index])
            pts_target = set(self.src_pts_target.loc[self.src_pts_target[self.field_pt_group] == group,
                                                     self.field_pt_index])

            # Compile permutations as DataFrame.
            pts_source_, pts_target_ = zip(*product(pts_source, pts_target))
            pt_pairs.append(pd.DataFrame({"group": group, "source": pts_source_, "target": pts_target_}))

            logger.info(f"Compiled {len(pt_pairs[-1])} source - target pairs for group: {group}.")

        # Concatenate all permutations into single DataFrame.
        self.pt_pairs = pd.concat(pt_pairs, axis=0, ignore_index=True)

        # Subset point pairs, if required.
        if (self.pts_index_start >= 0) and (self.pts_index_end > self.pts_index_start):

            self.pt_pairs = self.pt_pairs.loc[(self.pt_pairs.index >= self.pts_index_start) &
                                              (self.pt_pairs.index <= self.pts_index_end)].copy(deep=True)
            self.pt_pairs.reset_index(drop=True, inplace=True)

            logger.info(f"Source - target point pairs subset to indexes: {self.pts_index_start} - {self.pts_index_end} "
                        f"(inclusively).")


@click.command()
@click.argument("src_nodes",
                type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path))
@click.argument("field_index_source", type=click.STRING)
@click.argument("field_index_target", type=click.STRING)
@click.argument("field_cost", type=click.STRING)
@click.argument("src_pts",
                type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path))
@click.argument("layer_pts_source", type=click.STRING)
@click.argument("layer_pts_target", type=click.STRING)
@click.argument("field_pt_index", type=click.STRING)
@click.argument("field_pt_group", type=click.STRING)
@click.argument("src_index_pt_lookup",
                type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path))
@click.argument("field_lookup_index", type=click.STRING)
@click.argument("field_lookup_x", type=click.STRING)
@click.argument("field_lookup_y", type=click.STRING)
@click.argument("dst_name", type=click.STRING)
@click.option("--pts_index_start", type=click.INT, default=-1, show_default=True)
@click.option("--pts_index_end", type=click.INT, default=-1, show_default=True)
def main(src_nodes: Path, field_index_source: str, field_index_target: str, field_cost: str, src_pts: Path,
         layer_pts_source: str, layer_pts_target: str, field_pt_index: str, field_pt_group: str,
         src_index_pt_lookup: Path, field_lookup_index: str, field_lookup_x: str, field_lookup_y: str, dst_name: str,
         pts_index_start: int = -1, pts_index_end: int = -1) -> None:
    """
    \b
    Description: Creates an igraph directed Graph from a set of node index pairs as edges, with associated cost values.
    For each permutation of source and target points within each named group of source and target points, calculates
    the least-cost path along the Graph, using the cost values as the weight. Outputs a new layer to GeoPackage
    'least_cost_paths.gpkg', based on a given name, within the same directory as `src_nodes` containing the following
    attributes for each least-cost path:
        - group: Group name of source - target point pair.
        - source: Index of source point.
        - target: Index of target point.
        - cost: Sum of weights for least-cost path.
        - distance: Least-cost path geometry length.
        - geometry: Least-cost path geometry (LineString).

    \b
    Assumptions:
        - All spatial data is in the same, meter-based projection.
        - All source and target points match node coordinates added to the Graph.
        - All source and target points have named groups that exist in the other.
        - All permutations of source - target pairs can be reached using the Graph.
        - The collection of node pairs added to the Graph as edges does not contain any negative cost values.
        - All input CSVs have headers as the first row and comma (,) as the delimiter.

    \b
    :param Path src_nodes: CSV (.csv) containing each pair of source (from) and target (to) node indexes and the
        associated cost value to be added to a Graph as edges.
    :param str field_index_source: CSV field containing source node indexes.
    :param str field_index_target: CSV field containing target node indexes.
    :param str field_cost: CSV field containing cost values for each node index pair.
    :param Path src_pts: GeoPackage (.gpkg) containing source and target point layers.
    :param str layer_pts_source: GeoPackage layer containing source point geometries, associated node indexes, and
        names used to group sets of points.
    :param str layer_pts_target: GeoPackage layer containing target point geometries, associated node indexes, and
        names used to group sets of points.
    :param str field_pt_index: Field containing geometry-node indexes for source and target point layers.
    :param str field_pt_group: Field containing point group names for source and target point layers.
    :param Path src_index_pt_lookup: CSV (.csv) containing lookup data for node indexes and their geometry coordinates.
    :param str field_lookup_index: Lookup field containing node indexes.
    :param str field_lookup_x: Lookup field containing node longitude (x) values.
    :param str field_lookup_y: Lookup field containing node latitude (y) values.
    :param str dst_name: Output GeoPackage layer name.
    :param int pts_index_start: Starting index of grouped point pairs to be processed (inclusive; allows subsetting of
        results).
    :param int pts_index_end: Ending index of grouped point pairs to be processed (inclusive; allows subsetting of
        results).
    """

    try:

        least_cost_paths = LeastCostPaths(src_nodes, field_index_source, field_index_target, field_cost, src_pts,
                                          layer_pts_source, layer_pts_target, field_pt_index, field_pt_group,
                                          src_index_pt_lookup, field_lookup_index, field_lookup_x, field_lookup_y,
                                          dst_name, pts_index_start, pts_index_end)
        least_cost_paths()

    except KeyboardInterrupt:
        logger.exception("KeyboardInterrupt: Exiting program.")
        sys.exit(1)


if __name__ == "__main__":
    main()
