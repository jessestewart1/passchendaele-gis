import click
import geopandas as gpd
import logging
import pandas as pd
import rasterio as rio
import sys
from itertools import chain
from operator import itemgetter
from pathlib import Path
from shapely import box, Point

# Set logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
logger.addHandler(handler)


class TopographicProfile:
    """Defines the TopographicProfile class."""

    def __init__(self, src_lines: Path, src_layer: str, src_name: str, src_rasters: Path) -> None:
        """Initializes the TopographicProfile class."""

        self.src_lines = src_lines
        self.src_lines_layer = src_layer
        self.src_lines_name = src_name
        self.src_rasters = src_rasters
        self.dst = self.src_lines.parent / "topographic_profile"
        self.crs = "EPSG:3043"
        self.nodata = -99999

        # Compile src data.
        logger.info(f"Compiling source data - LineStrings: {self.src_lines}, layer={self.src_lines_layer}")
        self.lines = gpd.read_file(self.src_lines, layer=self.src_lines_layer)

        logger.info(f"Compiling source data - rasters: {self.src_rasters}")
        self.rasters = pd.Series(map(lambda src: rio.open(src), self.src_rasters.glob("*.tif")))

        # Create output directory.
        logger.info(f"Creating output: {self.dst}")
        if not self.dst.exists():
            self.dst.mkdir(parents=True)

    def __call__(self) -> None:
        """Executes the TopographicProfile class."""

        self.interpolate_points()
        self.sample_rasters()
        self.gen_output()

    def gen_output(self) -> None:
        """
        Generates an output .csv and GeoPackage layer for each input LineString with the following output columns:
            'distance': distance (meters) along the input LineString for each interpolated point.
            'elevation': elevation (meters asl) for each interpolated point.
            'geometry': Point geometry (EPSG:3043; only for GeoPackage).
        """

        logger.info("Generating output.")

        # Iterate LineString data:
        for name in set(self.lines[self.src_lines_name]):

            # Format output.
            elevation, g, points = self.lines.loc[self.lines[self.src_lines_name] == name,
                                                  ["elevation", "geometry", "points"]].iloc[0]
            output = pd.DataFrame({"distance": (*tuple(map(lambda idx: idx * 10, range(len(elevation)-1))), g.length),
                                   "elevation": elevation})

            # Round data.
            output["elevation"] = output["elevation"].round(1)
            output["distance"] = output["distance"].astype(int)
            if output["distance"].iloc[-2] == output["distance"].iloc[-1]:
                output.loc[output.index == output.index[-1], "distance"] += 1

            # Construct output names.
            name = name.lower().replace(" ", "_")
            dst_csv = self.dst / f"topographic_profile_{name}.csv"
            dst_gpkg = self.dst / f"topographic_profile.gpkg"
            dst_layer = f"topographic_profile_{name}"

            # Export .csv.
            output.to_csv(dst_csv, sep=",", header=True, index=False)

            # Compile Point geometry and export GeoPackage layer.
            df = gpd.GeoDataFrame(output, geometry=list(map(Point, points)), crs=self.crs)
            df.to_file(str(dst_gpkg), layer=dst_layer)

            logger.info(f"Successfully output file: {dst_csv}.")
            logger.info(f"Successfully output file: {dst_gpkg}|layer={dst_layer}.")

    def interpolate_points(self) -> None:
        """
        For each LineString, generates a sequence of points at every 10-meter interval, including the first and last
        point in the LineString.
        """

        logger.info("Interpolating points along LineStrings.")

        # Interpolate points.
        self.lines["points"] = self.lines["geometry"].map(
            lambda g: (*tuple(map(lambda dist: g.interpolate(dist * 10).coords[0], range(int(g.length / 10) + 1))),
                       g.coords[-1]))

    def sample_rasters(self) -> None:
        """
        For each point, retrieves the intersecting cell value from each raster, keeping only the first non-nodata value.
        """

        logger.info("Sampling rasters - preprocessing.")

        # Compile unique set of points as Series.
        points = pd.Series(tuple(set(chain.from_iterable(self.lines["points"]))))

        # Compile bounding boxes for rasters.
        bbox = gpd.GeoSeries(self.rasters.map(lambda r: box(*r.bounds)))

        # Generate bounding box index - raster lookup.
        bbox_raster_lookup = dict(zip(bbox.index, self.rasters))

        # Get index(es) of intersecting bounding box for each point.
        points_bbox_index = points.map(lambda pt: tuple(bbox.loc[bbox.intersects(Point(pt))].index))

        # Compile sampling input data as Series (so `map` can be used instead of `apply`).
        point_data = pd.DataFrame({"point": points, "bbox_index": points_bbox_index}).apply(dict, axis=1)

        # Create nodata Series for results.
        point_elevation = pd.Series([self.nodata] * len(point_data), dtype="Float64")

        logger.info(f"Sampling {len(points)} points against {len(self.rasters)} rasters.")

        # Sample raster associated with each point.
        # Note: Process is iterative since point - bounding box intersection may return multiple results, some of which
        # may be areas of nodata.
        remainder = (point_elevation == self.nodata)
        for idx in range(max(points_bbox_index.map(len))):

            point_elevation.loc[remainder] = point_data.loc[remainder].map(
                lambda data: bbox_raster_lookup[data["bbox_index"][idx]].sample((data["point"],))).map(
                lambda results: tuple(results)[0][0]
            )

            # Check for remaining nodata results.
            remainder = pd.Series(point_elevation == self.nodata)
            if not sum(remainder):
                break
            else:
                logger.info(f"Nodata value ({self.nodata}) sampled for {sum(remainder)} points. Resampling with next "
                            f"rasters.")

        logger.info("Assigning sampled values back to interpolated points.")

        # Assign sampled values to interpolated points using point - sampled value lookup.
        points_sample_lookup = dict(zip(points, point_elevation))
        self.lines["elevation"] = self.lines["points"].map(lambda pts: itemgetter(*pts)(points_sample_lookup))


@click.command()
@click.argument("src_lines", type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True))
@click.argument("src_layer", type=click.STRING)
@click.argument("src_name", type=click.STRING)
@click.argument("src_rasters", type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True))
def main(src_lines: Path, src_layer: str, src_name: str, src_rasters: Path) -> None:
    """
    \b
    Description: For one or more LineStrings representing the path of the desired topographic profile(s), a sequence of
    points is interpolated at every 10-meter interval, including the first and last points in the LineString. Then, for
    each point, the elevation value is retrieved from the cell of the DTM raster which the point intersects. For each
    input LineString, both a .csv and GeoPackage layer is output containing the following columns:
        - 'distance': the distance of each interpolated point along the LineString, including 0 and the distance of the
          final point in the LineString.
        - 'elevation': the elevation value of each point.
        - 'geometry': Point geometry (EPSG:3043; only for GeoPackage).
    The output files will be created in a new directory, 'topographic_profile', within the same directory containing
    the LineString source.

    \b
    Assumptions:
        - All data (LineString and rasters) is in the same, meter-based projection. This program assumes EPSG:3043.
        - All rasters are in GeoTIFF (.tif) format.
        - All rasters have the same nodata value as each other. This program assumed -99999.
        - The raster source directory contains rasters covering the entirety of the input LineString such that no
          point interpolated along the LineStrings will fail to intersect an input raster.

    \b
    :param Path src_lines: source GeoPackage.
    :param str src_layer: layer containing one or more LineStrings within the source GeoPackage.
    :param str src_name: layer column used to uniquely identify each row and form the name of the output .csv.
    :param Path src_rasters: directory containing one or more GeoTIFF DTM rasters.
    """

    try:

        topographic_profile = TopographicProfile(src_lines, src_layer, src_name, src_rasters)
        topographic_profile()

    except KeyboardInterrupt:
        logger.exception("KeyboardInterrupt: Exiting program.")
        sys.exit(1)


if __name__ == "__main__":
    main()
