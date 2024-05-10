import os
import tempfile
from io import BytesIO
from pathlib import Path
import sys

import boto3
import geopandas as gp
import numpy as np
import pyarrow as pa
import rasterio
import requests
import torch
import yaml
from box import Box
from geoarrow.pyarrow import io
from pystac_client import Client
from rasterio.io import MemoryFile
from stacchip.chipper import Chipper
from stacchip.indexer import NoStatsChipIndexer
from stacchip.processors.prechip import normalize_latlon, normalize_timestamp
from torchvision.transforms import v2

from src.model_clay_v1 import ClayMAEModule

STAC_API = "https://planetarycomputer.microsoft.com/api/stac/v1"
BUCKET = "clay-v1-california-chips"
YEAR = 2022
DEVICE = "cuda"

os.environ["CPL_TMPDIR"] = "/tmp"
os.environ["GDAL_CACHEMAX"] = "75%"
os.environ["GDAL_INGESTED_BYTES_AT_OPEN"] = "32768"
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
os.environ["GDAL_HTTP_MERGE_CONSECUTIVE_RANGES"] = "YES"
os.environ["GDAL_HTTP_MULTIPLEX"] = "YES"
os.environ["GDAL_HTTP_VERSION"] = "2"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["VSI_CACHE"] = "TRUE"


def get_item(idx: int, tmpdir: Path):
    df = gp.read_file(
        "https://clay-mgrs-samples.s3.amazonaws.com/naip-mojave-desert.fgb"
        # "https://clay-mgrs-samples.s3.amazonaws.com/naip_california_quads.fgb"
    )

    row = df.iloc[idx]

    client = Client.open(STAC_API)
    date_range = f"{YEAR}-01-01T00:00:00Z/{YEAR+1}-01-01T00:00:00Z"
    search = client.search(
        collections=["naip"],
        filter_lang="cql2-json",
        filter={
            "op": "and",
            "args": [
                {
                    "op": "s_intersects",
                    "args": [
                        {"property": "geometry"},
                        row.geometry.centroid.__geo_interface__,
                    ],
                },
                {"op": "anyinteracts", "args": [{"property": "datetime"}, date_range]},
                {"op": "like", "args": [{"property": "id"}, "ca_%"]},
            ],
        },
    )

    items = search.get_all_items()

    if len(items) != 1:
        raise ValueError(f"Found more than one item for row {row}")

    item = items[0]

    # Remove unused assets
    keys = list(item.assets.keys())
    for key in keys:
        if key != "image":
            del item.assets[key]

    # Download naip file for speed
    name = tmpdir / "naip.tif"
    download_file(item.assets["image"].href, name)
    item.assets["image"].href = name

    return item


def download_file(url: str, name: str):
    print(f"Downloading file {url}")
    with requests.get(url) as r:
        r.raise_for_status()
        with open(name, "wb") as f:
            f.write(r.content)


def load_model(ckpt: str):
    torch.set_default_device(DEVICE)

    model = ClayMAEModule.load_from_checkpoint(
        ckpt, metadata_path="configs/metadata.yaml", shuffle=False, mask_ratio=0
    )
    model.eval()

    return model.to(DEVICE)


def write_tiff(chip, meta, x, y, item_id):
    with MemoryFile() as memfile:
        with memfile.open(**meta, compress="deflate") as dst:
            dst.write(chip)
        memfile.seek(0)
        s3 = boto3.resource("s3")
        s3_bucket = s3.Bucket(name=BUCKET)
        new_key = f"chips/{item_id}/chip_{x}_{y}.tif"
        print(f"Writing {new_key}")
        s3_bucket.put_object(
            Key=new_key,
            Body=memfile.read(),
        )


def process(idx, chkpt, tmpdir):
    model = load_model(ckpt=chkpt)
    item = get_item(idx=idx, tmpdir=tmpdir)

    # Extract mean, std, and wavelengths from metadata
    metadata = Box(yaml.safe_load(open("configs/metadata.yaml")))

    mean = list(metadata["naip"].bands.mean.values())
    std = list(metadata["naip"].bands.std.values())
    waves = list(metadata["naip"].bands.wavelength.values())

    transform = v2.Compose(
        [
            v2.Normalize(mean=mean, std=std),
        ]
    )

    # Create indexer and index
    indexer = NoStatsChipIndexer(item)
    index = indexer.create_index()

    with rasterio.open(tmpdir / "naip.tif") as rst:
        meta = rst.meta.copy()
        meta["width"] = 256
        meta["height"] = 256

    # For each chip create embeddings and store chip
    embeddings = []
    for i in range(len(index)):
        x = index["chip_index_x"][i].as_py()
        y = index["chip_index_y"][i].as_py()
        print(f"Processing {item.id} chip {x} {y}")
        chipper = Chipper(
            platform="naip",
            item_id="granule",
            chip_index_x=x,
            chip_index_y=y,
            indexer=indexer,
            bucket="non-existing-bucket",
        )
        chip = chipper.chip["image"]

        trsf = list(meta["transform"])
        trsf[2] = (
            chipper.indexer.bbox[0]
            + x * chipper.indexer.transform[0] * chipper.indexer.chip_size
        )
        trsf[5] = (
            chipper.indexer.bbox[3]
            + (y + 1) * chipper.indexer.transform[4] * chipper.indexer.chip_size
        )

        meta["transform"] = rasterio.Affine(*trsf)
        write_tiff(chip, meta, x, y, item.id)

        # Prep pixels
        pixels = transform(np.expand_dims(chip, axis=0))

        # Prep datetimes embedding
        times = [normalize_timestamp(item.datetime)]
        week_norm = [dat[0] for dat in times]
        hour_norm = [dat[1] for dat in times]

        # Prep lat/lon embedding
        geom = index["geometry"][i].wkt
        latlons = [normalize_latlon(geom)]
        lat_norm = [dat[0] for dat in latlons]
        lon_norm = [dat[1] for dat in latlons]

        # Prepare additional information
        datacube = {
            "platform": "naip",
            "time": torch.tensor(
                np.hstack((week_norm, hour_norm)),
                dtype=torch.float32,
                device=DEVICE,
            ),
            "latlon": torch.tensor(
                np.hstack((lat_norm, lon_norm)), dtype=torch.float32, device=DEVICE
            ),
            "pixels": torch.tensor(pixels, dtype=torch.float32, device=DEVICE),
            "gsd": torch.tensor(item.properties["gsd"], device=DEVICE),
            "waves": torch.tensor(waves, device=DEVICE),
        }

        with torch.no_grad():
            unmsk_patch, unmsk_idx, msk_idx, msk_matrix = model.model.encoder(datacube)

        # The first embedding is the class token, which is the
        # overall single embedding.
        embeddings.append(unmsk_patch[0, 0, :].cpu().numpy())

        # Detach tensors
        for key, val in datacube.items():
            if key != "platform":
                val.detach()

        unmsk_patch.detach()
        unmsk_idx.detach()
        msk_idx.detach()
        msk_matrix.detach()
    
    del model

    # Add embeddings to index
    print("Done processing tiles, adding embedddings to index")
    index = index.append_column(
        "embeddings", pa.FixedShapeTensorArray.from_numpy_ndarray(np.array(embeddings))
    )

    # Centralize the index files to make combining them easier later on
    print("Writing index to file")
    writer = pa.BufferOutputStream()
    io.write_geoparquet_table(index, writer)
    body = bytes(writer.getvalue())

    print(f"Uploading index to {BUCKET} using upload_fileobj")
    s3 = boto3.client("s3")
    with BytesIO(body) as fl:
        s3.upload_fileobj(fl, BUCKET, f"index/{item.id}/index_{item.id}.parquet")
    print("Finished Uploading parquet file")


def main_process():
    chkpt = "s3://clay-model-ckpt/v0.5.7/mae_v0.5.7_epoch-13_val-loss-0.3098.ckpt"
    index = int(os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX", 23))
    with tempfile.TemporaryDirectory() as tmpdir:
        process(index, chkpt, Path(tmpdir))
    print("Finished processing")
    raise SystemError("This is for the batch job to exit, otherwise it hangs.")


if __name__ == "__main__":
    main_process()
