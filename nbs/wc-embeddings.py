import datetime
import os

import boto3
import geopandas as gp
import numpy as np
import pyarrow as pa
import rasterio
import shapely
import torch
import yaml
from box import Box
from rasterio.errors import RasterioIOError
from stacchip.processors.prechip import normalize_latlon, normalize_timestamp
from torchvision.transforms import v2

from src.model_clay_v1 import ClayMAEModule

BANDS = ["blue", "green", "red", "nir"]
DEVICE = "cpu"
ROWS_PER_BATCH = 1000

os.environ["CPL_TMPDIR"] = "/tmp"
os.environ["GDAL_CACHEMAX"] = "75%"
os.environ["GDAL_INGESTED_BYTES_AT_OPEN"] = "32768"
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
os.environ["GDAL_HTTP_MERGE_CONSECUTIVE_RANGES"] = "YES"
os.environ["GDAL_HTTP_MULTIPLEX"] = "YES"
os.environ["GDAL_HTTP_VERSION"] = "2"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["VSI_CACHE"] = "TRUE"


def load_model(ckpt: str):
    torch.set_default_device(DEVICE)

    model = ClayMAEModule.load_from_checkpoint(
        ckpt, metadata_path="configs/metadata.yaml", shuffle=False, mask_ratio=0
    )
    model.eval()

    return model.to(DEVICE)


def embed_chip(model, path):
    # Extract mean, std, and wavelengths from metadata
    metadata = Box(yaml.safe_load(open("configs/metadata.yaml")))

    mean = list(metadata["sentinel-2-l2a"].bands.mean[band] for band in BANDS)
    std = list(metadata["sentinel-2-l2a"].bands.std[band] for band in BANDS)
    waves = list(metadata["sentinel-2-l2a"].bands.wavelength[band] for band in BANDS)

    transform = v2.Compose(
        [
            v2.Normalize(mean=mean, std=std),
        ]
    )

    # Prep pixels
    with rasterio.open(path) as rst:
        chip = rst.read([1, 2, 3, 4]).astype("float32")
        bounds = shapely.box(*rst.bounds).wkt
    pixels = transform(np.expand_dims(chip, axis=0))

    # Prep datetimes embedding
    times = [normalize_timestamp(datetime.datetime(2022, 8, 1))]
    week_norm = [dat[0] for dat in times]
    hour_norm = [dat[1] for dat in times]

    # Prep lat/lon embedding
    latlons = [normalize_latlon(bounds)]
    lat_norm = [dat[0] for dat in latlons]
    lon_norm = [dat[1] for dat in latlons]

    # Prepare additional information
    datacube = {
        "platform": "sentinel-2-l2a",
        "time": torch.tensor(
            np.hstack((week_norm, hour_norm)),
            dtype=torch.float32,
            device=DEVICE,
        ),
        "latlon": torch.tensor(
            np.hstack((lat_norm, lon_norm)), dtype=torch.float32, device=DEVICE
        ),
        "pixels": torch.tensor(pixels, dtype=torch.float32, device=DEVICE),
        "gsd": torch.tensor(10, device=DEVICE),
        "waves": torch.tensor(waves, device=DEVICE),
    }
    with torch.no_grad():
        unmsk_patch, unmsk_idx, msk_idx, msk_matrix = model.model.encoder(datacube)

    for key, val in datacube.items():
        if key != "platform":
            val.detach()

    embedding = unmsk_patch[0, 0, :].cpu().numpy()

    unmsk_patch.detach()
    unmsk_idx.detach()
    msk_idx.detach()
    msk_matrix.detach()

    return embedding


def process():
    model = load_model(
        "/home/tam/pCloudDrive/DevSeed/clay/mae_v0.5.7_epoch-13_val-loss-0.3098.ckpt"
    )
    # model = load_model("s3://clay-model-ckpt/v0.5.7/mae_v0.5.7_epoch-13_val-loss-0.3098.ckpt")

    df = gp.read_file(
        "s3://clay-california-worldcover-rgbnir-vvvh-chips/california-worldcover-chips.fgb"
    )

    index = int(os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX", 42))
    embeddings = []
    chips = []
    idx_min = ROWS_PER_BATCH * index
    idx_max = min(len(df), ROWS_PER_BATCH * (index + 1))
    for i in range(idx_min, idx_max):
        print(f"Row {i}")
        row = df.iloc[i]
        chip = f"s3://clay-california-worldcover-rgbnir-vvvh-chips/chips/worldcover_california_chip_{row.col}_{row.row}.tif"
        try:
            embedding = embed_chip(model, chip)
        except RasterioIOError:
            print(f"Chip not found {chip}")
            continue

        embeddings.append(embedding)
        chips.append(chip)

    if not len(embeddings):
        print("No embeddings created")
        return

    # Add embeddings to index
    arrays = [
        pa.array(chips),
        pa.FixedShapeTensorArray.from_numpy_ndarray(np.array(embeddings)),
    ]
    table = pa.Table.from_arrays(arrays, names=["chips", "embeddings"])

    writer = pa.BufferOutputStream()
    pa.parquet.write_table(table, writer)
    body = bytes(writer.getvalue())
    s3 = boto3.resource("s3")
    s3_bucket = s3.Bucket(name="clay-california-worldcover-rgbnir-vvvh-chips")
    s3_bucket.put_object(
        Body=body,
        Key=f"embeddings-v1/embeddings-v1_{index}.parquet",
    )


if __name__ == "__main__":
    process()
