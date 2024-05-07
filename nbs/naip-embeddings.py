import geopandas as gp
import numpy as np
import torch
import yaml
from box import Box
from pystac_client import Client
from stacchip.chipper import Chipper
from stacchip.indexer import NoStatsChipIndexer
from stacchip.processors.prechip import normalize_latlon, normalize_timestamp
from torchvision.transforms import v2

from src.model_clay_v1 import ClayMAEModule

STAC_API = "https://planetarycomputer.microsoft.com/api/stac/v1"
BUCKET = "test"
YEAR = 2022


def get_item(idx: int):
    df = gp.read_file(
        "https://clay-mgrs-samples.s3.amazonaws.com/naip_california_quads.fgb"
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

    return items[0]


def load_model(ckpt="/home/tam/Downloads/mae_v0.53_last.ckpt", device="cpu"):
    torch.set_default_device(device)

    model = ClayMAEModule.load_from_checkpoint(
        ckpt, metadata_path="configs/metadata.yaml", shuffle=False, mask_ratio=0
    )
    model.eval()

    return model.to(device)


def process(idx, device="cpu"):
    item = get_item(idx)
    model = load_model()

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

    # Remove unused assets
    keys = list(item.assets.keys())
    for key in keys:
        if key != "image":
            del item.assets[key]

    # Create indexer and index
    indexer = NoStatsChipIndexer(item)
    index = indexer.create_index()

    # For each chip create embeddings and store chip
    embeddings = []
    datacubes = []
    for i in range(len(index)):
        chipper = Chipper(
            platform="naip",
            item_id="granule",
            chip_index_x=index["chip_index_x"][i].as_py(),
            chip_index_y=index["chip_index_y"][i].as_py(),
            indexer=indexer,
            bucket="placeholder",
        )
        chip = chipper.chip["image"]

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
                device=device,
            ),
            "latlon": torch.tensor(
                np.hstack((lat_norm, lon_norm)), dtype=torch.float32, device=device
            ),
            "pixels": torch.tensor(pixels, dtype=torch.float32, device=device),
            "gsd": torch.tensor(item.properties["gsd"], device=device),
            "waves": torch.tensor(waves, device=device),
        }

        with torch.no_grad():
            unmsk_patch, unmsk_idx, msk_idx, msk_matrix = model.model.encoder(datacube)

        # The first embedding is the class token, which is the
        # overall single embedding.
        embeddings.append(unmsk_patch[0, 0, :].cpu().numpy())
        datacubes.append(datacube)
        break

    np.savez_compressed(
        f"/home/tam/Desktop/clay-v1-data-chips-naip/chip_naip_row_{idx}_chip_{i}.npz",
        datacubes=datacubes,
        embeddings=embeddings,
    )


process(23)
