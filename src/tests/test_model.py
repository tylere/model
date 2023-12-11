"""
Tests for neural network model architecture.

Based loosely on Lightning's testing method described at
https://github.com/Lightning-AI/lightning/blob/2.1.0/.github/CONTRIBUTING.md#how-to-add-new-tests
"""
import os
import tempfile

import geopandas as gpd
import lightning as L
import numpy as np
import pytest
import torch
import torchdata
import torchdata.dataloader2

from src.model_vit import ViTLitModule


# %%
@pytest.fixture(scope="function", name="datapipe")
def fixture_datapipe() -> torchdata.datapipes.iter.IterDataPipe:
    """
    A torchdata DataPipe with random data to use in the tests.
    """
    datapipe = torchdata.datapipes.iter.IterableWrapper(
        iterable=[
            {
                "image": torch.randn(2, 13, 512, 512).to(dtype=torch.float16),
                "bbox": torch.tensor(
                    data=[
                        [499975.0, 3397465.0, 502535.0, 3400025.0],
                        [530695.0, 3397465.0, 533255.0, 3400025.0],
                    ]
                ),
                "date": ["2020-01-01", "2020-12-31"],
                "epsg": torch.tensor(data=[32646, 32646]),
            },
        ]
    )
    return datapipe


# %%
def test_model_vit(datapipe):
    """
    Run a full train, val, test and prediction loop using 1 batch.
    """
    # Get some random data
    dataloader = torchdata.dataloader2.DataLoader2(datapipe=datapipe)

    # Initialize model
    model: L.LightningModule = ViTLitModule()

    # Run tests in a temporary folder
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Training
        trainer: L.Trainer = L.Trainer(
            accelerator="auto",
            devices=1,
            precision="16-mixed",
            fast_dev_run=True,
            default_root_dir=tmpdirname,
        )
        trainer.fit(model=model, train_dataloaders=dataloader)

        # Prediction
        trainer.predict(model=model, dataloaders=dataloader)
        assert os.path.exists(path := f"{tmpdirname}/data/embeddings/embeddings_0.gpq")
        geodataframe: gpd.GeoDataFrame = gpd.read_parquet(path=path)

        assert geodataframe.shape == (2, 3)
        assert all(geodataframe.columns == ["date", "embeddings", "geometry"])
        assert geodataframe.date.dtype == "date32[day][pyarrow]"
        assert geodataframe.embeddings.dtype == "object"
        assert geodataframe.geometry.dtype == gpd.array.GeometryDtype()

        for embeddings in geodataframe.embeddings:
            assert embeddings.shape == (768,)
            assert embeddings.dtype == "float32"
            assert not np.isnan(embeddings).any()
