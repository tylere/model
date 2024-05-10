import numpy as np
import pyarrow as pa
from geoarrow.pyarrow import io
from pyarrow import dataset as ds
from pyarrow.parquet import read_table
from sklearn import decomposition

# Download all parquet files into a local folder, then use
# the following lines to combine all indices into one.
part = ds.partitioning(field_names=["item_id"])
data = ds.dataset(
    "/home/tam/Desktop/clay-naip-california-embeddings",
    format="parquet",
    partitioning=part,
)
ds.write_dataset(
    data,
    "/home/tam/Desktop/clay-mojave-full-index",
    format="parquet",
)

# Read the combined table, but only the embeddings (needs tons of ram)
table = read_table(
    "/home/tam/Desktop/clay-mojave-full-index/part-0.parquet", columns=["embeddings"]
)
emb = table["embeddings"].to_numpy()

# Convert to numpy
embeddings = np.array([_ for _ in emb])
del emb

# Randomly sample 1M embeddings
sample = embeddings[np.random.choice(embeddings.shape[0], 1000000, replace=False)]

# Compute PCA
pca = decomposition.PCA(n_components=1)
pca_result = pca.fit_transform(sample)
result = pca.transform(embeddings)

# Write PCA and geometries to  new file
table = read_table(
    "/home/tam/Desktop/clay-mojave-full-index/part-0.parquet", columns=["geometry"]
)
table = table.append_column("pca", pa.array(result.ravel()))
io.write_geoparquet_table(table, "/home/tam/Desktop/clay-v1-mojave-pca.parquet")
