from functools import partial

from qdax.core.containers.mapelites_repertoire import compute_euclidean_centroids

from ribs.archives import GridArchive


def get_qdax_euclidean_centroids(archive):
    grid_shape = archive.dims
    minval = archive._lower_bounds
    maxval = archive._upper_bounds

    centroids = compute_euclidean_centroids(
        grid_shape, minval, maxval
    )

    return centroids

def archive_to_centroids(archive):
    if isinstance(archive, GridArchive):
        centroids = get_qdax_euclidean_centroids(archive)
    else:
        raise NotImplementedError("Only GridArchive is supported")

    return centroids