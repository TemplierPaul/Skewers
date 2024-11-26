"""Core components of the MAP-Elites algorithm."""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.custom_types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Metrics,
    RNGKey,
)

from qdax.core.map_elites import MAPElites as QDaxMAPElites

from skewers.translation import archive_to_centroids
from skewers.reshaper import build_reshaper

class SkewersEvaluator:
    def __init__(
        self,
        scoring_function: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ],
        metrics_function: Callable[[MapElitesRepertoire], Metrics],
        archive: Any,
        init_variables: Any
    ) -> None:
        
        self.scoring_function = jax.jit(scoring_function)
        self.metrics_function = metrics_function
        self.centroids = archive_to_centroids(archive)
        self.random_key = None
        self.repertoire = None
        self.reshaper = build_reshaper(init_variables)

        # retrieve one genotype from the population
        first_genotype = jax.tree_util.tree_map(lambda x: x[0], init_variables)

        self.repertoire = MapElitesRepertoire.init_default(
            genotype=first_genotype,
            centroids=self.centroids,
        )

    def evaluate(
            self, 
            genotypes: Genotype,
    ):            
        genotypes = self.reshaper.unflatten(genotypes)
        fitnesses, descriptors, extra_scores, self.random_key = self.scoring_function(
            genotypes, self.random_key
        )
        self.repertoire = self.repertoire.add(genotypes, descriptors, fitnesses, extra_scores)

        fitnesses = np.array(fitnesses)
        descriptors = np.array(descriptors)
        return fitnesses, descriptors

