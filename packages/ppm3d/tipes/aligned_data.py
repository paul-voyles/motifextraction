"""Datastructure for holding and working with the outputs of an alignment."""
import json
import numpy as np
from collections import Counter

from .lazy_property import lazyproperty
from .cluster import Positions, Cluster

# The model was aligned ONTO the target. So the model is modified, but the target isn't. I.e. R(M[mapping]) =~= T.
# The number of atoms in the model is >= the number of atoms in the target. This requires the a subset of the atoms in the model to be chosen for alignment.
# So the model moves, and the target stays the same.

class AlignedData(object):
    @classmethod
    def from_mapping(cls, mapping):
        return cls(R=mapping["R"], T=mapping["T"], mapping=mapping["mapping"], error=mapping["error_lsq"],
                   inverted=mapping["inverted"], swapped=mapping["swapped"],
                   model_file=mapping["model"], model_scale=mapping["model_rescale"],
                   target_file=mapping["target"], target_scale=mapping["target_rescale"],
                   aligned_model_coords=mapping["aligned_model"])

    @classmethod
    def from_json(cls, filename):
        return cls.from_mapping(json.load(open(filename)))

    def __init__(self, R, T, mapping, error, inverted, swapped,
                 model_file=None, model_coords=None, model_symbols=None, model_scale=1.0,
                 target_file=None, target_coords=None, target_symbols=None, target_scale=1.0,
                 aligned_model_coords=None, aligned_model_symbols=None):
        """
             model_file OR (model_coords AND model_symbols)
             target_file OR (target_coords AND target_symbols)
             aligned_model_coords, OPTIONAL aligned_model_symbols
        """
        self.R = np.matrix(R)
        self.T = np.matrix(T)

        self.mapping = np.array(mapping)
        self.error = error

        self._inverted = inverted
        self._swapped = swapped

        self._model_scale = model_scale
        self._target_scale = target_scale

        self._input_model_symbols = model_symbols
        self._input_model = Positions(model_coords) if model_coords else None
        if self._input_model is not None and len(self._input_model) == 3:
            self._input_model = self._input_model.T
        else:
            self._model_file = model_file

        self._input_target_symbols = target_symbols
        self._input_target = Positions(target_coords) if target_coords else None
        if self._input_target is not None and len(self._input_target) == 3:
            self._input_target = self._input_target.T
        else:
            self._target_file = target_file

        self._aligned_model_symbols = aligned_model_symbols
        self._aligned_model = Positions(aligned_model_coords)
        if len(self._aligned_model) == 3:
            self._aligned_model = self._aligned_model.T

        try:
            assert not np.isnan(self.error)
            self.validate_data()
        except AssertionError:
            if not np.isnan(self.error):
                print(f"WARNING! Data for {self.model.filename} and {self.target.filename} is incorrect!")
            self._successful = False

    @property
    def successful(self) -> bool:
        if hasattr(self, '_successful'):
            return self._successful  # TODO Remove this once I fix the validation issues.
        return max(Counter(self.mapping).values()) == 1

    @property
    def swapped(self) -> bool:
        return self._swapped

    @property
    def inverted(self) -> bool:
        return self._inverted

    @lazyproperty
    def _model(self):
        if self._model_file is None:
            positions = self._input_model
            symbols = self._input_model_symbols
            c = Cluster(positions=positions, symbols=symbols)
            del self._input_model
            del self._input_model_symbols
        else:
            c = Cluster(filename=self._model_file)
        return c

    @lazyproperty
    def _target(self):
        if self._target_file is None:
            c = Cluster(positions=self._input_target, symbols=self._input_target_symbols)
            del self._input_target
            del self._input_target_symbols
        else:
            c = Cluster(filename=self._target_file)
        return c

    @lazyproperty
    def model(self) -> Cluster:
        if not self.swapped:
            model = self._model
        else:
            model = self._target
        if self.inverted:
            model._positions = -model._positions
        return model

    @lazyproperty
    def target(self) -> Cluster:
        if not self.swapped:
            target = self._target
        else:
            target = self._model
        return target

    @lazyproperty
    def model_scale(self) -> float:
        if not self.swapped:
            return self._model_scale
        else:
            return self._target_scale

    @lazyproperty
    def target_scale(self) -> float:
        if not self.swapped:
            return self._target_scale
        else:
            return self._model_scale

    def _get_aligned_model_symbols(self):
        indices = np.argsort(self.mapping)
        if self._aligned_model_symbols is not None:
            return self._aligned_model_symbols
        elif hasattr(self, '_model_symbols') and self._input_model_symbols is not None:
            return list(np.array(self._input_model_symbols)[indices])
        else:
            return list(np.array(self.model.symbols)[indices])

    @lazyproperty
    def aligned_model(self) -> Cluster:
        c = Cluster(positions=self._aligned_model, symbols=self._get_aligned_model_symbols())
        del self._aligned_model
        return c

    @property
    def rescaled_target_positions(self) -> Positions:
        return self.target.positions / self.target_scale

    def align_model(self, rescale=True, apply_mapping=True) -> Positions:
        model = self.model.positions.copy()
        scale = self.model_scale
        if apply_mapping:
            use_this = self.mapping  # This is correct

            expected_mapping_size = min(len(self.model), len(self.target))
            assert len(use_this) == expected_mapping_size
            model = model[use_this]
        R = self.R
        T = self.T
        if rescale:
            model /= scale
            model = model.apply_transformation(R, T, invert=False)
        else:
            model = model.apply_transformation(R, T*scale, invert=False)
        return model

    def validate_data(self):
        # (R(M) == A) ~= T

        # These two should be identical
        m = self.aligned_model.positions
        t = self.align_model(rescale=True, apply_mapping=True)
        l2 = Cluster._L2Norm(t, m)
        # Sometimes this fails, and it always seems to be when the mapping assigns the center atom in the model to the
        #  an atom that _isn't_ the center atom in the target.
        # I'm happy just marking those alignments as "failed" because clearly the structures are so different that
        #  they shouldn't be considered similar.
        assert np.isclose(l2, 0)
        m = self.aligned_model.positions * self.model_scale
        t = self.align_model(rescale=False, apply_mapping=True)
        l2 = Cluster._L2Norm(t, m)
        assert np.isclose(l2, 0)

        # The L2 between the aligned data (ie aligned_model) and the (reorderd) target should be equal to the reported error
        m = self.aligned_model.positions
        t = self.rescaled_target_positions
        l2 = Cluster._L2Norm(t, m)
        assert np.isclose(l2, self.error)

        assert np.isclose(self.L2Norm(), self.error)

    def to_dict(self):
        d = {}
        d['R'] = self.R
        d['T'] = self.T
        d['mapping'] = self.mapping
        d['inverted'] = self.inverted
        d['error'] = self.error

        if hasattr(self, '_model_file'):
            d['model'] = self._model_file
        elif hasattr(self, '_model'):
            d['model'] = self._input_model
        else:
            d['model'] = self.model.to_dict()

        if hasattr(self, '_target_file'):
            d['target'] = self._target_file
        elif hasattr(self, '_target'):
            d['target'] = self._input_target
        else:
            d['target'] = self.target.to_dict()

        if hasattr(self, '_aligned_model'):
            d['aligned_model'] = self._aligned_model
        else:
            d['aligned_model'] = self.aligned_model.to_dict()
            del d['aligned_model']['symbols']
        return d

    def to_json(self, **kwargs):
        d = self.to_dict()
        for key, value in d.items():
            if isinstance(value, np.ndarray):
                d[key] = value.tolist()
        return json.dumps(d, **kwargs)

    def L2Norm(self):
        return Cluster._L2Norm(self.align_model(rescale=True, apply_mapping=True), self.rescaled_target_positions)

    def L2Norm2(self):
        return Cluster._L2Norm2(self.align_model(rescale=True, apply_mapping=True), self.rescaled_target_positions)

    def L1Norm(self):
        return Cluster._L1Norm(self.align_model(rescale=True, apply_mapping=True), self.rescaled_target_positions)

    def LinfNorm(self):
        return Cluster._LinfNorm(self.align_model(rescale=True, apply_mapping=True), self.rescaled_target_positions)

    def angular_variation(self, neighbor_cutoff):
        return Cluster._angular_variation(self.align_model(rescale=True, apply_mapping=True), self.rescaled_target_positions, neighbor_cutoff)
