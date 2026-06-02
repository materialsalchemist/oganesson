import numpy as np
from operator import itemgetter
from oganesson.ogstructure import OgStructure
from ase_ga.offspring_creator import OffspringCreator
from ase_ga.utilities import get_distance_matrix
from ase_ga.particle_mutations import Mutation
from ase import Atoms

def get_nndist(atoms, distance_matrix, rmax):
    """Returns an estimate of the nearest neighbor bond distance
    in the supplied atoms object given the supplied distance_matrix.

    The estimate comes from the first peak in the radial distribution
    function.
    """
    nbins = 200
    og = OgStructure(atoms)
    rdf, dists = og.get_rdf(rmax, nbins, distance_matrix)
    return dists[np.argmax(rdf)]

class _NeighborhoodPermutation(Mutation):
    """Helper class that holds common functions to all permutations
    that look at the neighborhoods of each atoms."""
    @classmethod
    def get_possible_poor2rich_permutations(cls, atoms, inverse=False,
                                            recurs=0, distance_matrix=None, rmax=10):
        dm = distance_matrix
        if dm is None:
            dm = get_distance_matrix(atoms)
        # Adding a small value (0.2) to overcome slight variations
        # in the average bond length
        nndist = get_nndist(atoms, dm, rmax) + 0.2
        same_neighbors = {}

        def f(x):
            return x[1]
        for i, atom in enumerate(atoms):
            same_neighbors[i] = 0
            neighbors = [j for j in range(len(dm[i])) if dm[i][j] < nndist]
            for n in neighbors:
                if atoms[n].symbol == atom.symbol:
                    same_neighbors[i] += 1
        sorted_same = sorted(same_neighbors.items(), key=f)
        if inverse:
            sorted_same.reverse()
        poor_indices = [j[0] for j in sorted_same
                        if abs(j[1] - sorted_same[0][1]) <= recurs]
        rich_indices = [j[0] for j in sorted_same
                        if abs(j[1] - sorted_same[-1][1]) <= recurs]
        permuts = Mutation.get_list_of_possible_permutations(atoms,
                                                             poor_indices,
                                                             rich_indices)

        if len(permuts) == 0:
            _NP = _NeighborhoodPermutation
            return _NP.get_possible_poor2rich_permutations(atoms, inverse,
                                                           recurs + 1, dm, rmax)
        return permuts


class Poor2richPermutation(_NeighborhoodPermutation):
    """The poor to rich (Poor2rich) permutation operator described in
    S. Lysgaard et al., Top. Catal., 2014, 57 (1-4), pp 33-39

    Permutes two atoms from regions short of the same elements, to
    regions rich in the same elements.
    (Inverse of Rich2poorPermutation)

    Parameters:

    elements: Which elements to take into account in this permutation

    rng: Random number generator
        By default numpy.random.
    """

    def __init__(self, elements=[], num_muts=1, rng=np.random, rmax=10):
        _NeighborhoodPermutation.__init__(self, num_muts=num_muts, rng=rng)
        self.descriptor = 'Poor2richPermutation'
        self.elements = elements
        self.rmax = rmax

    def get_new_individual(self, parents):
        f = parents[0].copy()

        diffatoms = len(set(f.numbers))
        assert diffatoms > 1, 'Permutations with one atomic type is not valid'

        indi = self.initialize_individual(f)
        indi.info['data']['parents'] = [f.info['confid']]

        for _ in range(self.num_muts):
            Poor2richPermutation.mutate(f, self.elements, rng=self.rng, rmax=self.rmax)

        for atom in f:
            indi.append(atom)

        return (self.finalize_individual(indi),
                self.descriptor + ':Parent {0}'.format(f.info['confid']))

    @classmethod
    def mutate(cls, atoms, elements, rng=np.random, rmax=10):
        _NP = _NeighborhoodPermutation
        # indices = [a.index for a in atoms if a.symbol in elements]
        ac = atoms.copy()
        del ac[[atom.index for atom in ac
                if atom.symbol not in elements]]
        permuts = _NP.get_possible_poor2rich_permutations(ac, rmax=rmax)
        swap = list(rng.choice(permuts))
        atoms.symbols[swap] = atoms.symbols[swap[::-1]]


class Rich2poorPermutation(_NeighborhoodPermutation):
    """
    The rich to poor (Rich2poor) permutation operator described in
    S. Lysgaard et al., Top. Catal., 2014, 57 (1-4), pp 33-39

    Permutes two atoms from regions rich in the same elements, to
    regions short of the same elements.
    (Inverse of Poor2richPermutation)

    Parameters:

    elements: Which elements to take into account in this permutation

    rng: Random number generator
        By default numpy.random.
    """

    def __init__(self, elements=None, num_muts=1, rng=np.random, rmax=10):
        _NeighborhoodPermutation.__init__(self, num_muts=num_muts, rng=rng)
        self.descriptor = 'Rich2poorPermutation'
        self.elements = elements
        self.rmax = rmax

    def get_new_individual(self, parents):
        f = parents[0].copy()

        diffatoms = len(set(f.numbers))
        assert diffatoms > 1, 'Permutations with one atomic type is not valid'

        indi = self.initialize_individual(f)
        indi.info['data']['parents'] = [f.info['confid']]

        if self.elements is None:
            elems = list(set(f.get_chemical_symbols()))
        else:
            elems = self.elements
        for _ in range(self.num_muts):
            Rich2poorPermutation.mutate(f, elems, rng=self.rng, rmax=self.rmax)

        for atom in f:
            indi.append(atom)

        return (self.finalize_individual(indi),
                self.descriptor + ':Parent {0}'.format(f.info['confid']))

    @classmethod
    def mutate(cls, atoms, elements, rng=np.random, rmax=10):
        _NP = _NeighborhoodPermutation
        ac = atoms.copy()
        del ac[[atom.index for atom in ac
                if atom.symbol not in elements]]
        permuts = _NP.get_possible_poor2rich_permutations(ac,
                                                          inverse=True, rmax=rmax)
        swap = list(rng.choice(permuts))
        atoms.symbols[swap] = atoms.symbols[swap[::-1]]
