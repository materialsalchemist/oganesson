from ase.io import write
from ase_ga import get_raw_score
from ase_ga.data import DataConnection
from ase_ga.population import Population
from ase_ga.utilities import closest_distances_generator, CellBounds
from ase_ga.ofp_comparator import OFPComparator
from ase_ga.offspring_creator import OperationSelector
from ase_ga.standardmutations import StrainMutation
from ase_ga.soft_mutation import SoftMutation
from ase_ga.cutandsplicepairing import CutAndSplicePairing
from oganesson.genetic_algorithms.particle_mutations import (
    Poor2richPermutation,
    Rich2poorPermutation,
)
from ase import Atoms
from ase.data import atomic_numbers
from ase_ga.startgenerator import StartGenerator
from ase_ga.data import PrepareDB
from ase_ga import set_raw_score
from oganesson.ogstructure import OgStructure
import os
from typing import List
import numpy as np
import json


class GA:
    def finalize(self, atoms, energy):
        raw_score = -energy
        set_raw_score(atoms, raw_score)

    def relax(self, atoms, cellbounds=None, model="diep"):
        structure = OgStructure(atoms)
        structure.relax(verbose=self.verbose, steps=self.steps, model=model)
        e = float(structure.total_energy)
        s = structure.structure
        self.finalize(atoms, energy=e)
        return OgStructure.pymatgen_to_ase(s), e

    def __init__(
        self,
        population: List[OgStructure] = None,
        species=None,
        population_size=20,
        box_volume=240,
        a: List = [3, 10],
        b: List = [3, 10],
        c: List = [3, 10],
        phi: List = [35, 145],
        chi: List = [35, 145],
        psi: List = [35, 145],
        verbose=False,
        steps=1000,
        experiment_tag=None,
        write_initial_structures=False,
        rmax=10,
        model="diep",
    ) -> None:
        # Either establish a new population from scratch by randomly filling boxes,
        # or start from a specified population of structures
        self.path = "og_lab/"
        self.verbose = verbose
        self.steps = steps
        self.experiment_tag = experiment_tag
        self.write_initial_structures = write_initial_structures
        self.rmax = rmax
        self.model = model
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        if experiment_tag is None:
            import uuid

            self.path = "og_lab/ga_" + uuid.uuid4().hex
        else:
            self.path = "og_lab/ga_" + self.experiment_tag

        if not os.path.isdir(self.path):
            os.mkdir(self.path)

        self.path_relaxed = self.path + "/relaxed/"
        if not os.path.isdir(self.path_relaxed):
            os.mkdir(self.path_relaxed)

        self.fitness_each_iteration = {}

        self._create_or_read_fitness_file()

        h_max = 0

        print("og:Starting structural optimization using genetic algorithms..")
        if population is None:
            self.N = population_size
            self.volume = box_volume
            self.species = species
            self.Z = [atomic_numbers[x] for x in self.species]

            self.blmin = closest_distances_generator(
                atom_numbers=self.Z, ratio_of_covalent_radii=0.6
            )
            self.cellbounds = CellBounds(
                bounds={"phi": phi, "chi": chi, "psi": psi, "a": a, "b": b, "c": c}
            )

            self.splits = {(2,): 1, (1,): 1}
            self.slab = Atoms("", pbc=True)

            self.sg = StartGenerator(
                self.slab,
                self.species,
                self.blmin,
                box_volume=self.volume,
                number_of_variable_cell_vectors=3,
                cellbounds=self.cellbounds,
                splits=self.splits,
            )

            # Create the database
            self.database = PrepareDB(
                db_file_name=self.path + "/ga.db", stoichiometry=self.Z
            )

            # Generate N random structures
            # and add them to the database
            for i in range(self.N):
                a = self.sg.get_new_candidate()
                self.database.add_unrelaxed_candidate(a)
                if self.write_initial_structures:
                    self.path_initial = self.path + "/initial/"
                    if not os.path.isdir(self.path_initial):
                        os.mkdir(self.path_initial)

                    write(self.path_initial + "/" + str(a.info["confid"]) + ".cif", a)

                cell = a.cell
                vol = a.cell.volume
                axb = np.cross(cell[(i + 1) % 3, :], cell[(i + 2) % 3, :])
                h = vol / np.linalg.norm(axb)
                if h > h_max:
                    h_max = h

            # Connect to the database and retrieve some information
            self.database_connection = DataConnection(self.path + "/ga.db")
            self.slab = self.database_connection.get_slab()
            atom_numbers_to_optimize = (
                self.database_connection.get_atom_numbers_to_optimize()
            )
            self.n_top = len(atom_numbers_to_optimize)

        elif isinstance(population, list):
            # First, ensure that all structures have their symbols ordered.
            # This is required by ASE's GA.

            self.N = len(population)
            for i in range(self.N):
                population[i] = population[i].sort_species()
            self.species = [x.symbol for x in population[0].structure.species]
            self.Z = population[0].structure.atomic_numbers
            a_values = []
            b_values = []
            c_values = []
            volume_values = []
            alpha_values = []
            beta_values = []
            gamma_values = []
            for p in population:
                a_values += [p.structure.lattice.a]
                b_values += [p.structure.lattice.b]
                c_values += [p.structure.lattice.c]
                alpha_values += [p.structure.lattice.alpha]
                beta_values += [p.structure.lattice.beta]
                gamma_values += [p.structure.lattice.gamma]
                volume_values += [p.structure.volume]

            self.volume = max(volume_values) * 1.5
            self.blmin = closest_distances_generator(
                atom_numbers=self.Z, ratio_of_covalent_radii=0.6
            )
            self.cellbounds = CellBounds(
                bounds={
                    "alpha": [min(alpha_values) * 0.5, max(alpha_values) * 1.5],
                    "beta": [min(beta_values) * 0.5, max(beta_values) * 1.5],
                    "gamma": [min(gamma_values) * 0.5, max(gamma_values) * 1.5],
                    "a": [min(a_values) * 0.5, max(a_values) * 1.5],
                    "b": [min(b_values) * 0.5, max(b_values) * 1.5],
                    "c": [min(c_values) * 0.5, max(c_values) * 1.5],
                }
            )
            self.splits = {(2,): 1, (1,): 1}
            self.database = PrepareDB(
                db_file_name=self.path + "/ga.db", stoichiometry=self.Z
            )
            for i in range(self.N):
                a = population[i].to_ase()
                self.database.add_unrelaxed_candidate(a)
                if self.write_initial_structures:
                    self.path_initial = self.path + "/initial/"
                    if not os.path.isdir(self.path_initial):
                        os.mkdir(self.path_initial)

                    write(self.path_initial + "/" + str(a.info["confid"]) + ".cif", a)

                cell = a.cell
                vol = a.cell.volume
                axb = np.cross(cell[(i + 1) % 3, :], cell[(i + 2) % 3, :])
                h = vol / np.linalg.norm(axb)
                if h > h_max:
                    h_max = h

            self.database_connection = DataConnection(self.path + "/ga.db")
            self.slab = self.database_connection.get_slab()
            atom_numbers_to_optimize = (
                self.database_connection.get_atom_numbers_to_optimize()
            )
            self.n_top = len(atom_numbers_to_optimize)

        else:
            raise Exception("og:Wrong input arguments!")

        print("og:Population size:", self.N)

        self.comp = OFPComparator(
            n_top=self.n_top,
            dE=1.0,
            cos_dist_max=1e-3,
            rcut=10.0,
            binwidth=0.05,
            pbc=[True, True, True],
            sigma=0.05,
            nsigma=4,
            recalculate=False,
        )

        self.pairing = CutAndSplicePairing(
            self.slab,
            self.n_top,
            self.blmin,
            p1=1.0,
            p2=0.0,
            minfrac=0.15,
            number_of_variable_cell_vectors=3,
            cellbounds=self.cellbounds,
            use_tags=False,
        )

        self.strainmut = StrainMutation(
            self.blmin,
            stddev=0.7,
            cellbounds=self.cellbounds,
            number_of_variable_cell_vectors=3,
            use_tags=False,
        )
        self.blmin_soft = closest_distances_generator(atom_numbers_to_optimize, 0.1)
        self.softmut = SoftMutation(
            self.blmin_soft,
            bounds=[2.0, 5.0],
            use_tags=False,
            used_modes_file=self.path + "/soft_mutations_modes.json",
        )
        print("og:rmax threshold:", h)
        if h_max >= self.rmax:
            print("og:Including the Rich2poorPermutation in the list of GA operators")
            self.rich2poor = Rich2poorPermutation(self.species, rmax=rmax)
            self.operators = OperationSelector(
                [3, 2, 3],
                # [self.rich2poor, self.pairing, self.softmut, self.strainmut],
                [self.rich2poor, self.pairing, self.strainmut],
            )
        else:
            self.operators = OperationSelector(
                # [4, 3, 3], [self.pairing, self.softmut, self.strainmut]
                [4, 6],
                [self.pairing, self.strainmut],
            )
        self.relaxed_population = False

    def _create_or_read_fitness_file(self):
        if not os.path.isfile(self.path + "/fitness_each_iteration.json"):
            f = open(self.path + "/fitness_each_iteration.json", "w")
            json.dump(self.fitness_each_iteration, f)
            f.close()
        else:
            f = open(self.path + "/fitness_each_iteration.json", "r")
            self.fitness_each_iteration = json.load(f)
            f.close()

    def _update_fitness_file(self):
        f = open(self.path + "/fitness_each_iteration.json", "w")
        json.dump(self.fitness_each_iteration, f)
        f.close()

    def evolve(self, num_offsprings=20):
        energies_for_step = []

        if not self.relaxed_population:
            while self.database_connection.get_number_of_unrelaxed_candidates() > 0:
                a = self.database_connection.get_an_unrelaxed_candidate()
                if self.write_initial_structures:
                    self.path_initial = self.path + "/initial/"
                    if not os.path.isdir(self.path_initial):
                        os.mkdir(self.path_initial)

                    write(self.path_initial + "/" + str(a.info["confid"]) + ".cif", a)
                print("og:Fitness function: PES potential", self.model)
                relaxed_a, e = self.relax(
                    a, cellbounds=self.cellbounds, model=self.model
                )
                energies_for_step += [e]
                a.positions = relaxed_a.positions
                a.cell = relaxed_a.cell
                a.pbc = True
                self.database_connection.add_relaxed_step(a)

                cell = a.get_cell()
                if not self.cellbounds.is_within_bounds(cell):
                    self.database_connection.kill_candidate(a.info["confid"])
                else:
                    relaxed_a.write(
                        self.path_relaxed + str(a.info["confid"]) + ".cif", "cif"
                    )
            self.relaxed_population = True
        else:
            self.population = Population(
                data_connection=self.database_connection,
                population_size=self.N,
                comparator=self.comp,
                logfile=self.path + "/log.txt",
                use_extinct=True,
            )

            current_pop = self.population.get_current_population()
            self.strainmut.update_scaling_volume(current_pop, w_adapt=0.5, n_adapt=4)
            self.pairing.update_scaling_volume(current_pop, w_adapt=0.5, n_adapt=4)

            for step in range(num_offsprings):

                print("og:Now starting configuration number {0}".format(step))

                a3 = None
                while a3 is None:
                    a1, a2 = self.population.get_two_candidates()
                    a3, desc = self.operators.get_new_individual([a1, a2])

                # Save the unrelaxed candidate
                self.database_connection.add_unrelaxed_candidate(a3, description=desc)

                if self.write_initial_structures:
                    self.path_initial = self.path + "/initial/"
                    if not os.path.isdir(self.path_initial):
                        os.mkdir(self.path_initial)

                    write(self.path_initial + "/" + str(a3.info["confid"]) + ".cif", a3)

                # Relax the new candidate and save it
                relaxed_a, e = self.relax(
                    a3, cellbounds=self.cellbounds, model=self.model
                )
                energies_for_step += [e]
                a3.positions = relaxed_a.positions
                a3.cell = relaxed_a.cell
                a3.pbc = True
                self.database_connection.add_relaxed_step(a3)

                # If the relaxation has changed the cell parameters
                # beyond the bounds we disregard it in the population
                cell = a3.get_cell()
                if not self.cellbounds.is_within_bounds(cell):
                    self.database_connection.kill_candidate(a3.info["confid"])
                else:
                    a3.write(self.path_relaxed + str(a3.info["confid"]) + ".cif", "cif")

                # Update the population
                self.population.update()

                if step % 10 == 0:
                    current_pop = self.population.get_current_population()
                    self.strainmut.update_scaling_volume(
                        current_pop, w_adapt=0.5, n_adapt=4
                    )
                    self.pairing.update_scaling_volume(
                        current_pop, w_adapt=0.5, n_adapt=4
                    )
                    write(self.path + "/current_population.traj", current_pop)

            self._create_or_read_fitness_file()

            iterations_labels = list(self.fitness_each_iteration.keys())
            iterations_labels = [int(x) for x in iterations_labels]
            if len(iterations_labels) > 0:
                current_iteration_label = max(iterations_labels) + 1
            else:
                current_iteration_label = 1

            self.fitness_each_iteration[current_iteration_label] = min(
                energies_for_step
            )
            self._update_fitness_file()

            print("og:GA finished after step %d" % step)
            hiscore = get_raw_score(current_pop[0])
            print("og:Highest raw score = -%8.4f eV" % hiscore)

            all_candidates = self.database_connection.get_all_relaxed_candidates()
            write(self.path + "/all_candidates.traj", all_candidates)

            current_pop = self.population.get_current_population()
            write(self.path + "/current_population.traj", current_pop)
