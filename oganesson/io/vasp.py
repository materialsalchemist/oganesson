from pymatgen.core import Structure
from oganesson.ogstructure import OgStructure
from pymatgen.io.vasp import Poscar as MPPoscar
import numpy as np
from ase import Atoms
from ase.cell import Cell


class Poscar:
    def __init__(self, structure) -> None:
        self.ogstructure = OgStructure(structure)

    def freeze_up_to(self, z):
        """
        Produce the POSCAR file after modifying it to constraint the layers of atoms below z Angstroms
        """
        constraints = np.ndarray(self.ogstructure().cart_coords.shape)
        for i in range(len(self.ogstructure)):
            atom = self.ogstructure().cart_coords[i]
            if atom[2] < z:
                constraints[i, :] = False
            else:
                constraints[i, :] = True
        mpposcar = MPPoscar(self.ogstructure(), selective_dynamics=constraints)
        return mpposcar.get_string()


class Potcar:
    def __init__(self, file_name="POTCAR") -> None:
        f = open(file_name)
        potcar_content = f.read()
        f.close
        self.element_data = {}
        blocks = potcar_content.strip().split("TITEL")
        blocks = blocks[1:]
        for block in blocks:
            block_lines = block.splitlines()
            a = block_lines[0].split()[2]
            a = a.split("_")[0]
            if a in self.element_data.keys():
                # Skip the next block
                continue
            else:
                self.element_data[a] = {}
            for line in block_lines:
                line = line.replace(";", "")
                if "ZVAL" in line:
                    self.element_data[a]["ZVAL"] = float(line.split()[5])
                if "RCORE" in line:
                    self.element_data[a]["RCORE"] = float(line.split()[2])
                if "POMASS" in line:
                    self.element_data[a]["POMASS"] = float(line.split()[2])


class Outcar:
    def __init__(
        self,
        outcar_directory: str = "./",
        outcar_file: str = "OUTCAR",
        poscar_file: str = None,
    ) -> None:
        self.outcar_directory = outcar_directory
        self.outcar_file = outcar_file
        self.poscar_file = poscar_file

    def get_md_data(self):
        return self._outcar_extractor()

    def write_md_data(self, file_name: str = None, path: str = None):
        energies, structures, forces_vectors, stress_matrices = self._outcar_extractor()
        if file_name is None:
            file_name = self.outcar_file + ".json"
        if path is None:
            path = self.outcar_directory
        output = {
            "energies": energies,
            "structures": structures,
            "forces": forces_vectors,
            "stresses": stress_matrices,
        }
        import json

        with open(path + file_name, "w") as f:
            json.dump(output, f)

    def _outcar_extractor(self):
        def get_indices(l, s):
            indices = []
            for i in range(len(l)):
                if s in l[i]:
                    indices += [i]
            return indices

        def get_vectors(l, positions, displacement, rows, columns):
            vectors = []
            for position in positions:
                vector = []
                s = l[position + displacement : position + displacement + rows]
                for r in s:
                    c = r.split()
                    c = c[0:columns]
                    c = [float(x) for x in c]
                    vector += [c]
                vectors += [vector]
            return vectors

        def get_atoms_counts(l):
            ion_counts = []
            ions = []
            for s in l:
                if "ions per type" in s:
                    s = s.replace("ions per type", "").replace("=", "").split()
                    ion_counts = [int(x) for x in s]
                    break
            for s in l:
                if "POSCAR =" in s:
                    s = s.replace("POSCAR =", "").split()
                    ions = s
            return ion_counts, ions

        outcarf = open(self.outcar_directory + self.outcar_file, "r")
        outcar = outcarf.readlines()
        outcarf.close()

        structures = []
        atoms = []
        if self.poscar_file is not None:
            structure = Structure.from_file(self.outcar_directory + self.poscar_file)
            number_of_atoms = len(structure)
            atoms = [s.specie.symbol for s in structure]
        else:
            ion_counts, ions = get_atoms_counts(outcar)

            for i in range(len(ions)):
                atoms += ion_counts[i] * [ions[i]]

            number_of_atoms = sum(ion_counts)
        energies_lines = [x for x in outcar if "free  energy   TOTEN" in x]
        energies = [x.split()[4] for x in energies_lines]
        direct_lattice_vectors_positions = get_indices(outcar, "direct lattice vectors")
        positions_forces_vectors_positions = get_indices(
            outcar, "TOTAL-FORCE (eV/Angst)"
        )
        stresses_positions = get_indices(outcar, "stress matrix")
        lattice_vectors = get_vectors(outcar, direct_lattice_vectors_positions, 1, 3, 3)
        positions_forces_vectors = np.array(
            get_vectors(
                outcar, positions_forces_vectors_positions, 2, number_of_atoms, 6
            )
        )
        positions_vectors = positions_forces_vectors[:, :, 0:3]
        forces_vectors = positions_forces_vectors[:, :, 3:6]
        stress_matrices = get_vectors(outcar, stresses_positions, 1, 3, 3)
        for i in range(len(positions_vectors)):
            structures += [
                Structure(
                    species=atoms,
                    coords=positions_vectors[i],
                    lattice=lattice_vectors[i],
                    coords_are_cartesian=True,
                ).as_dict()
            ]
        length_of_least = min(
            [
                len(energies),
                len(structures),
                len(forces_vectors.tolist()),
                len(stress_matrices),
            ]
        )
        return (
            energies[:length_of_least],
            structures[:length_of_least],
            forces_vectors.tolist()[:length_of_least],
            stress_matrices[:length_of_least],
        )

    def get_maximum_force(self):
        TAG = "TOTAL-FORCE (eV/Angst)"
        if self.poscar_file is None:
            print(
                "og:Must supply a POSCAR file along with the OUTCAR file to extract the maximum force."
            )
            return None
        outcarf = open(self.outcar_file, "r")
        outcar = outcarf.readlines()
        outcarf.close()
        poscarf = open(self.poscar_file, "r")
        poscar = poscarf.readlines()
        poscarf.close()
        num_atoms = sum([int(x) for x in poscar[6].split()])
        list_of_tags = []
        for il in range(len(outcar)):
            if TAG in outcar[il]:
                list_of_tags += [il]
        i = list_of_tags[-1]
        if i >= len(outcar) or i + 2 + num_atoms >= len(outcar):
            i = list_of_tags[-2]
        lines = outcar[i + 2 : i + 2 + num_atoms]
        forces = np.array([l.split()[3:6] for l in lines]).astype(np.float64)
        print("Maximum force =", forces.max())
        return forces.max()


def string_to_ase(poscar_str, convert=1):
    poscar = MPPoscar.from_str(poscar_str)
    structure = poscar.structure
    fc = structure.frac_coords
    lattice = structure.lattice
    cc = structure.cart_coords

    a = Atoms(
        scaled_positions=fc,
        numbers=structure.atomic_numbers,
        pbc=True,
        cell=Cell.fromcellpar(
            [
                lattice.a * convert,
                lattice.b * convert,
                lattice.c * convert,
                lattice.alpha,
                lattice.beta,
                lattice.gamma,
            ]
        ),
    )

    return a


class Xdatcar:
    def __init__(
        self,
        directory: str = "./",
        filename: str = "XDATCAR",
    ) -> None:
        self.directory = directory
        self.filename = filename
        f = open(self.filename, "r")
        traj = f.readlines()
        f.close()

        lattice_list = traj[0:8]
        lattice_str = "".join(lattice_list)
        atom_counts = [int(x) for x in traj[6].split()]
        num_atoms = sum(atom_counts)
        number_of_lines = num_atoms
        trajectory = traj
        tot_num_images = int(len(trajectory) / (number_of_lines+8))

        print("file: ", self.filename, "total number of images:", str(tot_num_images))

        trajectory_list_ang = []

        for i in range(0, tot_num_images):
            positions_str = "".join(
                traj[8*(i+1) + i * (num_atoms) : 8*(i+1) + (i + 1) * (num_atoms)]
            )
            a_ang = string_to_ase(lattice_str + positions_str)
            trajectory_list_ang += [a_ang]

        self.trajectory = trajectory_list_ang

    def get_trajectory(self):
        return self.trajectory

    def get_displacements(self):
        """[site, time step, axis]"""
        displacements = np.zeros([len(self.trajectory[0]), len(self.trajectory) - 1, 3])
        t0 = self.trajectory[0]
        print(t0)
        for it in range(len(self.trajectory[1:])):
            t = self.trajectory[it]
            for isite in range(len(t)):
                displacements[isite][it] = t.positions[isite] - t0.positions[isite]
        return displacements


def get_nelect(vasp_folder="./"):
    try:
        og = OgStructure(file_name=vasp_folder + "/POSCAR")
        potcar = Potcar(file_name=vasp_folder + "/POTCAR")
        nelect = 0
        for m in potcar.element_data.keys():
            nelect += (
                og.structure.composition.as_dict()[m] * potcar.element_data[m]["ZVAL"]
            )
        return nelect
    except Exception as e:
        print("Problem processing VASP input files", e)


def get_bandgap(vasp_folder="./"):
    try:
        feigenval = open(vasp_folder + "EIGENVAL", "r")
        eigenval = feigenval.readlines()
        feigenval.close()
        fdoscar = open(vasp_folder + "DOSCAR", "r")
        doscar = fdoscar.readlines()
        fdoscar.close()
        fermi = float(doscar[5].split()[3])
        info = eigenval[5].split()
        num_points = int(info[2])
        num_bands = int(info[1])

        VB = []
        CB = []

        for b in range(num_bands):
            vals = eigenval[7 + (num_points + 2) * b : 7 + (num_points + 2) * (b + 1)]
            vals = vals[1:-1]
            vals = np.array([v.split() for v in vals])[:, 1].astype(np.float64)
            VB += [vals[vals <= fermi]]
            CB += [vals[vals > fermi]]

        CBM = min([x[0] for x in CB])
        VBM = max([x[-1] for x in VB])
        bandgap = CBM - VBM
        print(bandgap)
        return bandgap
    except:
        pass
