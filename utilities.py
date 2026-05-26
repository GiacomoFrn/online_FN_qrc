"""Basic Hilbert-space utilities for the amplitude-encoding QRC notebooks.

The classes in this module are intentionally lightweight. They provide the
basis states, local operators, composite operators, reduced density matrices,
and expectation values needed by the tutorial notebook.
"""

import numpy as np


class HilbertSpace:
    """Represent a finite-dimensional spin or truncated photonic Hilbert space.

    Parameters
    ----------
    n : int
        Number of qubits for ``system='spin'``. For ``system='photon'``, this
        sets the truncation such that the basis dimension is ``n + 1``.
    system : {'spin', 'photon'}
        Type of local Hilbert space.
    flag_operators : bool, optional
        If ``True``, generate the corresponding local operators.
    """

    def __init__(self, n, system, flag_operators=True):
        if system not in {"spin", "photon"}:
            raise ValueError("system must be either 'spin' or 'photon'")

        self.n = n
        self.system = system
        self.basis = self.generate_basis()

        # Single-qubit identity and Pauli matrices. These are used to build
        # many-body spin operators through Kronecker products.
        self.id_local = np.eye(2)
        self.sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])
        self.sigma_y = 1j * np.array([[0.0, -1.0], [1.0, 0.0]])
        self.sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]])

        if flag_operators:
            if self.system == "spin":
                self.generate_spin_operators()
            elif self.system == "photon":
                self.generate_photon_operators()

    def generate_basis(self):
        """Generate the computational/Fock basis for the selected system."""
        if self.system == "photon":
            return self.generate_photon_fock_basis()
        if self.system == "spin":
            return self.generate_spin_fock_basis()

    def generate_photon_fock_basis(self):
        """Generate the truncated single-mode photonic Fock basis.

        The basis is represented as one-hot vectors with dimension ``n + 1``.
        """
        self.dimension = self.n + 1
        fock_basis = []
        for i in range(self.dimension):
            v_ph = np.zeros(self.dimension, dtype=int)
            v_ph[i] = 1
            fock_basis.append(tuple(v_ph))
        return fock_basis

    def generate_spin_fock_basis(self):
        """Generate the computational basis for ``n`` spin-1/2 particles."""
        self.dimension = 2**self.n
        spin_fock_basis = []
        for i in range(self.dimension):
            binary_repr = np.binary_repr(i, width=self.n)
            spin_fock_basis.append(tuple(int(bit) for bit in binary_repr))

        # The reversed convention is kept for consistency with the notebooks.
        spin_fock_basis.reverse()
        return spin_fock_basis

    def generate_spin_operators(self):
        """Generate single-site Pauli operators embedded in the spin space."""
        self.X = {}
        self.Y = {}
        self.Z = {}
        for i in range(1, self.n + 1):
            left_identity = np.eye(2 ** (i - 1))
            right_identity = np.eye(2 ** (self.n - i))
            self.X[i] = np.kron(
                left_identity, np.kron(self.sigma_x, right_identity)
            )
            self.Y[i] = np.kron(
                left_identity, np.kron(self.sigma_y, right_identity)
            )
            self.Z[i] = np.kron(
                left_identity, np.kron(self.sigma_z, right_identity)
            )

    def generate_photon_operators(self):
        """Generate annihilation, creation, and number operators."""
        if self.n == 0:
            self.a_ph = 0
            self.a_dag_ph = 0
            self.n_ph = 0
        elif self.n == 1:
            self.a_ph = np.array([[0, 1], [0, 0]])
            self.a_dag_ph = self.a_ph.T
            self.n_ph = self.a_dag_ph @ self.a_ph
        else:
            self.a_ph = np.diag(np.sqrt(np.arange(1, self.n + 1)), k=1)
            self.a_dag_ph = self.a_ph.T
            self.n_ph = self.a_dag_ph @ self.a_ph


class CompositeHilbertSpace:
    """Tensor product of multiple ``HilbertSpace`` objects.

    The order of the tensor product follows the insertion order of
    ``hilbert_spaces_dict``. In the tutorial this is typically ``{'R': ..., 'M': ...}``,
    where ``R`` denotes input/readout qubits and ``M`` denotes memory qubits.
    """

    def __init__(self, hilbert_spaces_dict, flag_operators=True):
        self.hilbert_spaces_dict = hilbert_spaces_dict
        self.composite_basis = self.generate_composite_basis()
        self.basis_to_index = {
            base: i for i, base in enumerate(self.composite_basis)
        }

        systems = list(self.hilbert_spaces_dict.values())
        photon_indices = [
            i for i, hs in enumerate(systems) if hs.system == "photon"
        ]
        self.flag_photon = len(photon_indices) > 0
        self.idx_photon = photon_indices[0] if self.flag_photon else None
        self.idx_spins = [
            i for i, hs in enumerate(systems) if hs.system == "spin"
        ]

        if flag_operators:
            self.composite_operators = self.generate_composite_operators()

        # Used to cache index lists for repeated partial traces.
        self.flag_partial_composite_basis = False

    def generate_composite_basis(self):
        """Generate the tensor-product basis as tuples of local basis labels."""
        hs_list = list(self.hilbert_spaces_dict.values())
        tmp_basis = hs_list[0].basis
        i = 0
        while i + 1 < len(hs_list):
            curr_basis = tmp_basis
            tmp_basis = []
            for base_v1 in curr_basis:
                for base_v2 in hs_list[i + 1].basis:
                    tmp_basis.append(base_v1 + base_v2)
            i += 1
        return tmp_basis

    def generate_composite_operators(self):
        """Embed local operators into the full tensor-product Hilbert space."""
        dimensions = [hs.dimension for hs in self.hilbert_spaces_dict.values()]
        self.dimension = np.prod(dimensions)

        # Photonic operators, if a photonic subsystem is present.
        dim_spins = np.prod([dimensions[i] for i in self.idx_spins])
        if self.flag_photon:
            ph_space = list(self.hilbert_spaces_dict.values())[self.idx_photon]
            self.a_ph = np.kron(ph_space.a_ph, np.eye(dim_spins))
            self.a_dag_ph = self.a_ph.T
            self.n_ph = self.a_dag_ph @ self.a_ph
            self.g_2_correlator_operator = (
                self.a_dag_ph @ self.a_dag_ph @ self.a_ph @ self.a_ph
            )

        # Spin operators. The index ``op_idx`` labels spin sites across all
        # spin subsystems in the composite space.
        self.X = {}
        self.Y = {}
        self.Z = {}
        op_idx = 1
        for i, hs in enumerate(self.hilbert_spaces_dict.values()):
            if hs.system == "spin":
                prev_dim = np.prod(dimensions[:i]) if dimensions[:i] else 1
                next_dim = np.prod(dimensions[i + 1 :]) if dimensions[i + 1 :] else 1
                for j in range(1, hs.n + 1):
                    self.X[op_idx] = np.kron(
                        np.eye(prev_dim), np.kron(hs.X[j], np.eye(next_dim))
                    )
                    self.Y[op_idx] = np.kron(
                        np.eye(prev_dim), np.kron(hs.Y[j], np.eye(next_dim))
                    )
                    self.Z[op_idx] = np.kron(
                        np.eye(prev_dim), np.kron(hs.Z[j], np.eye(next_dim))
                    )
                    op_idx += 1

        # Collective spin operators.
        n_spins = np.sum(
            [
                hs.n
                for hs in self.hilbert_spaces_dict.values()
                if hs.system == "spin"
            ]
        )
        self.X_total = self.X[1].copy()
        self.Y_total = self.Y[1].copy()
        self.Z_total = self.Z[1].copy()
        for i in range(2, n_spins + 1):
            self.X_total += self.X[i]
            self.Y_total += self.Y[i]
            self.Z_total += self.Z[i]

        self.S_plus = self.X_total + 1j * self.Y_total
        self.S_minus = self.X_total - 1j * self.Y_total

    def get_reduced_density_matrix(self, rho, keys_to_keep):
        """Return the reduced density matrix for selected subsystems.

        Parameters
        ----------
        rho : ndarray
            Density matrix in the full composite Hilbert space.
        keys_to_keep : str or sequence of str
            Dictionary key(s) identifying the subsystem(s) to keep. The current
            implementation assumes that the kept subsystem and the traced-out
            subsystem are consecutive blocks in the tensor-product ordering.
        """
        subsystem_dims = np.prod(
            [self.hilbert_spaces_dict[key].dimension for key in keys_to_keep]
        )
        rho_subsystem = np.zeros((subsystem_dims, subsystem_dims), dtype=complex)

        keys_to_trace_out = [
            key for key in self.hilbert_spaces_dict.keys() if key not in keys_to_keep
        ]

        if (
            self.flag_partial_composite_basis
            and tuple(keys_to_keep) in self.indices_storage.keys()
        ):
            for i in range(subsystem_dims):
                for j in range(subsystem_dims):
                    indices = self.indices_storage[tuple(keys_to_keep)][i, j]
                    sum_tmp = 0
                    for idx in indices:
                        sum_tmp += rho[idx[0], idx[1]]
                    rho_subsystem[i, j] = sum_tmp
            return rho_subsystem

        # Build and cache the basis of the subsystem that is kept.
        subsystems_dict = {}
        for key in keys_to_keep:
            subsystems_dict[key] = self.hilbert_spaces_dict[key]
        subsystems_composite_space = CompositeHilbertSpace(
            subsystems_dict, flag_operators=False
        )
        self.partial_composite_basis = {}
        self.partial_composite_basis[
            tuple(keys_to_keep)
        ] = subsystems_composite_space.composite_basis
        self.flag_partial_composite_basis = True
        del subsystems_composite_space

        # Build and cache the basis of the complementary subsystem.
        subsystems_dict = {}
        for key in keys_to_trace_out:
            subsystems_dict[key] = self.hilbert_spaces_dict[key]
        subsystems_composite_space = CompositeHilbertSpace(
            subsystems_dict, flag_operators=False
        )
        self.partial_composite_basis[
            tuple(keys_to_trace_out)
        ] = subsystems_composite_space.composite_basis
        del subsystems_composite_space

        # Determine whether the kept subsystem appears before or after the
        # subsystem that is traced out in the tensor-product order.
        idx_keys_dict = {
            key: i for i, key in enumerate(self.hilbert_spaces_dict.keys())
        }
        if idx_keys_dict[keys_to_keep[0]] < idx_keys_dict[keys_to_trace_out[0]]:
            to_keep_first = True
        else:
            to_keep_first = False

        self.indices_storage = {}
        self.indices_storage[tuple(keys_to_keep)] = {}
        for i, v_tk in enumerate(
            self.partial_composite_basis[tuple(keys_to_keep)]
        ):
            for j, w_tk in enumerate(
                self.partial_composite_basis[tuple(keys_to_keep)]
            ):
                sum_tmp = 0
                self.indices_storage[tuple(keys_to_keep)][i, j] = []
                for v_tto in self.partial_composite_basis[tuple(keys_to_trace_out)]:
                    if to_keep_first:
                        v = v_tk + v_tto
                        w = w_tk + v_tto
                    else:
                        v = v_tto + v_tk
                        w = v_tto + w_tk

                    row = self.basis_to_index[v]
                    col = self.basis_to_index[w]
                    sum_tmp += rho[row, col]
                    self.indices_storage[tuple(keys_to_keep)][i, j].append(
                        (row, col)
                    )
                rho_subsystem[i, j] = sum_tmp
        return rho_subsystem


class DensityMatrix:
    """Small wrapper for pure states and density matrices."""

    def __init__(self, input):
        if input.ndim == 1:
            self.state = input
            self.state = self.normalize()
            self.dm = self.get_density_matrix()
        elif input.ndim == 2:
            if np.trace(input) != 1:
                self.dm = input / np.trace(input)
            else:
                self.dm = input
        else:
            raise ValueError("Input must be a 1d state vector or a 2d matrix")

    def normalize(self):
        """Normalize the stored state vector."""
        return self.state / np.linalg.norm(self.state)

    def get_density_matrix(self):
        """Return ``|psi><psi|`` for the stored pure state."""
        return np.outer(self.state, self.state.conj())

    def get_expectation_value(self, operator):
        """Compute ``Tr(operator @ rho)`` for the stored density matrix."""
        tmp = operator @ self.dm
        return np.trace(tmp).real

    def __str__(self):
        return str(self.dm)
