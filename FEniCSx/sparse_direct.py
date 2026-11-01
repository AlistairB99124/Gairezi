"""Sparse direct solves with MUMPS (PETSc), reusing the factorisation of an identical matrix.

SciPy's SuperLU refactorises on every call; on the full-dam cohesive network
(about 178k DOFs with interface springs) one bonded solve took about 195 s.
MUMPS factorises the same system in seconds and the factors are kept while
the matrix is unchanged (for example across bisection load factors, which
only scale the load). Symmetric systems (including the indefinite cohesive
saddle-point operators) use the MUMPS LDL^T factorisation, about 1.7 times
faster than LU. MUMPS does its dense work in BLAS, so the installed BLAS
dominates the factorisation time: the reference BLAS is several times slower
than OpenBLAS. Systems smaller than ``MUMPS_MINIMUM_UNKNOWNS`` keep
SciPy SuperLU, which is fast there and reproduces earlier results exactly: the
cohesive Newton path branches on the roundoff sign of near-zero openings, so a
different (equally accurate) solver can take a different path. SuperLU is also
used when petsc4py or MUMPS is unavailable. Singular systems raise
``SingularMatrixError`` on both paths.
"""
from collections import OrderedDict
import hashlib
import warnings

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import MatrixRankWarning, spsolve

# Factorised matrices kept (each can take gigabytes on the full dam).
CACHE_SIZE = 2
# Smaller systems use SuperLU (unchanged results); larger ones use MUMPS.
MUMPS_MINIMUM_UNKNOWNS = 20000
# MUMPS ICNTL(14): percentage workspace headroom; indefinite saddle-point systems need more than the default.
WORKSPACE_HEADROOM_PERCENT = 100
_MUMPS_NUMERICALLY_SINGULAR = -10
# MUMPS ICNTL(24)=1 detects null pivots (reported in INFOG(28)) so singular systems are rejected like SuperLU.
_MUMPS_NULL_PIVOT_DETECTION = 24
_MUMPS_NULL_PIVOTS = 28
# Relative asymmetry (max |A - A^T| / max |A|) below which a matrix is factorised as symmetric.
SYMMETRY_TOLERANCE = 1e-12


class SingularMatrixError(RuntimeError):
    """The matrix is (numerically) singular; no solution is returned."""


_cache: "OrderedDict[bytes, _Factor]" = OrderedDict()
_petsc_module = None


def _petsc():
    global _petsc_module
    if _petsc_module is None:
        try:
            from petsc4py import PETSc
        except ImportError:
            _petsc_module = False
        else:
            _petsc_module = PETSc if PETSc.Sys.hasExternalPackage("mumps") else False
    return _petsc_module or None


def backend() -> str:
    return "mumps" if _petsc() is not None else "superlu"


def _canonical(matrix):
    canonical = sparse.csr_matrix(matrix, dtype=float, copy=True)
    canonical.sum_duplicates()
    canonical.sort_indices()
    return canonical


def _key(matrix) -> bytes:
    digest = hashlib.blake2b(digest_size=20)
    digest.update(np.asarray(matrix.shape, dtype=np.int64).tobytes())
    for array in (matrix.indptr, matrix.indices, matrix.data):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.digest()


def _symmetric(matrix) -> bool:
    scale = float(np.max(np.abs(matrix.data), initial=0.))
    return float(np.max(np.abs((matrix - matrix.T).data), initial=0.)) <= SYMMETRY_TOLERANCE * scale


def _same(a, b) -> bool:
    return (a.shape == b.shape and np.array_equal(a.indptr, b.indptr) and np.array_equal(a.indices, b.indices)
            and np.array_equal(a.data, b.data))


class _Factor:
    def __init__(self, PETSc, matrix):
        self.PETSc = PETSc
        self.source = matrix
        self.symmetric = _symmetric(matrix)
        self.matrix = PETSc.Mat().createAIJ(
            size=matrix.shape, comm=PETSc.COMM_SELF,
            csr=(matrix.indptr.astype(PETSc.IntType), matrix.indices.astype(PETSc.IntType), matrix.data))
        if self.symmetric:
            # MUMPS then uses the general symmetric (indefinite) LDL^T, not the SPD variant.
            self.matrix.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        self.ksp = PETSc.KSP().create(PETSc.COMM_SELF)
        self.ksp.setOperators(self.matrix)
        self.ksp.setType("preonly")
        pc = self.ksp.getPC()
        pc.setType("cholesky" if self.symmetric else "lu")
        pc.setFactorSolverType("mumps")
        pc.setFactorSetUpSolverType()
        self.factor = pc.getFactorMatrix()
        self.factor.setMumpsIcntl(14, WORKSPACE_HEADROOM_PERCENT)
        self.factor.setMumpsIcntl(_MUMPS_NULL_PIVOT_DETECTION, 1)
        try:
            self.ksp.setUp()
        except PETSc.Error as error:
            status = self.factor.getMumpsInfog(1)
            self.destroy()
            if status == _MUMPS_NUMERICALLY_SINGULAR:
                raise SingularMatrixError("MUMPS: matrix is numerically singular (INFOG(1)=-10)") from error
            raise RuntimeError(f"MUMPS factorisation failed (INFOG(1)={status})") from error
        null_pivots = self.factor.getMumpsInfog(_MUMPS_NULL_PIVOTS)
        if null_pivots > 0:
            self.destroy()
            raise SingularMatrixError(f"MUMPS: matrix is singular ({null_pivots} null pivot(s))")

    def solve(self, rhs):
        solution, load = self.matrix.createVecs()
        try:
            load.setArray(np.ascontiguousarray(rhs, dtype=float))
            self.ksp.solve(load, solution)
            if self.ksp.getConvergedReason() < 0:
                if self.factor.getMumpsInfog(1) == _MUMPS_NUMERICALLY_SINGULAR:
                    raise SingularMatrixError("MUMPS: matrix is numerically singular (INFOG(1)=-10)")
                raise RuntimeError(f"MUMPS solve failed (KSP reason {self.ksp.getConvergedReason()})")
            return solution.getArray().copy()
        finally:
            solution.destroy()
            load.destroy()

    def destroy(self):
        self.ksp.destroy()
        self.matrix.destroy()


def _superlu(matrix, rhs):
    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        try:
            return spsolve(matrix, rhs)
        except MatrixRankWarning as error:
            raise SingularMatrixError(str(error)) from error


def clear_cache() -> None:
    while _cache:
        _cache.popitem(last=False)[1].destroy()


def solve(matrix, rhs) -> np.ndarray:
    """Solve ``matrix @ x = rhs`` for a square sparse matrix and a 1-D right-hand side."""
    rhs = np.asarray(rhs, dtype=float)
    if matrix.shape[0] != matrix.shape[1] or rhs.shape != (matrix.shape[0],):
        raise ValueError("solve needs a square matrix and a matching 1-D right-hand side")
    PETSc = _petsc() if matrix.shape[0] >= MUMPS_MINIMUM_UNKNOWNS else None
    if PETSc is None:
        return _superlu(sparse.csc_matrix(matrix), rhs)
    canonical = _canonical(matrix)
    key = _key(canonical)
    factor = _cache.get(key)
    if factor is not None and _same(factor.source, canonical):
        _cache.move_to_end(key)
    else:
        if factor is not None:
            _cache.pop(key).destroy()
        while _cache and len(_cache) >= CACHE_SIZE:
            _cache.popitem(last=False)[1].destroy()
        factor = _Factor(PETSc, canonical)
        if CACHE_SIZE <= 0:
            try:
                return factor.solve(rhs)
            finally:
                factor.destroy()
        _cache[key] = factor
    return factor.solve(rhs)
