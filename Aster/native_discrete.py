"""Native per-cell characteristics without hundreds of thousands of MED families."""
from __future__ import annotations

import numpy as np
import time


def read_mesh(path, archive):
    """Allocate characteristic groups with the mesh, using Aster 17.4's importer."""
    from code_aster import CA
    from code_aster.ObjectsExt.medctoaster import MEDCouplingMeshHelper
    with np.load(archive) as data:
        point_groups = data["point_group_ids"]
        link_groups = data["link_group_ids"]

    class DiscreteMeshHelper(MEDCouplingMeshHelper):
        def parseMesh(self):
            super().parseMesh()
            # The native importer stores one-based cell numbers and allocates
            # its group table AFTER parsing. Add identities before allocation.
            for aggregate, prefix, group_ids in (
                ("QS_POINTS", "QS_P_", point_groups), ("QS_LINKS", "QS_L_", link_groups),
            ):
                cells = self._groups_of_cells.get(aggregate)
                if not cells or len(cells) != len(group_ids):
                    raise ValueError(f"Missing native quadratic mesh group {aggregate}")
                ordered = np.argsort(group_ids, kind="stable")
                splits = np.flatnonzero(np.diff(group_ids[ordered])) + 1
                for members in np.split(ordered, splits):
                    name = f"{prefix}{int(group_ids[members[0]]):06d}"
                    if name in self._groups_of_cells:
                        raise ValueError(f"Duplicate native characteristic group {name}")
                    self._groups_of_cells[name] = [int(cells[index]) for index in members]

    helper = DiscreteMeshHelper()
    helper.readMedFile(path)
    return helper.buildMesh(CA.Mesh())


def characteristics(mesh, model, archive):
    from code_aster.Cata.Syntax import _F
    from code_aster.Commands import AFFE_CARA_ELEM

    started = time.monotonic()
    entries = []
    connectivity = mesh.getConnectivity()
    with np.load(archive) as data:
        for key, ids_key, aggregate, prefix, cara, kind in (
            ("point_matrices_n_per_m", "point_node_ids", "QS_POINTS", "QS_P_", "K_T_N", "point"),
            ("link_matrices_n_per_m", "link_node_ids", "QS_LINKS", "QS_L_", "K_T_L", "link"),
        ):
            cells = mesh.getCells(aggregate)
            width = 1 if cara == "K_T_N" else 2
            raw_ids = data[ids_key]
            if raw_ids.shape != ((len(cells),) if width == 1 else (len(cells), width)):
                raise ValueError(f"Native {aggregate} node identity archive is incomplete")
            expected = raw_ids.reshape(len(cells), width)
            actual = np.asarray([connectivity[cell] for cell in cells])
            if not np.array_equal(actual, expected):
                raise ValueError(f"Native {aggregate} node identities/order differ from stiffness archive")
            matrices = data[key]
            if matrices.shape != (len(cells), 3, 3) or not np.isfinite(matrices).all():
                raise ValueError(f"Native {aggregate} stiffness archive is incomplete/nonfinite")
            grouped = data[f"{kind}_group_matrices_n_per_m"]
            group_ids = data[f"{kind}_group_ids"]
            if (group_ids.shape != (len(cells),) or not np.issubdtype(group_ids.dtype, np.integer)
                    or group_ids.min() < 0 or group_ids.max() >= len(grouped)
                    or grouped.shape[1:] != (3, 3) or not np.isfinite(grouped).all()):
                raise ValueError("Invalid grouped native stiffness inventory")
            error = np.max(np.abs(grouped[group_ids] - matrices), axis=(1, 2))
            magnitude = np.max(np.abs(matrices), axis=(1, 2))
            if np.any(magnitude <= 0) or np.any(error > 1e-10 * magnitude):
                raise ValueError("Grouped native stiffness exceeds relative encoding error bound")
            ordered = np.argsort(group_ids, kind="stable")
            members = np.split(ordered, np.flatnonzero(np.diff(group_ids[ordered])) + 1)
            if len(members) != len(grouped):
                raise ValueError("Grouped native stiffness contains unused groups")
            for i, (indices, block) in enumerate(zip(members, grouped)):
                group = f"{prefix}{i:06d}"
                if sorted(mesh.getCells(group)) != sorted(cells[index] for index in indices):
                    raise ValueError(f"Native characteristic group {group} has incorrect cell identity")
                matrix = np.zeros((6, 6)) if cara == "K_T_L" else block
                if cara == "K_T_L":
                    matrix[:3, 3:] = block
                    matrix[3:, :3] = block.T
                if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-8):
                    raise ValueError("Nonsymmetric native quadratic discrete stiffness")
                entries.append(_F(
                    GROUP_MA=group, CARA=cara, REPERE="GLOBAL", SYME="OUI",
                    VALE=tuple(matrix[row, column]
                               for column in range(len(matrix)) for row in range(column + 1)),
                ))
    print(f"Quadratic discrete characteristics: prepared {len(entries)} entries in "
          f"{time.monotonic() - started:.3f}s", flush=True)
    result = AFFE_CARA_ELEM(MODELE=model, DISCRET=entries)
    print(f"Quadratic discrete characteristics: assigned stiffness after {time.monotonic() - started:.3f}s",
          flush=True)
    return result
