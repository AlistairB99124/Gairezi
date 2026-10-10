** FEniCSx: Import and audit dam mesh **
`````
{
  "source_nodes": 121097,
  "imported_nodes": 123337,
  "added_nodes": 2240,
  "source_cells": 101724,
  "tetrahedra": 614824,
  "source_volume_m3": 13530.102850593674,
  "tetrahedral_volume_m3": 13530.102850593674,
  "relative_volume_change": 0.0,
  "maximum_parent_cell_relative_volume_change": 3.189005308891992e-14,
  "minimum_tetrahedron_volume_m3": 0.003388732133817932,
  "bounds_m": [
    [
      7.576444001553635,
      -48.8362786073285,
      -29.0
    ],
    [
      80.99999674477964,
      80.57433783575995,
      0.0
    ]
  ],
  "facet_triangle_counts": {
    "1": 8840,
    "2": 24386,
    "3": 26178,
    "4": 5400,
    "5": 20,
    "6": 20,
    "7": 3400,
    "8": 5280,
    "9": 5280
  },
  "region_tetrahedron_counts": {
    "1": 116440,
    "2": 498384
  },
  "conversion": "pulling triangulation with shared centres on warped faces and affected cell centres"
}
`````
** FEniCSx: Verify dam elastic parity **
`````
dam_elastic_parity: imported 614824 tetrahedra
dam_elastic_parity: passed_provisional_elastic_parity; results at /mnt/c/Users/alist/Projects/Gairezi/FEniCSx/results/dam_elastic_parity/20261010T063733621602Z 
`````