# Arch dam failure mechanisms: research digest and Gairezi scenario mapping

Compiled 2026-11-02 to redirect the displacement sweep campaign from shell
degradation (shown to be benign) toward the mechanisms that actually fail arch
dams. Full source URLs retained inline.

## Why shell damage is the wrong target

The hydraulic sweeps against job `8bef80bd1c324e808a7358271a5b578f` showed that
degrading even the 2000 most-stressed wall cells leaves peak displacement on the
intact linear curve (~8.5 mm at full load) until static equilibrium is lost
abruptly; weakening accelerates the instability but never adds compliance. This
matches the historical record: no arch dam failure is attributed to local shell
damage or openings. Kariba's gated openings are not its risk — decades of
spillway-jet scour forming an 80–90 m plunge pool that could undermine the
foundation is (Kariba Dam Rehabilitation Project,
https://international-partnerships.ec.europa.eu/policies/programming/projects/kariba-dam-rehabilitation-project-kdrp_en).

## Documented failure-cause distribution (concrete dams)

ICOLD Bulletin 99 (2019 update, ~55 recorded concrete-dam failures):
foundation/abutment deficiency ~53%; structural/materials ~20%; overtopping ~9%;
spillway failure ~9%; uplift ~7%; seismic ~2%
(https://donlingoldseis.com/docs/draft-seis-references/ICOLD%202019.pdf,
https://www.usbr.gov/damsafety/risk/BestPractices/Presentations/D7-ConcreteDamFoundationsPP.pdf).
USACE EM 1110-2-2201 (Arch Dam Design): sliding on weak planes/discontinuities in
the foundation or abutments is the most common cause of historical arch dam
failures; no arch dam has failed from seismic load alone
(https://www.publications.usace.army.mil/Portals/76/Publications/EngineerManuals/EM_1110-2-2201.pdf).

## Canonical cases

- **Malpasset, France 1959** (66.5 m thin arch; 424 deaths): abutment rock wedge
  bounded by a foliation plane and an undetected fault; first-filling water
  pressurized the planes (stress-sensitive permeability, no drainage), uplift
  rotated the wedge out, and the structurally faultless arch collapsed within
  seconds of losing its support
  (https://environment.uwe.ac.uk/geocal/totalgeology/histories/case12.htm,
  https://damfailures.org/case-study/malpasset-dam-france-1959,
  https://www.barrages-cfbr.eu/IMG/pdf/2022_06_04_cfbr_malpasset_geology_engineering_humanities_2018-186.full.pdf).
- **St. Francis, USA 1928** (gravity; canonical foundation-geology case):
  saturated schist over a reactivated ancient landslide plus clay/gypsum seams;
  translational foundation sliding; uplift ignored in design
  (https://damfailures.org/sites/default/files/wp-pdf/120_Reassessment-of-the-St.-Francis-Dam-Failure.pdf).
- **Vajont, Italy 1963**: 260+ million m³ landslide into the reservoir overtopped
  the 262 m arch by ~100 m or more; ~2000 deaths downstream — yet the arch
  survived because its abutments (competent limestone) kept their load path
  (https://damfailures.org/case-study/vajont-dam-italy-1963). Lesson: abutment
  integrity is everything; reservoir-rim slope stability is a separate hazard.
- **Kölnbrein, Austria 1978–79** (200 m arch, near-miss): base cracking on first
  fillings with uplift pressures on the order of full reservoir head inside the
  cracks and up to 200 L/s seepage; required elastic re-grouting and ultimately a
  500,000 m³ downstream thrust block (https://en.wikipedia.org/wiki/K%C3%B6lnbrein_Dam,
  https://onemine.org/documents/rehabilitation-of-the-k-lnbrein-dam-austria).
  Standard conservative criterion: a through-crack carries full reservoir head as
  uplift along the entire cracked plane.
- **Meihua, China 1981** (masonry arch with perimetral joint): failed in overload
  (actual depth 20.8 m); yielding spread from bank-slope sections and the joint
  corner, final failure by sliding along the joint plus rotation about the
  abutment (https://www.waterpowermagazine.com/news/analysis-of-the-failure-of-meihua-arch-dam-with-perimetral-joint/).
- **Matilija, USA** (50 m thin arch): progressive alkali-silica reaction distress;
  remediated by notching, not a collapse
  (https://www.engr.psu.edu/ce/courses/ce584/concrete/library/chemical/asrcase.html).

Unverifiable cases: "Lake Lanier" and "Moyie River" arch failures (no evidence
found); Idbar (Bosnia) anecdotal only.

## Modeling consensus for quasi-static FE

Credible failure simulation represents degradation of foundation/abutment
boundary conditions — interface strength/stiffness loss, uplift increase,
hydrostatic overload — rather than arch-shell material damage
(USACE EM 1110-2-2100; https://www.mdpi.com/2076-3417/16/14/7281;
https://www.mdpi.com/2075-5309/16/3/567). Scour is represented by its structural
end state (removed/softened support), not by erosion hydraulics.

## Gairezi scenario mapping

The model's supports are linear springs, so interface strength reduction is
approximated by stiffness release (the structural end state), consistent with the
support-loss representation above.

| Priority | Mechanism | Real case | Gairezi representation |
|---|---|---|---|
| S1 | Abutment support loss | Malpasset | Scale `SPR_OUT` elastic-layer support matrices on one flank (angular sector of the seat) by a ladder of factors; undegraded control must reproduce the intact baseline within 1% |
| S2 | Foundation contact sliding/debonding | St. Francis | Scale `SPR_WF` wall–foundation interface spring stiffnesses (whole seat or upstream half); undrained uplift already applied |
| S3 | Base crack + crack uplift | Kölnbrein | Bank-to-bank cohesive layer (`--layer-scope`) at an explicit base elevation; crack-water pressure already applies reservoir head inside hydraulically connected open faces |
| S4 | Hydrostatic overload | Meihua | Extend the hydraulic load schedule beyond factor 1.0; report the limit load factor at loss of equilibrium as the failure metric (replaces the 100 mm displacement proxy, which the sweeps showed is unreachable on the stable branch) |
| Excluded | Seismic, AAR, scour hydraulics, shell holes | — | Out of scope: dynamics unsupported; AAR is a long-term serviceability process; scour enters only as S1-style support loss; shell openings are not a credible global mode |
