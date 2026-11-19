To model brittle concrete cracking at 1.12 MPa in Code_Aster, use the ENDO_LOCA_TC isotropic damage behavior law alongside the BETON_ECRO_LINE keyword in DEFI_MATERIAU. This model is robust, fast, and lets you directly specify the tensile cracking limit (SYT).
Below is a complete, ready-to-use .comm file snippet written in standard SI units (Meters, Pascals, Kilograms).

1. Material Definition (DEFI_MATERIAU)

Set SYT = 1.12E6 (1.12 MPa) to trigger cracking at your exact tensile threshold, and give D_SIGM_EPSI a steep negative slope to represent brittle softening after the peak.
python
# Define concrete properties in Pa, kg/m3, m
beton = DEFI_MATERIAU(
    ELAS=_F(
        E=25.0E9,        # Young's Modulus (25 GPa)
        NU=0.2,          # Poisson's Ratio
        RHO=2350.0,      # Density (2350 kg/m3)
    ),
    BETON_ECRO_LINE=_F(
        SYT=1.12E6,          # Tensile cracking limit (1.12 MPa)
        SYC=15.0E6,          # Compressive strength limit (15 MPa)
        D_SIGM_EPSI=-5.0E9,  # Post-peak softening slope (negative)
    ),
)
Use code with caution.

2. Nonlinear Solver Setup (STAT_NON_LINE)

To activate the damage and cracking behavior, assign RELATION='ENDO_LOCA_TC' inside COMPORTEMENT when running STAT_NON_LINE.
python
res = STAT_NON_LINE(
    MODELE=model,
    CHAM_MATER=field_mat,
    EXCIT=(
        _F(CHARGE=bc_fix),
        _F(CHARGE=load, FONC_MULT=ramp),
    ),
    COMPORTEMENT=_F(
        RELATION='ENDO_LOCA_TC',
    ),
    INCREMENT=_F(
        LIST_INST=time_steps,
    ),
    NEWTON=_F(
        MATRICE='TANGENTE',
        REAC_ITER=1,
    ),
)
Use code with caution.

Key Modeling & Convergence Tips

• Consistent units. Make sure your mesh dimensions match your material units. If your mesh is modeled in millimeters instead of meters, enter E = 25000.0, SYT = 1.12, and D_SIGM_EPSI = -5000.0 (MPa).
• Post-peak softening (D_SIGM_EPSI). This parameter controls how abruptly the concrete loses strength after reaching 1.12 MPa. If the solver fails to converge immediately after cracking starts, make D_SIGM_EPSI less steep (for example, change -5.0E9 to -1.0E9) or use displacement control instead of force control.
• Viewing cracks in ParaVis. In ENDO_LOCA_TC, the first internal variable (V1 in field VARI_ELGA or VARI_NOEU) represents the scalar damage variable D. A value of 0 means undamaged concrete, and 1 indicates a fully developed crack.
• Alternative damage models. If you need separate tension and compression damage evolution or cyclic behavior, you can also look into the Mazars damage model (RELATION='MAZARS'), where you set the tensile cracking strain threshold EPSD0 equal to SYT / E.
If you'd like help refining this setup, let me know:
• Your mesh unit system (meters or millimeters)
• Whether your loading is force-controlled or displacement-controlled
• If you need mesh-independent regularization (NON_LOCAL / 3D_GRAD_VARI) to prevent strain localization
I can adjust the .comm script or solver settings to prevent convergence issues during crack propagation.
