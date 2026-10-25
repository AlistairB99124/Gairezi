"""Check provisional damage material inputs with the installed native Aster macro.

This is a material-admissibility sensitivity check, NOT a dam cracking solve.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile
import time

from Aster.fracture_material import ROOT, material_contract
from Aster.run_case import detect_installation, file_hash, run_solver
from Elmer.model_config import CONFIG_PATH, load_config


def sensitivity_contracts(config: dict) -> list[dict]:
    nominal = material_contract(config)
    width = nominal["parameters"]["LARG_BANDE"]
    p = nominal["parameters"]["P"]
    return [
        material_contract(config, band_width_m=width * factor, softening_p=shape)
        for factor in (.5, 1., 2.)
        for shape in (max(1., p - 3), p, p + 3)
    ]


def command_text(contracts: list[dict]) -> str:
    parameters = [contract["parameters"] for contract in contracts]
    return (
        "from code_aster.Commands import *\n"
        "from libaster import AsterError\n"
        "import json\n"
        "DEBUT(ERREUR=_F(ALARME='EXCEPTION'))\n"
        f"inputs = {parameters!r}\n"
        "responses = []\n"
        "for parameters in inputs:\n"
        "    try:\n"
        "        material = DEFI_MATER_GC(ENDO_FISS_EXP=_F(**parameters))\n"
        "    except AsterError as error:\n"
        "        responses.append({'parameters': parameters, 'rejected': {\n"
        "            'message_id': str(getattr(error, 'id_message', '')), 'text': str(error)}})\n"
        "        continue\n"
        "    internal = {key: material.getValueReal('ENDO_FISS_EXP', key)\n"
        "                for key in ('M', 'P', 'Q', 'K', 'SIG0', 'TAU')}\n"
        "    internal['C_GRAD_VARI'] = material.getValueReal('NON_LOCAL', 'C_GRAD_VARI')\n"
        "    responses.append({'parameters': parameters, 'internal': internal})\n"
        "with open('fort.90', 'w', encoding='utf-8') as stream:\n"
        "    json.dump(responses, stream, indent=2, allow_nan=False)\n"
        "FIN()\n"
    )


def validate_responses(contracts: list[dict], responses: object,
                       nominal: dict | None = None) -> dict:
    """Require the nominal case to pass; other cases may only fail the native M >= P + 2 rule.

    Within one grid only GF/LARG_BANDE vary at fixed E, NU, FT and FC, so the native
    M * LARG_BANDE is invariant. Every rejection is checked against that invariant.
    """
    if not isinstance(responses, list) or len(responses) != len(contracts):
        raise ValueError("Native material check returned an incomplete sensitivity grid")
    nominal = contracts[len(contracts) // 2]["parameters"] if nominal is None else nominal
    accepted, rejected, invariants = [], [], []
    for contract, response in zip(contracts, responses):
        if not isinstance(response, dict) or not isinstance(
                response.get("internal", response.get("rejected")), dict):
            raise ValueError("Native material check returned a malformed response")
        if response["parameters"] != contract["parameters"]:
            raise ValueError("Native material check changed the requested parameters")
        p = contract["parameters"]
        if "rejected" in response:
            if "COMPOR1_94" not in (response["rejected"].get("message_id", "")
                                    + response["rejected"].get("text", "")):
                raise ValueError("Native material check rejected a case for an unexpected reason")
            rejected.append(p)
            continue
        internal = response["internal"]
        for name in ("M", "P", "Q", "K", "SIG0", "TAU", "C_GRAD_VARI"):
            value = internal[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Native material check returned invalid {name}")
        if internal["M"] < p["P"] + 2:
            raise ValueError("Native material check violated M >= P + 2")
        expected = {
            "P": p["P"], "Q": p["Q"],
            "K": 1.5 * p["GF"] / p["LARG_BANDE"],
            "C_GRAD_VARI": .1875 * p["GF"] * p["LARG_BANDE"],
        }
        if any(not math.isclose(internal[key], value, rel_tol=1e-10, abs_tol=1e-12)
               for key, value in expected.items()):
            raise ValueError("Native material check disagrees with the documented band conversion")
        accepted.append(p)
        invariants.append(internal["M"] * p["LARG_BANDE"])
    if nominal not in accepted:
        raise ValueError("Native material check rejected the nominal material")
    if any(not math.isclose(value, invariants[0], rel_tol=1e-10) for value in invariants):
        raise ValueError("Native material check M * LARG_BANDE is not invariant across the grid")
    for p in rejected:
        if invariants[0] / p["LARG_BANDE"] >= p["P"] + 2:
            raise ValueError("Native material check rejected a case that satisfies M >= P + 2")
    return {
        "accepted_cases": len(accepted),
        "inadmissible_cases": [
            {"LARG_BANDE": p["LARG_BANDE"], "P": p["P"],
             "predicted_M": invariants[0] / p["LARG_BANDE"], "required_M": p["P"] + 2}
            for p in rejected
        ],
        "m_times_band_width_m": invariants[0],
        "maximum_admissible_band_width_m_at_nominal_p": invariants[0] / (nominal["P"] + 2),
    }


def check(config_path: Path, output_dir: Path, time_limit_s: int = 120) -> dict:
    config = load_config(config_path)
    contracts = sensitivity_contracts(config)
    installation = detect_installation()
    if not installation["available"]:
        raise RuntimeError("Native material verification requires an installed Code_Aster launcher")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="material-", dir=output_dir.resolve()))
    case_path = run_dir / "material.comm"
    export_path = run_dir / "material.export"
    response_path = run_dir / "native_materials.json"
    case_path.write_text(command_text(contracts), encoding="utf-8")
    export_path.write_text(
        f"P time_limit {time_limit_s}\nP memory_limit 1024\nP mpi_nbcpu 1\n"
        f"F comm {case_path} D 1\nF libr {response_path} R 90\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": 1,
        "status": "running",
        "scope": "native_material_admissibility_only_not_a_cracking_solve",
        "installation": installation,
        "configuration_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, allow_nan=False).encode("utf-8")
        ).hexdigest(),
        "case_sha256": file_hash(case_path),
        "nominal": material_contract(config),
        "sensitivity_contracts": contracts,
        "run_dir": str(run_dir),
    }
    report_path = run_dir / "material_check.json"
    started = time.monotonic()
    try:
        completed = run_solver([installation["launcher_path"], str(export_path)], run_dir, time_limit_s)
        (run_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
        (run_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
        exceptions = set(re.findall(r"<EXCEPTION>\s*<(\w+)>", completed.stdout + completed.stderr))
        if completed.returncode != 0 or exceptions - {"COMPOR1_94"} or (
                "<EXCEPTION>" in completed.stdout + completed.stderr and not exceptions):
            raise RuntimeError(f"Native material verification failed; inspect {run_dir}")
        if not response_path.is_file():
            raise RuntimeError(f"Native material verification produced no response; inspect {run_dir}")
        responses = json.loads(response_path.read_text(encoding="utf-8"))
        report["admissibility"] = validate_responses(
            contracts, responses, report["nominal"]["parameters"])
        if bool(exceptions) != bool(report["admissibility"]["inadmissible_cases"]):
            raise RuntimeError(f"Native material exceptions disagree with recorded responses; inspect {run_dir}")
        report["native_responses"] = responses
        report["status"] = "material_admissibility_passed"
    except (RuntimeError, ValueError, KeyError, OSError, subprocess.TimeoutExpired) as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        report["elapsed_s"] = time.monotonic() - started
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "Aster" / "results" / "fracture_material")
    args = parser.parse_args()
    report = check(args.config, args.output_dir)
    print(json.dumps({
        "status": report["status"], "scope": report["scope"],
        "sensitivity_cases": len(report["native_responses"]),
        "admissibility": report["admissibility"], "run_dir": report["run_dir"],
    }, indent=2))


if __name__ == "__main__":
    main()
