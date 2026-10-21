# Dam Analysis Workbench - Test Plan & Verification Suite

## 1. Test Environment & Target Machine Profile

This test plan is designed for end-to-end acceptance testing on a clean, secondary machine (no Visual Studio, no Git, no standalone Python, and no developer SDKs pre-installed).

### Guinea Pig Machine Specifications
- **Model:** ASUS TUF GAMING
- **Processor:** AMD Ryzen 7 3750H with Radeon Vega Mobile Gfx @ 2.30 GHz
- **Installed RAM:** 16.0 GB
- **Dedicated GPU:** 4 GB
- **System Type:** 64-bit Operating System, x64-based processor
- **Operating System:** Windows 11 Home
- **Display Scaling:** Test at 100% and recommended DPI scaling (e.g., 125% / 150%)
- **Network State:** Verify both Online (Wi-Fi connected) and Offline (Airplane mode)

---

## 2. Test Execution Matrix

| Suite ID | Focus Area | Objective | Pass / Fail |
|---|---|---|---|
| **TS-01** | Media & AutoPlay | Verify USB drive branding, icon, and installation launcher | [ ] |
| **TS-02** | Installer Deployment | Verify WiX MSI installation, UAC elevation, and file payload | [ ] |
| **TS-03** | Startup & Backend Lifecycle | Verify process auto-spawn, loopback port binding, token security, and exit teardown | [ ] |
| **TS-04** | UI Navigation & Styling | Verify dark theme, screen switching, and high-DPI rendering | [ ] |
| **TS-05** | Configuration & Validation | Verify default config, parameter editing, schema validation, and file I/O | [ ] |
| **TS-06** | Job Submission & Execution | Verify job lifecycle, log monitoring, and solver dependency handling | [ ] |
| **TS-07** | History & Results | Verify persistence across restarts and ParaView integration | [ ] |
| **TS-08** | Negative & Stress Testing | Verify crash resilience, port collision, and corrupted inputs | [ ] |
| **TS-09** | Maintenance & Uninstall | Verify Add/Remove Programs cleanup and user data preservation | [ ] |

---

## 3. Detailed Test Cases

### TS-01: USB Flash Drive & AutoPlay Verification

* **Pre-conditions:** USB drive (`E:\` or formatted `D:\`) prepared with installer media and `autorun.inf`.
* **Test Steps:**
  1. Insert the USB drive into a USB port on the ASUS TUF laptop.
  2. Open **File Explorer** -> **This PC**.
  3. Observe the drive icon and label.
  4. Double-click the drive or run `Install-Workbench.bat`.
* **Expected Results:**
  - [ ] The drive displays the custom Lamb dam icon (`desktop_icon.ico`) and label `Lamb Dam Analysis Workbench`.
  - [ ] If AutoPlay prompt appears, clicking it triggers the installer.
  - [ ] Double-clicking or executing `Install-Workbench.bat` launches the setup wizard without path errors.

---

### TS-02: Installation & Payload Integrity

* **Pre-conditions:** Clean Windows 11 environment.
* **Test Steps:**
  1. Launch `Lamb.Windows.Installer.msi` (or `Lamb.Windows.Installer.Bundle.exe`).
  2. Observe the WiX installer dialogs (navy/red branding banner and dialog bitmaps).
  3. Accept UAC elevation prompt when requested.
  4. Select or verify default destination: `C:\Program Files\Lamb\DamAnalysisWorkbench\`.
  5. Check "Create desktop shortcut" and click **Install**.
  6. Inspect installation directory after completion.
* **Expected Results:**
  - [ ] Installer completes with exit status 0 (no rollback or generic 1603 errors).
  - [ ] Desktop shortcut `Lamb Dam Analysis Workbench` is created with the custom icon.
  - [ ] Start Menu entry is created under `Lamb Dam Analysis Workbench`.
  - [ ] `C:\Program Files\Lamb\DamAnalysisWorkbench\` contains:
    - [ ] `Lamb.Windows.App.exe` (self-contained WPF application)
    - [ ] `backend\` directory (containing `python.exe`, `run_server.py`, `app/`, `Elmer/`, `structural_model/`, `FEniCSx/`, and `config.json`)
    - [ ] `third-party-manifest.json`

---

### TS-03: Startup & Managed Backend Lifecycle

* **Pre-conditions:** Application installed; no developer servers running.
* **Test Steps:**
  1. Open **Task Manager** and switch to the **Details** tab.
  2. Launch **Lamb Dam Analysis Workbench** from the desktop shortcut.
  3. Observe processes created in Task Manager.
  4. Observe Home screen connection status within 5 seconds of launch.
  5. Close the application by clicking the top-right `X` window button.
  6. Observe Task Manager processes again.
* **Expected Results:**
  - [ ] Application window opens promptly with the dark theme.
  - [ ] A background `python.exe` process is spawned automatically (invisible window, no console popups).
  - [ ] File `%LOCALAPPDATA%\Lamb\DamAnalysisWorkbench\api_token.txt` is generated.
  - [ ] File `%LOCALAPPDATA%\Lamb\DamAnalysisWorkbench\logs\api.log` records server startup: `Uvicorn running on http://127.0.0.1:8721`.
  - [ ] Home screen status displays: **"Connected to Lamb.API."** and shows API Version (e.g., `0.1.0`).
  - [ ] When the main window is closed, the child `python.exe` process is terminated immediately. No orphan processes remain in Task Manager.

---

### TS-04: UI Navigation & Layout Verification

* **Pre-conditions:** Application running and connected to API.
* **Test Steps:**
  1. Click through the left sidebar navigation items:
     - **Home**
     - **Configuration**
     - **Run Results**
     - **History**
  2. Test resizing the window from minimum size to maximized on the 1080p display.
  3. Test with Windows display scaling set to 125% and 150%.
* **Expected Results:**
  - [ ] Screen transitions are instantaneous with no visual glitching or freezing.
  - [ ] Colors conform to design mock-up: dark navy background (`#0D1B2A`), red accents (`#C0392B`), crisp light typography.
  - [ ] Controls and contour grids scale cleanly without text truncation or clipped buttons at non-100% DPI.

---

### TS-05: Configuration Editor & Live Schema Validation

* **Pre-conditions:** On **Configuration** screen.
* **Test Steps:**
  1. **Default Load:** Verify default dam parameters are populated (Wall radii, Material properties, Water levels, Bedrock contours).
  2. **Valid Edit:** Change `water_height` to `30.0` and `overflow_head` to `1.5`. Observe status bar and JSON Preview.
  3. **File Save As:** Click `File` -> `Save As...`, save to `Documents\test_dam_config.json`.
  4. **Invalid Input (Negative Radius):** Enter a negative inner radius or a wall thickness of zero. Observe validation banner.
  5. **Invalid Input (String in Number Field):** Enter alphabetic characters into a numeric box.
  6. **File Open:** Click `File` -> `Open...`, open the saved `test_dam_config.json`.
* **Expected Results:**
  - [ ] JSON preview updates live as fields are modified.
  - [ ] Valid modifications report configuration as valid.
  - [ ] Invalid values trigger clear, inline validation messages identifying the exact field path (e.g. `geometry.wall.inner_radius_m`).
  - [ ] Saved JSON file matches schema and reloads identically into the UI.

---

### TS-06: Job Submission & Solver Execution

* **Pre-conditions:** Valid configuration active; on Configuration screen.
* **Scenario A: Clean Machine without Elmer / MS-MPI (Prerequisite Detection)**
  1. Verify Home screen badges: `ElmerGrid: Not available` and `ElmerSolver: Not available`.
  2. Click **Submit Job**.
  3. Observe transition to **Run Results** screen and log panel.
  * **Expected Result:**
    - [ ] App transitions smoothly to Run Results.
    - [ ] System reports a graceful failure explaining ElmerGrid/ElmerSolver executable is not found on PATH.
    - [ ] Application does **not** crash or freeze; status shows `failed` with diagnostic error message.

* **Scenario B: Machine with Elmer & MS-MPI Installed**
  1. Install Elmer and MS-MPI (or configure bundled dependencies).
  2. Verify Home screen displays green/ready indicators for ElmerGrid and ElmerSolver.
  3. Click **Submit Job**.
  4. Monitor live log output and stage progression:
     - Stage: `validating`
     - Stage: `meshing`
     - Stage: `converting`
     - Stage: `solving`
     - Stage: `postprocessing`
  * **Expected Result:**
    - [ ] Job transitions to `succeeded`.
    - [ ] Real-time solver logs stream into the log viewer without blocking the UI.
    - [ ] Results file `combined_results_mpi_t0001.pvtu` and analysis plots/JSON are created in `%LOCALAPPDATA%\Lamb\DamAnalysisWorkbench\jobs\<job-id>\`.
    - [ ] If crack simulations were enabled, FEniCSx scenario execution chains automatically.

---

### TS-07: Job History & Results Viewing

* **Pre-conditions:** At least one job has been submitted (either completed or failed).
* **Test Steps:**
  1. Navigate to **History** screen.
  2. Verify the submitted job appears in the history list with Job ID, status badge, and timestamp.
  3. Click the job to view details.
  4. Restart the desktop application.
  5. Navigate back to **History** screen.
  6. Click **Open Results in ParaView** (test both when ParaView is present and when absent).
* **Expected Results:**
  - [ ] History loads accurately from persistent local storage.
  - [ ] Job history survives complete application restarts.
  - [ ] If ParaView is installed, it launches displaying the mesh/solution.
  - [ ] If ParaView is not installed, a friendly toast notification informs the user rather than throwing an unhandled exception.

---

### TS-08: Resilience & Edge-Case Testing

* **Pre-conditions:** Application running.
* **Test Steps:**
  1. **Port Conflict:** Start an external process on port 8721 before launching the app, then launch the app.
  2. **Corrupted Token:** Delete `%LOCALAPPDATA%\Lamb\DamAnalysisWorkbench\api_token.txt` while app is running, then click Reconnect on Home.
  3. **Rapid Restart:** Close and reopen the application rapidly 3 times.
  4. **Offline Mode:** Turn off Wi-Fi (Airplane mode) and run configuration validation and job submission.
* **Expected Results:**
  - [ ] In port conflict, app detects port in use, attempts graceful connection, and notifies user if token mismatch occurs.
  - [ ] Reconnecting regenerates/reads token cleanly without crash.
  - [ ] Rapid restarts do not leave orphaned processes or lock errors on `api.log`.
  - [ ] Completely functional offline (since API runs strictly on loopback `127.0.0.1`).

---

### TS-09: Maintenance & Uninstallation

* **Pre-conditions:** Application installed and tested.
* **Test Steps:**
  1. Close the application.
  2. Open Windows **Settings** -> **Apps** -> **Installed apps**.
  3. Find **Lamb Dam Analysis Workbench** and click **Uninstall**.
  4. Inspect `C:\Program Files\Lamb\DamAnalysisWorkbench\` and shortcuts.
  5. Inspect `%LOCALAPPDATA%\Lamb\DamAnalysisWorkbench\`.
* **Expected Results:**
  - [ ] Uninstaller completes cleanly.
  - [ ] Desktop and Start Menu shortcuts are removed.
  - [ ] Binaries under `C:\Program Files\Lamb\DamAnalysisWorkbench\` are completely removed.
  - [ ] User job history in `%LOCALAPPDATA%` is preserved (not destroyed upon uninstall).

---

## 4. Diagnostic Runbook: Log File Locations

When a bug or unexpected behavior occurs during guinea pig laptop testing, collect the following files:

| Artifact | Location on Target Machine | Description |
|---|---|---|
| **API Server Log** | `%LOCALAPPDATA%\Lamb\DamAnalysisWorkbench\logs\api.log` | Python server console, error tracebacks, and uvicorn output |
| **API Access Token** | `%LOCALAPPDATA%\Lamb\DamAnalysisWorkbench\api_token.txt` | Current local authentication token |
| **Job Execution Data** | `%LOCALAPPDATA%\Lamb\DamAnalysisWorkbench\jobs\<job-id>\` | Job configuration snapshot, raw solver logs, and metrics |
| **MSI Installer Log** | `%TEMP%\lamb_install.log` | Verbose Windows Installer transaction log (if installed via `/l*v`) |

### Quick PowerShell Diagnostic Collector
To bundle all diagnostic logs on the target laptop into a single zip file for review:
```powershell
$logDir = "$env:LOCALAPPDATA\Lamb\DamAnalysisWorkbench"
$destZip = "$env:USERPROFILE\Desktop\Lamb_Diagnostics_$([DateTime]::Now.ToString('yyyyMMdd_HHmmss')).zip"
Compress-Archive -Path "$logDir\logs\*", "$logDir\jobs\*" -DestinationPath $destZip -Force
Write-Host "Diagnostics saved to $destZip"
```
