"""
Detached experiment launcher for Windows.
Spawns the experiment runner as a truly detached process that survives shell exit.
Writes PID and logs to results/ for monitoring.
"""
import subprocess
import sys
import os
import json
import pathlib
import argparse
from datetime import datetime, timedelta
from collections import Counter


def _is_pid_alive(pid: int) -> bool:
    """Cross-platform check if a PID is alive."""
    try:
        import ctypes
        PROCESS_QUERY_INFORMATION = 0x0400
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_INFORMATION, False, pid)
        if handle == 0:
            return False
        exit_code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        ctypes.windll.kernel32.CloseHandle(handle)
        STILL_ACTIVE = 259
        return exit_code.value == STILL_ACTIVE
    except Exception:
        return False


def _bar(done: int, total: int, width: int = 40) -> str:
    filled = int(width * done / total) if total else 0
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _fmt_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _filter_log(lines: list[str]) -> list[str]:
    """Remove Data Summary blocks (header, content lines, separators)."""
    import re
    NOISE = [
        re.compile(r"^=+$"),                          # pure separator lines
        re.compile(r"Data Summary"),                   # Data Summary header
        re.compile(r"^\w+:\s+[\d,]+ trading days"),   # "SPY: 2,515 trading days..."
    ]
    out = []
    for ln in lines:
        stripped = ln.strip()
        if any(p.search(stripped) for p in NOISE):
            continue
        out.append(ln)
    return out


def _load_run_metrics(result_dir: str) -> dict:
    """Load test + val metrics from a completed run's output directory."""
    rd = pathlib.Path(result_dir)
    metrics = {}
    for split, fname in [("test", "test_results.json"), ("val", "val_results.json")]:
        fp = rd / fname
        if fp.exists():
            try:
                d = json.loads(fp.read_text(encoding="utf-8"))
                metrics[f"{split}_sharpe"] = d.get("sharpe_ratio")
                metrics[f"{split}_mdd"]    = d.get("max_drawdown")
                metrics[f"{split}_sortino"] = d.get("sortino_ratio")
            except Exception:
                pass
    return metrics


def _print_status(pid_file: pathlib.Path, log_file: pathlib.Path) -> None:
    W = 70
    print("=" * W)
    print("  E1 EXPERIMENT STATUS")
    print("=" * W)

    # --- Process ---
    alive = False
    pid = None
    if pid_file.exists():
        pid = int(pid_file.read_text().strip())
        alive = _is_pid_alive(pid)
        state = "RUNNING" if alive else "STOPPED"
        print(f"  Process : PID {pid}  [{state}]")
    else:
        print("  Process : no experiment tracked")

    # --- Manifest ---
    m = pathlib.Path("results/experiments/manifest.json")
    if not m.exists():
        print("  Manifest: not found")
        print("=" * W)
        return

    data    = json.loads(m.read_text(encoding="utf-8"))
    runs    = data.get("runs", [])
    total   = len(runs)
    counts  = Counter(r["status"] for r in runs)
    n_done  = counts.get("completed", 0)
    n_fail  = counts.get("failed", 0)
    n_pend  = counts.get("pending", 0)

    # Start time from log
    started_str = ""
    if log_file.exists():
        first = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
        for ln in first[:5]:
            if ln.startswith("Started:"):
                try:
                    started_str = ln.split("Started:")[1].strip()[:16].replace("T", " ")
                except Exception:
                    pass
                break
    if started_str:
        print(f"  Started : {started_str}")

    # Elapsed / ETA
    completed_runs = [r for r in runs if r["status"] == "completed"]
    wall_times = [r.get("wall_time_seconds", 0) for r in completed_runs if r.get("wall_time_seconds")]
    avg_sec = sum(wall_times) / len(wall_times) if wall_times else None

    if started_str:
        try:
            started_dt = datetime.fromisoformat(started_str.replace(" ", "T"))
            elapsed = (datetime.now() - started_dt).total_seconds()
            print(f"  Elapsed : {_fmt_duration(elapsed)}", end="")
            if avg_sec and n_pend > 0:
                eta_sec = avg_sec * n_pend
                print(f"  |  ETA ~{_fmt_duration(eta_sec)} remaining", end="")
            print()
        except Exception:
            pass

    if avg_sec:
        print(f"  Avg/run : {_fmt_duration(avg_sec)}")

    # Progress bar
    print()
    pct = 100 * n_done // total if total else 0
    print(f"  PROGRESS  {_bar(n_done, total)}  {n_done}/{total}  ({pct}%)")
    print(f"  Completed: {n_done}  |  Failed: {n_fail}  |  Pending: {n_pend}  |  Total: {total}")

    # --- Completed runs table ---
    if completed_runs:
        print()
        print(f"  {'COMPLETED RUNS':^66}")
        hdr = f"  {'Algorithm':<10} {'Strategy':<20} {'Seed':>6}  {'TestSharpe':>10}  {'ValSharpe':>9}  {'TestMDD':>7}  {'Time':>7}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for r in completed_runs:
            mx = _load_run_metrics(r.get("result_dir", ""))
            ts  = f"{mx['test_sharpe']:+.3f}" if mx.get("test_sharpe") is not None else "  N/A "
            vs  = f"{mx['val_sharpe']:+.3f}"  if mx.get("val_sharpe")  is not None else "  N/A "
            mdd = f"{mx['test_mdd']*100:.1f}%" if mx.get("test_mdd")   is not None else "  N/A "
            wt  = _fmt_duration(r.get("wall_time_seconds", 0))
            print(f"  {r['algorithm']:<10} {r['strategy']:<20} {r['seed']:>6}  {ts:>10}  {vs:>9}  {mdd:>7}  {wt:>7}")

    # --- Failed runs ---
    failed_runs = [r for r in runs if r["status"] == "failed"]
    if failed_runs:
        print()
        print(f"  FAILED RUNS ({len(failed_runs)})")
        for r in failed_runs[:5]:
            err = r.get("error", "")[:60]
            print(f"  [!] {r['run_id']}  -- {err}")
        if len(failed_runs) > 5:
            print(f"      ... and {len(failed_runs)-5} more")

    # --- Next pending ---
    pending_runs = [r for r in runs if r["status"] == "pending"]
    if pending_runs:
        print()
        print(f"  NEXT PENDING ({len(pending_runs)} remaining)")
        for r in pending_runs[:5]:
            print(f"    - {r['algorithm']:<6} / {r['strategy']:<20} / seed {r['seed']}")
        if len(pending_runs) > 5:
            print(f"      ... and {len(pending_runs)-5} more")

    # --- Recent log (filtered) ---
    if log_file.exists():
        raw_lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
        clean = _filter_log(raw_lines)
        recent = [ln for ln in clean if ln.strip()][-15:]
        if recent:
            print()
            print("  RECENT LOG")
            print("  " + "-" * 60)
            for ln in recent:
                print("  " + ln)

    print()
    print("=" * W)
    print(f"  Monitor: python scripts/launch_experiment.py --status")
    print(f"  Stop:    python scripts/launch_experiment.py --kill")
    print("=" * W)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", default=["replay_comparison"])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--kill", action="store_true", help="Kill running experiment process")
    parser.add_argument("--status", action="store_true", help="Show process status")
    args = parser.parse_args()

    pid_file = pathlib.Path("results/experiment.pid")
    log_file = pathlib.Path("results/experiment.log")
    pathlib.Path("results").mkdir(exist_ok=True)

    # --- Kill mode ---
    if args.kill:
        if pid_file.exists():
            pid = int(pid_file.read_text().strip())
            if _is_pid_alive(pid):
                import ctypes
                handle = ctypes.windll.kernel32.OpenProcess(1, False, pid)
                ctypes.windll.kernel32.TerminateProcess(handle, 1)
                ctypes.windll.kernel32.CloseHandle(handle)
                print(f"Terminated PID {pid}")
            else:
                print(f"Process {pid} already stopped")
            pid_file.unlink()
        else:
            print("No PID file found")
        return

    # --- Status mode ---
    if args.status:
        _print_status(pid_file, log_file)
        return

    # --- Launch mode ---
    if pid_file.exists():
        pid = int(pid_file.read_text().strip())
        if _is_pid_alive(pid):
            print(f"Experiment already running (PID {pid}). Use --kill to stop it.")
            return
        else:
            pid_file.unlink()

    python_exe = sys.executable
    cmd = [
        python_exe, "-u", "scripts/run_experiments.py",
        "--experiments", *args.experiments,
        "--workers", str(args.workers),
    ]
    if args.timesteps:
        cmd += ["--timesteps", str(args.timesteps)]

    log_f = open(log_file, "w", encoding="utf-8")
    log_f.write(f"Started: {datetime.now().isoformat()}\n")
    log_f.write(f"Command: {' '.join(cmd)}\n\n")
    log_f.flush()

    # CREATE_NEW_PROCESS_GROUP: process survives parent shell exit on Windows.
    # Do NOT use DETACHED_PROCESS — it strips the console context and crashes Intel MKL
    # (used by NumPy) with "forrtl: error (200): program aborting due to window-CLOSE event".
    # Instead, hide the window via STARTUPINFO so it runs silently in the background.
    CREATE_NEW_PROCESS_GROUP = 0x00000200
    STARTF_USESHOWWINDOW = 0x00000001
    SW_HIDE = 0

    si = subprocess.STARTUPINFO()
    si.dwFlags = STARTF_USESHOWWINDOW
    si.wShowWindow = SW_HIDE

    proc = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        creationflags=CREATE_NEW_PROCESS_GROUP,
        startupinfo=si,
        close_fds=True,
    )

    pid_file.write_text(str(proc.pid))
    print(f"Launched experiment (PID {proc.pid})")
    print(f"Log: {log_file.absolute()}")
    print(f"Monitor: python scripts/launch_experiment.py --status")
    print(f"Stop:    python scripts/launch_experiment.py --kill")


if __name__ == "__main__":
    main()
