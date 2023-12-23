#!/usr/bin/python3
#
# This program is here because we need the user/group in the container to match the
# user/group of the bind_dir so that the files will appear as the proper user in the host
# filesystem. In order to accomplish this, we change the app user/group to match the user
# and group of the bind_dir. In the lucky and common case that it already matches, this step
# is skipped. This problem primarily occurs on Linux where there can be multiple users on a
# machine and the uid/gid does not have to be 1000/1000. This problem doesn't occur on MacOs/Linux
# because they handle the bind dir differenly.

import os
import platform
import subprocess
import sys

from pathlib import Path

print("Version-Info, Aryn Jupyter Branch:", os.environ.get("GIT_BRANCH", "missing"))
print("Version-Info, Aryn Jupyter Commit:", os.environ.get("GIT_COMMIT", "missing"))
print("Version-Info, Aryn Jupyter Diff:", os.environ.get("GIT_DIFF", "missing"))
print("Version-Info, Aryn Jupyter Architecture:", platform.uname().machine, flush=True)

def main():
    app_stat = os.lstat("/app")
    bind_stat = os.lstat("/app/work/bind_dir")
    if app_stat.st_uid != bind_stat.st_uid or app_stat.st_gid != bind_stat.st_gid:
        fix_ids(bind_stat.st_uid, bind_stat.st_gid)
    else:
        print("uid", app_stat.st_uid, "and gid", app_stat.st_gid,
              "already match between /app and /app/work/bind_dir")
        
    exec_run_jupyter(bind_stat.st_uid, bind_stat.st_gid)

def fix_ids(uid, gid):
    if uid == 0 or gid == 0:
        testname = "/app/work/bind_dir/.fixuser.test"
        subprocess.run(["sudo", "-u", "app", "touch", testname])
        if Path(testname).is_file():
            subprocess.run(["sudo", "-u", "app", "rm", testname])
            print("WARNING: On a machine doing weirdness with the bind_mount.")
            print("WARNING: owner or group is root, but able to write files as app.")
            print("WARNING: this happens on MacOS with a special driver, so leaving ids alone.")
            return
        raise Exception("Refusing to change id to uid == 0 or gid == 0\n" +
                        "Make sure bind dir has a non-root uid and gid")
    print("WARNING: Fixing IDs. This step can take a long time", flush=True)
    print("  Fixing app group to have gid", gid, flush=True)
    subprocess.run(["groupmod", "--gid", str(gid), "app"])
    print("  Fixing app user to have uid", uid, "and gid", gid, flush=True)
    subprocess.run(["usermod", "--uid", str(uid), "--gid", str(gid), "app"])
    print("  Running recursive chown", flush=True)
    subprocess.run(["chown", "-R", "app:app", "/app"])
    print("SUCCESS: uid & gid fixed", flush=True)

def exec_run_jupyter(uid, gid):
    sys.stdout.flush()
    sys.stderr.flush()
    args = ["/usr/bin/sudo", "-E", "-u", "app", "/app/run-jupyter.sh"] + sys.argv[1:]
    os.execv(args[0], args)

main()
