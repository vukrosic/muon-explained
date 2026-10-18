"""Pop the first line of a queue file under a lock (two workers share one queue). Exit 1 when empty."""
import fcntl, sys
with open(sys.argv[1], "r+") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    lines = f.readlines()
    if not lines: sys.exit(1)
    f.seek(0); f.writelines(lines[1:]); f.truncate()
    print(lines[0].strip())
