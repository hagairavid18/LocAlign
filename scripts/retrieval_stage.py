"""Node-local staging of the packed retrieval shards.

The shards (~84 GB, packed_features/shard_XXX.pkl) live on NFS, and reading
them is the bottleneck on some GPU nodes (~3-6 MB/s measured on n-h200, against
~30-40 MB/s on a CPU node). ShardStager keeps one verified copy per node under
<stage_root>/packed_<digest>/ (default stage_root /tmp/localign_retrieval_<user>,
on the node's local disk; this cluster has no per-job /tmp cleanup, so the copy
persists for later jobs on the same node).

  * Reuse: if <stage_root>/packed_<digest>/COMPLETE exists and every shard has
    the expected size, shards are read from there. <digest> is derived from
    packed_features/checksums.json, so a re-packed store gets a new directory.
  * Copy: otherwise, if the local disk has room (shards + 20 GB margin) and no
    other job on the node is copying (non-blocking flock on
    <stage_root>/.stage.lock), shards are copied with n parallel streams, in
    the order the job needs them, into packed_<digest>.incomplete/. Each file
    is SHA-256-checked against checksums.json before it is renamed into place,
    so verified files survive a preemption and are not copied again. When all
    shards are present the directory is renamed to packed_<digest>/ and
    COMPLETE is written. The job uses each shard as soon as its copy is done.
  * Follow: if another job on the node holds the lock (is copying), this job
    waits for each verified shard to appear locally instead of competing for
    NFS bandwidth; if the copier dies, the remaining shards are read from NFS.
  * Fallback: no checksums.json or not enough space -> shards are read from NFS.
Shard contents are byte-identical in all modes, so scores do not depend on it.
Only files under <stage_root> are ever created or removed. To free the space on
a node: rm -rf /tmp/localign_retrieval_<user> (from a job on that node).
"""
import fcntl
import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

CHUNK = 16 * 1024 * 1024


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(CHUNK), b''):
            h.update(b)
    return h.hexdigest()


class ShardStager:
    def __init__(self, packed, shard_order, stage_root=None, n_streams=8, margin_gb=20, log=print):
        self.packed, self.log = packed, log
        self.mode, self.dir = 'nfs', packed
        self._futures, self._lock_fd = {}, None
        ck_path = os.path.join(packed, 'checksums.json')
        if stage_root is None or not os.path.exists(ck_path):
            log(f'[stage] disabled ({"no stage_root" if stage_root is None else "no checksums.json"}); reading shards from NFS')
            return
        with open(ck_path) as f:
            self.ck = json.load(f)  # {"shard_000.pkl": {"size": int, "sha256": str}, ...}
        digest = hashlib.sha1(json.dumps(self.ck, sort_keys=True).encode()).hexdigest()[:12]
        os.makedirs(stage_root, exist_ok=True)
        self.final = os.path.join(stage_root, f'packed_{digest}')
        self.work = self.final + '.incomplete'
        total = sum(v['size'] for v in self.ck.values())
        if self._valid(self.final):
            self.mode, self.dir = 'local', self.final
            log(f'[stage] using verified local copy {self.final}')
            return
        free = shutil.disk_usage(stage_root).free
        have = sum(os.path.getsize(os.path.join(self.work, n)) for n in self.ck
                   if os.path.exists(os.path.join(self.work, n))) if os.path.isdir(self.work) else 0
        if free + have < total + margin_gb * 2**30:
            log(f'[stage] not enough local space ({free / 2**30:.0f} GB free, need {total / 2**30:.0f}+{margin_gb} GB); NFS')
            return
        self._lock_fd = open(os.path.join(stage_root, '.stage.lock'), 'w')
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock_fd.close()
            self._lock_fd = None
            # Follow the other job's copy instead of competing with it for NFS bandwidth:
            # use each verified shard as soon as it appears locally.
            self.mode, self._lock_path = 'follow', os.path.join(stage_root, '.stage.lock')
            log('[stage] another job on this node is copying; using its verified shards as they arrive')
            return
        if self._valid(self.final):  # finished by another job while we checked
            self._release()
            self.mode, self.dir = 'local', self.final
            return
        os.makedirs(self.work, exist_ok=True)
        for n in os.listdir(self.work):  # leftovers of an interrupted copy of ours
            if n.endswith('.part'):
                os.remove(os.path.join(self.work, n))
        self.mode, self.dir = 'copying', self.work
        names = [f'shard_{si:03d}.pkl' for si in shard_order]
        names += sorted(n for n in self.ck if n not in names)
        self._t0, self._bytes = time.time(), 0
        self._pool = ThreadPoolExecutor(max_workers=n_streams)
        for n in names:
            self._futures[n] = self._pool.submit(self._copy, n)
        log(f'[stage] copying {len(names)} shards ({total / 2**30:.0f} GB) to {self.work} with {n_streams} streams')

    def _valid(self, d):
        if not os.path.exists(os.path.join(d, 'COMPLETE')):
            return False
        return all(os.path.exists(os.path.join(d, n)) and os.path.getsize(os.path.join(d, n)) == v['size']
                   for n, v in self.ck.items())

    def _copy(self, name):
        dst = os.path.join(self.work, name)
        exp = self.ck[name]
        if os.path.exists(dst) and os.path.getsize(dst) == exp['size']:
            return True  # verified and renamed by an earlier (interrupted) job
        part = dst + '.part'
        h = hashlib.sha256()
        with open(os.path.join(self.packed, name), 'rb') as fi, open(part, 'wb') as fo:
            for b in iter(lambda: fi.read(CHUNK), b''):
                h.update(b)
                fo.write(b)
        if h.hexdigest() != exp['sha256'] or os.path.getsize(part) != exp['size']:
            os.remove(part)
            self.log(f'[stage] checksum mismatch for {name}; this shard will be read from NFS')
            return False
        os.replace(part, dst)
        self._bytes += exp['size']
        return True

    def _copier_alive(self):
        with open(self._lock_path, 'w') as fd:
            try:
                fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(fd, fcntl.LOCK_UN)
            return False

    def path(self, si):
        name = f'shard_{si:03d}.pkl'
        if self.mode == 'follow':
            size = self.ck[name]['size']
            while True:
                for d in (self.final, self.work):  # verified files only ever appear under these names
                    p = os.path.join(d, name)
                    if os.path.exists(p) and os.path.getsize(p) == size:
                        return p
                if not self._copier_alive():
                    return os.path.join(self.packed, name)
                time.sleep(10)
        if self.mode == 'copying':
            ok = self._futures[name].result()
            return os.path.join(self.work, name) if ok else os.path.join(self.packed, name)
        return os.path.join(self.dir, name)

    def finish(self):
        if self.mode != 'copying':
            return
        ok = all(f.result() for f in self._futures.values())
        dt = time.time() - self._t0
        self.log(f'[stage] copied {self._bytes / 2**30:.1f} GB in {dt:.0f}s ({self._bytes / 2**20 / max(dt, 1):.1f} MB/s)')
        if ok and all(os.path.getsize(os.path.join(self.work, n)) == v['size'] for n, v in self.ck.items()):
            with open(os.path.join(self.work, 'COMPLETE'), 'w') as f:
                json.dump({'time': time.ctime(), 'job': os.environ.get('SLURM_JOB_ID')}, f)
            if os.path.exists(self.final):  # an invalid (e.g. truncated) earlier copy of ours
                shutil.rmtree(self.final)
            os.replace(self.work, self.final)
            self.log(f'[stage] local copy complete: {self.final}')
        self._release()

    def _release(self):
        if self._lock_fd is not None:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            self._lock_fd.close()
            self._lock_fd = None


def write_checksums(packed, workers=8):
    names = sorted(n for n in os.listdir(packed) if n.startswith('shard_') and n.endswith('.pkl'))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        sums = dict(zip(names, pool.map(lambda n: sha256_file(os.path.join(packed, n)), names)))
    ck = {n: {'size': os.path.getsize(os.path.join(packed, n)), 'sha256': sums[n]} for n in names}
    tmp = os.path.join(packed, 'checksums.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(ck, f, indent=1, sort_keys=True)
    os.replace(tmp, os.path.join(packed, 'checksums.json'))
    print(f'wrote checksums for {len(ck)} shards')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--write_checksums', default='/home/iscb/wolfson/hagairavid/LocAlign_retrieval_work/packed_features')
    write_checksums(ap.parse_args().write_checksums)
