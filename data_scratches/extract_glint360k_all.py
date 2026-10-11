"""Stream the glint360k tar parts (gunzipped, on stdin) and write EVERY identity
as <out_root>/<label>/<NNNN>.jpg (1-indexed per-identity counter in record
order -- same naming as the original glint360k_images tree and its eye-center
metadata). Identities listed in --skip-manifest (the already-restored 36k/9k
arch-comparison subset) are skipped. Parses MXNet RecordIO directly (no mxnet).

Usage:
  cat glint360k_0[0-6] | pigz -dc | python3 extract_glint360k_all.py \
      <skip_manifest.json> <out_root> <counts_out.json>
"""
import sys, os, json, struct, tarfile, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

skip_manifest, out_root, counts_path = sys.argv[1], sys.argv[2], sys.argv[3]
_m = json.load(open(skip_manifest))
existing = set(_m["train_ids"]) | set(_m["val_ids"])

MAGIC = 0xCED7230A
counters = defaultdict(int)
made_dirs = set()
pool = ThreadPoolExecutor(48)
pending = []

def write(path, data):
    with open(path, "wb") as f:
        f.write(data)

def read_exact(f, n):
    buf = f.read(n)
    while len(buf) < n:
        more = f.read(n - len(buf))
        if not more:
            raise EOFError
        buf += more
    return buf

def records(f):
    while True:
        try:
            magic, lrec = struct.unpack("<II", read_exact(f, 8))
        except EOFError:
            return
        assert magic == MAGIC, hex(magic)
        cflag, length = lrec >> 29, lrec & ((1 << 29) - 1)
        data = read_exact(f, length)
        read_exact(f, (4 - length % 4) % 4)
        if cflag == 0:
            yield data
        elif cflag == 1:
            parts = [data]
        else:  # continuation: the split point was a magic word
            parts.append(struct.pack("<I", MAGIC) + data)
            if cflag == 3:
                yield b"".join(parts)

t0, n_rec, n_out = time.time(), 0, 0
tf = tarfile.open(fileobj=sys.stdin.buffer, mode="r|")
for member in tf:
    print("member:", member.name, member.size, flush=True)
    if not member.name.endswith("train.rec"):
        continue
    f = tf.extractfile(member)
    for rec in records(f):
        n_rec += 1
        if n_rec % 500000 == 0:
            print(f"records={n_rec} written={n_out} ids={len(counters)} {time.time()-t0:.0f}s", flush=True)
        flag, label, _, _ = struct.unpack("<IfQQ", rec[:24])
        img_off = 24 + 4 * flag
        if rec[img_off:img_off + 2] != b"\xff\xd8":
            continue  # header / identity-range records (no JPEG payload)
        lab = int(struct.unpack("<f", rec[24:28])[0]) if flag > 0 else int(label)
        if str(lab) in existing:
            continue
        counters[lab] += 1
        d = os.path.join(out_root, str(lab))
        if d not in made_dirs:
            os.makedirs(d, exist_ok=True)
            made_dirs.add(d)
        pending.append(pool.submit(write, os.path.join(d, "%04d.jpg" % counters[lab]), rec[img_off:]))
        n_out += 1
        if len(pending) > 20000:
            for p in pending: p.result()
            pending = []

for p in pending: p.result()
pool.shutdown()
json.dump({str(k): v for k, v in counters.items()}, open(counts_path, "w"))
print(f"DONE records={n_rec} written={n_out} ids={len(counters)} {time.time()-t0:.0f}s", flush=True)
