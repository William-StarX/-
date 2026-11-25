"""Download only the official frozen ViT-S/14 backbone and verify its exact bytes."""
import argparse
import hashlib
from pathlib import Path
import urllib.request

URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/backbone/dinov2_vits14_pretrain.pth"))
    args = parser.parse_args()
    if args.output.exists() and digest(args.output) == SHA256:
        print("Verified " + str(args.output))
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".download")
    urllib.request.urlretrieve(URL, temporary)
    if digest(temporary) != SHA256:
        temporary.unlink()
        raise ValueError("Official backbone SHA-256 mismatch")
    temporary.replace(args.output)
    print("Downloaded and verified " + str(args.output))


if __name__ == "__main__":
    main()
