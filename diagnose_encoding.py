import os

def diag(filepath):
    if not os.path.isfile(filepath):
        print(f"File not found: {filepath}")
        return
    with open(filepath, "rb") as f:
        raw = f.read()
    print(f"File: {filepath}")
    print(f"Size: {len(raw)} bytes")
    print(f"First 20 bytes (hex): {raw[:20].hex()}")
    has_bom = raw[:3] == b'\xef\xbb\xbf'
    print(f"Has UTF-8 BOM: {has_bom}")
    print(f"First 20 bytes (repr): {raw[:20]!r}")
    if has_bom:
        try:
            text = raw[3:].decode("utf-8")
            print(f"Content (BOM stripped): {text[:100]}...")
        except:
            print("Failed to decode as UTF-8")
    else:
        try:
            text = raw.decode("utf-8")
            print(f"Content (no BOM): {text[:100]}...")
        except:
            print("Failed to decode as UTF-8")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        diag(sys.argv[1])
    else:
        print("Usage: python diagnose_encoding.py <filepath>")
