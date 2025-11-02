import sys

def main():
    mods = [
        ("flask", "__version__"),
        ("flask_cors", None),
        ("dotenv", None),
        ("requests", "__version__"),
        ("matplotlib", "__version__"),
        ("numpy", "__version__"),
        ("torch", "__version__"),
        ("PIL", None),
        ("transformers", "__version__"),
        ("arango", None),
    ]
    for name, attr in mods:
        try:
            m = __import__(name)
            ver = getattr(m, attr) if attr else "OK"
            print(f"{name}: {ver}")
        except Exception as e:
            print(f"{name}: ERROR -> {e}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
