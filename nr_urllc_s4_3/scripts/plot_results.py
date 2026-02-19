# scripts/plot_results.py
import argparse
from nr_urllc.plots import plot_multi_from_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", nargs="+", required=True, help="One or more result JSON files")
    ap.add_argument("--labels", nargs="*", default=None, help="Legend labels (optional)")
    ap.add_argument("--out", default="artifacts/overlay_curves.png", help="Output PNG")
    ap.add_argument("--title", default="BER / EVM / MSE vs SNR (overlay)", help="Figure title")
    ap.add_argument("--show", action="store_true", help="Show window")
    args = ap.parse_args()

    plot_multi_from_json(args.json, labels=args.labels, title=args.title, save_path=args.out, show=args.show)

if __name__ == "__main__":
    main()
