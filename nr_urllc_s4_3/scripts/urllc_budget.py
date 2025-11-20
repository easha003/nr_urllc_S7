
# scripts/urllc_budget.py
import argparse, yaml, json, sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from nr_urllc.urllc import compute_budget

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', required=True, help='YAML config with urllc: and nr:/ofdm: blocks')
    p.add_argument('--out', default='artifacts/urllc_budget.json', help='Output JSON file path')
    args = p.parse_args()

    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f) or {}

    urllc_cfg = cfg.get('urllc', {})
    if not urllc_cfg:
        raise SystemExit("Config has no 'urllc:' block.")

    block = compute_budget(urllc_cfg, cfg_all=cfg)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(block, f, indent=2)
    print('[info] URLLC budget written to', args.out)

if __name__ == '__main__':
    main()
