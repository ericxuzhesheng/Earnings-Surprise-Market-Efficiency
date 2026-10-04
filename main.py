"""The default entry point uses the auditable frozen-source validation workflow."""
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-pipeline", action="store_true", help="Historical pipeline; does not publish current validated conclusions")
    parser.add_argument("--refresh-factors", action="store_true")
    parser.add_argument("--baseline", action="store_true", help="Reproduce the preserved 300-stock frozen audit")
    args = parser.parse_args()
    if args.legacy_pipeline:
        from src.pipeline import run_pipeline
        run_pipeline()
    else:
        if args.refresh_factors and not args.baseline:
            parser.error("Expanded source collection uses scripts/expand_event_data.py prices; --refresh-factors is only for --baseline")
        from scripts.run_event_validation import run
        run(refresh_factors=args.refresh_factors, expanded=not args.baseline)
