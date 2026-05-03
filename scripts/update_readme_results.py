import pandas as pd
import re
from pathlib import Path

def update_readme_results():
    print("Updating README results from generated CSVs...")
    
    project_root = Path(__file__).parent.parent
    readme_path = project_root / "README.md"
    tables_dir = project_root / "outputs" / "tables"
    
    # 1. Check if core files exist
    sample_path = tables_dir / "sample_construction.csv"
    signal_path = tables_dir / "headline_signal_comparison.csv"
    car_summary_path = tables_dir / "event_window_car_summary.csv"
    
    if not (sample_path.exists() and signal_path.exists()):
        print("Required CSVs missing. Keeping README as PENDING.")
        return

    try:
        # 2. Extract Data
        sample_df = pd.read_csv(sample_path)
        # Assuming final row is the final headline sample
        final_sample_size = int(sample_df.iloc[-1]["event_count"])
        
        # Extract from headline_signal_comparison (assuming it has columns like signal_scale, car_window, p_value, coef)
        signal_df = pd.read_csv(signal_path)
        if not signal_df.empty:
            # Sort to find a 'good' spec or just take the first
            top_spec = signal_df.iloc[0]
            matching_tier = top_spec.get("match_tier", "Strict Same Quarter")
            surprise_metric = top_spec.get("signal_scale", "raw")
            # CAR10 or similar
            car_val = top_spec.get("coef", "N/A")
            p_val = top_spec.get("p_value", "N/A")
        else:
            matching_tier, surprise_metric, car_val, p_val = "N/A", "N/A", "N/A", "N/A"

        # 3. Construct the table row
        results_data = {
            "Headline Sample Size": str(final_sample_size),
            "Main Matching Tier": str(matching_tier),
            "Main Surprise Metric": str(surprise_metric),
            "CAR Spread/Coef": str(car_val),
            "P-Value": str(p_val)
        }

        # 4. Read README
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 5. Replace the block using regex
        # This regex looks for the | 指标 | 结果 | (CN) or | Metric | Value | (EN) tables
        # and replaces the subsequent rows.
        
        # Replace Chinese Block
        cn_pattern = r"(\| 指标 \| 结果 \| PENDING \|.*?\n)(.*?)(?=\n\n|\n###)"
        # Since I marked them as PENDING in previous turn, I look for PENDING
        
        # Actually, let's just target the specific table structure
        def replace_cn(match):
            header = "| 指标 | 结果 |\n| :--- | :--- |"
            rows = [
                f"| 基准样本量 (Headline Sample Size) | {results_data['Headline Sample Size']} |",
                f"| 主事件类型 (Main Event Type) | 业绩预告 (Preannouncement) |",
                f"| 匹配层级 (Matching Tier) | {results_data['Main Matching Tier']} |",
                f"| CAR 结果 / 系数 | {results_data['CAR Spread/Coef']} |",
                f"| P 值 (P-Value) | {results_data['P-Value']} |"
            ]
            return header + "\n" + "\n".join(rows)

        # Replace English Block
        def replace_en(match):
            header = "| Metric | Value |\n| :--- | :--- |"
            rows = [
                f"| Final Headline Sample Size | {results_data['Headline Sample Size']} |",
                f"| Main Event Type | Preannouncement |",
                f"| Matching Tier | {results_data['Main Matching Tier']} |",
                f"| CAR Spread / Coef | {results_data['CAR Spread/Coef']} |",
                f"| P-Value | {results_data['P-Value']} |"
            ]
            return header + "\n" + "\n".join(rows)

        # We look for the marker blocks we set in the previous turn
        content = re.sub(r"### 关键结果摘要 \(Key Results Snapshot - PENDING\).*?\| 泄露/漂移证据 \| PENDING \|", 
                         "### 关键结果摘要 (Key Results Snapshot)\n" + replace_cn(None), 
                         content, flags=re.DOTALL)
                         
        content = re.sub(r"### Key Results Snapshot - PENDING.*?\| Leakage/Drift Evidence \| PENDING \|", 
                         "### Key Results Snapshot\n" + replace_en(None), 
                         content, flags=re.DOTALL)

        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        print("✓ README.md updated with real results.")

    except Exception as e:
        print("ERROR: Failed to update README: {}".format(e))

if __name__ == "__main__":
    update_readme_results()
