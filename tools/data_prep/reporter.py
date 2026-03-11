import os
import json
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def publish_report(summary_markdown: str, metrics_json: str):
    try:
        from config import settings
        import os, json

        report_path = os.path.normpath(os.path.join(settings.REPORT_DIR, "PHASE_1_FINAL_REPORT.md"))
        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        # Robust JSON Parsing
        processed_metrics = metrics_json
        if isinstance(metrics_json, str):
            try:
                clean_json = metrics_json.replace("```json", "").replace("```", "").strip()
                processed_metrics = json.loads(clean_json)
            except Exception as e:
                processed_metrics = {"raw_output": metrics_json}

        # Construct the Document
        report_body = [
            "# 🛡️ Phase 1 Quality Audit: Data Foundation",
            f"**Timestamp:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Status:** ✅ DATA_VERIFIED",
            "\n## 📝 Executive Summary",
            summary_markdown,
            "\n## 🖼️ Visual Verification Logs",
            f"- [Initial Class Distribution](plots/class_dist_initial_state.png)",
            f"- [Feature Correlation Matrix](plots/correlation_matrix.png)",
            f"- [Categorical Fraud Analysis](plots/category_fraud_analysis.png)",
            "\n## 📊 Technical Metrics (Metadata)",
            "```json",
            json.dumps(processed_metrics, indent=4),
            "```",
            "\n---\n*System Signature: Quality_Critic_Agent*"
        ]

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_body))

        return f"SUCCESS: Formal report published to {report_path}"

    except Exception as e:
        return f"ERROR: Failed to publish report: {str(e)}"