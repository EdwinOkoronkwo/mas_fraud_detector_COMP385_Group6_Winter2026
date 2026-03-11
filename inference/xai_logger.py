import json
import os
from datetime import datetime


class XAILogger:
    """Handles the persistence and summarization of AI explainability reports."""

    def __init__(self, base_dir="logs/xai_reports"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    # Inside your XAILogger.save_report method
    def save_report(self, cc_last4, text_report, numerical_scores):
        import os
        import json
        from datetime import datetime

        timestamp = datetime.now().strftime("%H%M%S")

        # FORCE UTF-8 encoding to handle arrows, emojis, and special math symbols
        text_path = os.path.join(self.base_dir, f"{cc_last4}_{timestamp}_report.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text_report)

        json_path = os.path.join(self.base_dir, f"{cc_last4}_{timestamp}_data.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(numerical_scores, f, indent=4)

        return text_path

    def get_summary(self, report_content: str) -> str:
        return report_content[:100].replace("\n", " ").strip() + "..."