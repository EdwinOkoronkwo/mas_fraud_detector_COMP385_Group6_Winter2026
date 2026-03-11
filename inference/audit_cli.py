
class AuditCLI:
    """Handles the terminal presentation of the audit trail."""
    @staticmethod
    def print_log(results):
        print("\n" + "═" * 80)
        print("📂 XAI AUDIT TRAIL (Reports Saved to /logs/xai_reports/)")
        print("═" * 80)
        for res in results:
            if res["xai_path"] != "N/A":
                print(f"📄 Card: {res['CC (Last 4)']} | Path: {res['xai_path']}")
                print(f"   Summary: {res['xai_summary']}\n")
            else:
                print(f"⚪ Card: {res['CC (Last 4)']} | Low Risk - No Deep Dive Generated.")