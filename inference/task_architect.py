import pandas as pd


class TaskArchitect:
    """Translates raw transaction data into natural language tasks for AI agents.

    This class handles the "Prompt Engineering" layer, injecting dynamic
    sensitivity instructions based on transaction metadata like amount.
    """

    def create_investigation_task(self, row: pd.Series) -> str:
        """Constructs a detailed prompt for the RAG Investigator Team.

        Args:
            row: A pandas Series representing a single transaction record.

        Returns:
            A formatted string containing the investigation task and
            amount-based sensitivity notes.
        """
        cc_str = str(row['cc_num']).split('.')[0]
        amt = row['amt']

        sensitivity_note = ""
        if amt > 1000:
            sensitivity_note = "CRITICAL: High-value transaction. Prioritize RNN anomalies."
        elif 300 < amt <= 1000:
            sensitivity_note = "NOTICE: Use high-sensitivity threshold for borderline patterns."

        return (f"Investigate CC {cc_str} for a transaction of ${amt:.2f}. "
                f"{sensitivity_note} "
                "Provide a breakdown for RF, RNN, and DBSCAN, and the final 'TOTAL RISK SCORE'.")