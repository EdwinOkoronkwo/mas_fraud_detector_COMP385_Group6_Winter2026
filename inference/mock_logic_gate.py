# Mock of your extraction tool
from inference.LogicGate import LogicGatekeeper


def extract_detailed_scores(text):
    # Simulating what your regex would do
    import re
    scores = {}
    for m in ['lr', 'rnn', 'db']:
        match = re.search(fr"{m.upper()}: ([\d.]+)", text)
        if match: scores[m] = float(match.group(1))
    return scores

# --- TEST EXECUTION ---
gate = LogicGatekeeper(current_accuracies={'lr': 0.90, 'rnn': 0.85, 'db': 0.70})

# Scenario 1: The Low-Value Shield
# Fraud score is 0.85 (High), but amount is only $10.
raw_output_1 = "LR: 0.85, RNN: 0.40, DB: 0.30"
verdict_1 = gate.apply_final_verdict(raw_output_1, amt=10.0)

# Scenario 2: The Synergistic Extreme Anomaly
# LR missed it (0.10), but RNN and DB both say it's weird (0.85). High Value ($600).
raw_output_2 = "LR: 0.10, RNN: 0.85, DB: 0.85"
verdict_2 = gate.apply_final_verdict(raw_output_2, amt=600.0)

print(f"Scenario 1 (Shield) -> Total Score: {verdict_1['Total']} | Label: {verdict_1.get('Override')}")
print(f"Scenario 2 (Synergy) -> Total Score: {verdict_2['Total']} | Label: {verdict_2.get('Override')}")