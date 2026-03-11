
class PerformanceTracker:
    """Encapsulates accuracy calculations and pillar performance tracking."""

    def __init__(self, threshold: float):
        self.threshold = threshold
        # ADDED 'baseline' to the monitored pillars
        self.stats = {
            p: {'corr': 0, 'total': 0}
            for p in ['baseline', 'supervised', 'neural', 'clustering', 'total']
        }

    def update(self, scores: dict, actual_is_fraud: bool):
        for pillar in self.stats:
            if pillar in scores:
                score = scores.get(pillar, 0.0)
                # Logic: If score > threshold, we predict FRAUD.
                # If that matches actual_is_fraud, the model was correct.
                is_correct = (score >= self.threshold) == actual_is_fraud
                self.stats[pillar]['total'] += 1
                if is_correct:
                    self.stats[pillar]['corr'] += 1

    def get_final_accuracies(self) -> dict:
        # Returns a clean dict of {pillar: 0.XX} for the ReportGenerator
        return {
            p: (s['corr'] / s['total']) if s['total'] > 0 else 0.0
            for p, s in self.stats.items()
        }