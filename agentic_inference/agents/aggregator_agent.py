class AgenticAggregator:
    @staticmethod
    def fuse_decision(trinity_score, policy_risk, threshold=0.3):
        # Weighted fusion
        p_score = policy_risk['policy_score']
        final_consensus = (trinity_score * 0.7) + (p_score * 0.3)

        # Decision logic
        is_fraud = final_consensus >= threshold

        # Strategic Weighting (Per NET-004)
        # If Trinity is high AND policy is critical, bump to 1.0
        if trinity_score > 0.6 and p_score >= 0.95:
            final_consensus = 1.0

        return {
            "final_score": round(final_consensus, 3),
            "verdict": "🚨 FRAUD" if is_fraud else "✅ CLEAR",
            "citations": policy_risk['codes_cited']
        }