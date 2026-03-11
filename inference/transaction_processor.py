class TransactionProcessor:
    def __init__(self, architect, orchestrator, gatekeeper, analyst, threshold):
        self.architect = architect
        self.orchestrator = orchestrator
        self.gatekeeper = gatekeeper
        self.analyst = analyst
        self.threshold = threshold

    async def execute(self, row, scaled_output):  # Added scaled_output here
        # Pass the scaled data into the architect
        task = self.architect.create_investigation_task(row, scaled_output)

        raw_output = await self.orchestrator.execute_with_resilience(task)

        # Ensure gatekeeper handles the dictionary of scores correctly
        final_scores = self.gatekeeper.apply_final_verdict(raw_output, row['amt'])

        score = final_scores['total']

        # HITL Logic: Check if score is in the "Grey Area"
        # Example: If Threshold is 0.3, range is [0.25 to 0.30]
        is_hitl = (self.threshold - 0.05) <= score < self.threshold

        decision = "FRAUD" if score >= self.threshold else "NORMAL"
        if is_hitl:
            decision = "PENDING_REVIEW"

        # Generate XAI for Fraud or HITL
        xai_story = "No deep-dive analysis performed (Low Risk)."
        if score > (self.threshold - 0.1):  # Lowered to capture HITL cases
            xai_story = await self.analyst.generate_explanation(
                scores=final_scores,
                agent_context=raw_output
            )

        return {
            "final_decision": decision,
            "scores": final_scores,
            "xai_report": xai_story,
            "is_hitl": is_hitl
        }