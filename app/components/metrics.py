import streamlit as st
import pandas as pd


class MetricsComponent:
    @staticmethod
    def render_performance_comparison(df: pd.DataFrame, threshold: float = 0.30):
        """Displays the 'Baseline vs. MAS' gap with full Statistical rigor."""
        st.divider()
        st.header("📊 Multi-Agent System (MAS) Performance Delta")

        def get_stats(col):
            tp = ((df[col] >= threshold) & (df['ACT'] == 1)).sum()
            tn = ((df[col] < threshold) & (df['ACT'] == 0)).sum()
            fp = ((df[col] >= threshold) & (df['ACT'] == 0)).sum()
            fn = ((df[col] < threshold) & (df['ACT'] == 1)).sum()

            pre = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (pre * rec) / (pre + rec) if (pre + rec) > 0 else 0
            acc = (tp + tn) / len(df)
            return acc, rec, f1, fn

        # Extract stats for both systems
        b_acc, b_rec, b_f1, b_fn = get_stats('BASE')
        m_acc, m_rec, m_f1, m_fn = get_stats('MATH')

        # 1. Executive KPI Row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAS Accuracy", f"{m_acc:.1%}", delta=f"{(m_acc - b_acc):.1%}")
        c2.metric("Fraud Recall", f"{m_rec:.1%}", delta=f"{(m_rec - b_rec):.1%}")
        c3.metric("F1-Score", f"{m_f1:.2f}", delta=f"{(m_f1 - b_f1):.2f}")
        c4.metric("Missed Frauds (FN)", int(m_fn), delta=int(m_fn - b_fn), delta_color="inverse")

        # 2. Detailed Metric Comparison Chart
        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "Recall", "F1-Score"],
            "Baseline": [b_acc, b_rec, b_f1],
            "MAS Neural": [m_acc, m_rec, m_f1]
        }).melt(id_vars="Metric", var_name="System", value_name="Score")

        st.vega_lite_chart(metrics_df, {
            'mark': {'type': 'bar', 'tooltip': True},
            'encoding': {
                'x': {'field': 'Metric', 'type': 'nominal', 'axis': {'labelAngle': 0}},
                'y': {'field': 'Score', 'type': 'quantitative', 'format': '.0%'},
                'color': {'field': 'System', 'scale': {'range': ['#475569', '#d946ef']}},
                'xOffset': {'field': 'System'}
            }
        }, use_container_width=True)

    @staticmethod
    def render_trust_evolution(weight_history: list):
        """Renders the 'Agent Trust Curve' directly in the UI."""
        st.subheader("🤖 Dynamic Agent Trust Evolution")
        if not weight_history:
            st.warning("No weight history available yet.")
            return

        history_df = pd.DataFrame(weight_history)
        st.line_chart(history_df)
        st.caption(
            "This curve represents how the system dynamically re-weights agents based on their real-time performance.")

    @staticmethod
    def render_trust_weights(weights: dict):
        """Displays the CURRENT trust percentages for the active agents."""
        st.subheader("⚖️ Current Agent Allocations")
        if not weights:
            st.info("Waiting for first batch to calculate weights...")
            return

        cols = st.columns(len(weights))
        for i, (agent, val) in enumerate(weights.items()):
            # Clean up names for display (e.g., 'w_gold' -> 'Gold Pillar')
            display_name = agent.replace('w_', '').replace('_', ' ').title()
            cols[i].metric(display_name, f"{val:.1%}")

    @staticmethod
    def render_agent_trace(trace_data: list):
        """Renders the internal agent dialogue in a technical console."""
        with st.expander("🛠️ View Agent Reasoning Trace (Policy & SQL Lookups)", expanded=False):
            st.markdown("---")
            for log in trace_data:
                agent = log['agent'].upper()
                content = log['content']

                if "SQL" in agent:
                    st.write(f"📂 **[{agent}]**")
                    st.code(content, language="json")
                elif "VECTOR" in agent:
                    st.write(f"📜 **[{agent}]**")
                    st.info(content)
                else:
                    st.write(f"⚙️ **[{agent}]**")
                    st.caption(content)
                st.markdown("---")