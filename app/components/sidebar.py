import streamlit as st


class SidebarComponent:
    def render(self):
        with st.sidebar:
            st.header("🎛️ Calibration")
            n_center = st.slider("Neuro Sigma Center", 0.01, 0.40, 0.10)
            c_center = st.slider("Cluster Sigma Center", 0.5, 5.0, 1.8)
            batch_size = st.number_input("Batch Size", 10, 500, 100)

            st.divider()
            if st.button("♻️ Reset Agent Trust", key="reset_btn"):
                st.session_state.pipeline.scorer.reset_weights()
                st.success("Weights Reset!")

            return {
                "n_center": n_center,
                "c_center": c_center,
                "batch_size": batch_size
            }