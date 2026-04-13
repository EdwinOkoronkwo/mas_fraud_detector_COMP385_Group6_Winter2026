import streamlit as st
import asyncio
import pandas as pd
from data.database import SessionLocal
from data.models.customer import Customer


class SimulatorView:
    def __init__(self, controller):
        self.controller = controller

    def render(self, ui_params):
        st.header("🧪 Transaction Simulator")

        # 1. Fetch existing customers for the dropdown
        db = SessionLocal()
        customers = db.query(Customer).all()
        db.close()

        # 2. Customer Selection Logic
        cust_options = {c.customer_name: c for c in customers}
        selected_name = st.selectbox(
            "👤 Select Target Cardholder Profile",
            options=["-- Manual Entry --"] + list(cust_options.keys())
        )

        # 3. Auto-fill logic
        selected_cust = cust_options.get(selected_name)

        with st.form("manual_tx_form"):
            st.subheader("Transaction Details")
            c1, c2, c3 = st.columns(3)

            # Use customer defaults if selected, otherwise standard defaults
            cc_val = selected_cust.cc_num if selected_cust else "4111222233334444"
            u_lat_val = selected_cust.home_lat if selected_cust else 43.6532
            u_long_val = selected_cust.home_long if selected_cust else -79.3832

            amt = c1.number_input("Amount ($)", value=450.0)
            cat = c2.selectbox("Category", ["shopping_net", "food_dining", "grocery_pos", "entertainment"])
            gender = c3.selectbox("Gender", ["M", "F", "Non-binary"])

            st.divider()
            st.subheader("Geography & Identity")
            c4, c5, c6 = st.columns(3)
            cc_num = c4.text_input("CC Number", value=cc_val)
            u_lat = c5.number_input("User Lat", value=u_lat_val, format="%.4f")
            u_long = c6.number_input("User Long", value=u_long_val, format="%.4f")

            st.caption("📍 Tip: Change 'Merchant Lat/Long' below to test the distance-based RAG Audit.")
            c7, c8 = st.columns(2)
            m_lat = c7.number_input("Merchant Latitude", value=43.7000, format="%.4f")
            m_long = c8.number_input("Merchant Longitude", value=-79.4000, format="%.4f")

            submitted = st.form_submit_button("🛡️ Run Real-Time Audit")

            if submitted:
                tx_data = self._build_tx_dict(amt, cat, gender, cc_num, u_lat, u_long, m_lat, m_long)

                with st.spinner(f"Agents are analyzing behavior for {selected_name}..."):
                    res = asyncio.run(self.controller.run_single_inference(tx_data, ui_params))
                    self._display_results(res)

    def _build_tx_dict(self, amt, cat, gender, cc, u_lat, u_long, m_lat, m_long):
        return {
            "cc_num": cc, "amt": amt, "category": cat, "gender": gender,
            "lat": u_lat, "long": u_long, "merch_lat": m_lat, "merch_long": m_long,
            "zip": "M5V", "city_pop": 2000000, "unix_time": 1710712800, "is_fraud": 0
        }

    def _display_results(self, res):
        st.divider()
        # Magenta accent for the verdict
        st.markdown(f"### Verdict: <span style='color:#d946ef'>{res['mode']}</span>", unsafe_allow_html=True)
        st.metric("Final MAS Score", f"{res['mas_score']:.4f}")
        st.info(f"**AI Reasoning:** {res['explanation']}")