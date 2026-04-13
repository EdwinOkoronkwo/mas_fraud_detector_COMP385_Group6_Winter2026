import hashlib

import pandas as pd

import streamlit as st

from data.database import SessionLocal
from data.models.customer import Customer


import streamlit as st
import pandas as pd
from data.database import SessionLocal
from data.models.customer import Customer

import streamlit as st


class CustomerProfileView:
    def __init__(self, controller):
        self.controller = controller

    def render(self, customer_data):
        """
        Renders the deep-dive profile for a specific customer.
        customer_data: A dict or SQLAlchemy object containing customer details.
        """
        st.divider()

        # --- 1. Identity Header ---
        col1, col2 = st.columns([1, 3])

        # If customer_data is a dict from raw SQL/CSV
        name = customer_data.get('customer_name', f"{customer_data.get('first')} {customer_data.get('last')}")
        cc = customer_data.get('cc_num', customer_data.get('CC'))

        # Generate the photo URL dynamically
        seed = hashlib.md5(name.encode()).hexdigest()
        photo_url = f"https://i.pravatar.cc/150?u={seed}"

        with col1:
            # Use 'use_container_width' instead of 'use_column_width'
            st.image(photo_url, use_container_width=True, caption="Verified Identity")

        with col2:
            st.title(name)
            st.markdown(f"**Occupation:** {customer_data.get('job', 'Professional')}")
            st.markdown(f"**Card Number:** `**** **** **** {str(cc)[-4:]}`")

            # Status Badges
            b1, b2, b3 = st.columns(3)
            b1.metric("Historical Risk", f"{customer_data.get('risk_score', 0.15):.2f}")
            b2.metric("Avg. Transaction", f"${customer_data.get('avg_txn_amt', 0.0):.2f}")
            b3.error("Status: UNDER REVIEW") if customer_data.get('risk_score', 0) > 0.5 else b3.success(
                "Status: ACTIVE")

        # --- 2. Historical Analysis ---
        st.subheader("📊 Cross-Pillar Historical Trends")

        # Placeholder for historical data visualization
        chart_data = pd.DataFrame({
            "Pillar": ["Gold", "Neuro", "Cluster", "Baseline"],
            "Risk": [
                customer_data.get('GOLD', 0.1),
                customer_data.get('N_RAW', 0.1),
                0.2,  # Cluster placeholder
                customer_data.get('BASE', 0.1)
            ]
        })
        st.bar_chart(chart_data, x="Pillar", y="Risk")

        # --- 3. Return to Audit ---
        if st.button("⬅️ Return to Audit Page"):
            st.session_state.current_page = "SingleAudit"
            st.rerun()