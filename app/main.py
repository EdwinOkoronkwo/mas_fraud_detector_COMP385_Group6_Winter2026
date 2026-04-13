



import sys
import os



# Adds the project root (mas_fraud_detector) to the Python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import asyncio
from api.controllers.inference_controller import InferenceController
from config.llm_config import get_model_client
from config.settings import settings
from run_phase3 import InferencePipeline
from data.database import engine, Base
from data.models.inference_result import InferenceResult
from components.sidebar import SidebarComponent
from views.batch_view import BatchView
from views.simulator import SimulatorView
from views.single_audit_view import SingleAuditView
from views.customer_view import  CustomerProfileView

from agentic_inference.services.vector_service import VectorService

from factories.agent_factory import AgentFactory

Base.metadata.create_all(bind=engine)

# --- 1. PERSISTENT PIPELINE INITIALIZATION ---
if 'pipeline' not in st.session_state:
    # A. Instantiate the lower-level dependencies
    # These are the "Engine" and "Library" for your agents
    my_v_service = VectorService()

    # Assuming AgentFactory needs a model client as seen in your main()
    # Replace get_model_client() with your actual initialization function
    m_client = get_model_client()
    my_factory = AgentFactory(model_client=m_client, settings=settings)

    # B. Instantiate the Pipeline (The Orchestrator)
    st.session_state.pipeline = InferencePipeline(
        factory=my_factory,
        vector_service=my_v_service
    )

    # C. Instantiate the Controller (The Bridge to SQL/UI)
    st.session_state.controller = InferenceController(st.session_state.pipeline)


def apply_custom_style():
    st.markdown("""
        <style>
            /* Magenta Glow for the sidebar */
            [data-testid="stSidebar"] {
                border-right: 2px solid #d946ef33;
            }

            /* Custom styling for Metric Cards */
            [data-testid="stMetricValue"] {
                color: #d946ef !important;
            }

            /* Tabs styling - Deep Ocean Active State */
            .stTabs [data-baseweb="tab-list"] {
                gap: 24px;
            }
            .stTabs [data-baseweb="tab"] {
                height: 50px;
                white-space: pre-wrap;
                background-color: transparent;
                border-radius: 4px 4px 0px 0px;
                color: #94a3b8;
            }
            .stTabs [aria-selected="true"] {
                color: #d946ef !important;
                border-bottom: 2px solid #d946ef !important;
            }
        </style>
    """, unsafe_allow_html=True)


def main():
    # 1. Page Configuration
    st.set_page_config(
        page_title="MAS Fraud Detector",
        page_icon="🛡️",
        layout="wide"
    )

    # 2. Initialize Session State for Routing
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Main"

    # 3. Sidebar
    sidebar = SidebarComponent()
    ui_params = sidebar.render()

    # --- 4. ROUTER LOGIC ---

    # CASE A: Drill-Down Customer Profile Page
    if st.session_state.current_page == "CustomerProfile":
        st.title("👤 KYC: Customer Identity Deep-Dive")
        profile_view = CustomerProfileView(st.session_state.controller)
        # Pass the selected customer data saved during the Audit click
        profile_view.render(st.session_state.get('selected_customer', {}))

    # CASE B: Standard Dashboard (The Tabs)
    else:
        st.title("🛡️ MAS Fraud Detector: Live Calibration")

        tab1, tab2 = st.tabs(["📊 Batch Fraud Ledger", "🔎 Single Transaction Audit"])

        with tab1:
            batch_view = BatchView(st.session_state.controller)
            batch_view.render(ui_params)

        with tab2:
            audit_view = SingleAuditView(st.session_state.controller)
            audit_view.render(ui_params)


if __name__ == "__main__":
    main()