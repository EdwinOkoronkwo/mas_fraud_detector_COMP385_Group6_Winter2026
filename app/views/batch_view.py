import pandas as pd
import streamlit as st
import asyncio
import json
from app.components.metrics import MetricsComponent

import streamlit as st
import asyncio
import json
import pandas as pd

import streamlit as st
import asyncio
import json
import pandas as pd

import streamlit as st
import asyncio
import json
import pandas as pd

import streamlit as st
import asyncio
import json
import pandas as pd

import streamlit as st
import asyncio
import pandas as pd


class BatchView:
    def __init__(self, controller):
        self.controller = controller

    def render(self, ui_params: dict):
        st.header("📊 Batch Fraud Audit")
        st.caption("Process balanced samples from the 2026 transaction ledger.")

        # --- 1. THE LIVE CONSOLE ---
        st.subheader("🕵️ Live Agentic Reasoning Stream")
        console_placeholder = st.empty()

        def update_live_console(agent, message):
            if 'trace_buffer' not in st.session_state:
                st.session_state.trace_buffer = []
            log_line = f"**[{agent}]**: {message[:120]}..."
            st.session_state.trace_buffer.insert(0, log_line)
            st.session_state.trace_buffer = st.session_state.trace_buffer[:5]

            with console_placeholder.container():
                st.markdown(
                    f"""<div style="background-color: #0f172a; color: #38bdf8; padding: 15px; border-radius: 8px; border-left: 5px solid #d946ef; font-family: monospace;">
                    {"<br>".join(st.session_state.trace_buffer)}
                    </div>""", unsafe_allow_html=True
                )

        # --- 2. EXECUTION LOGIC ---
        if "last_batch_df" not in st.session_state:
            st.session_state.last_batch_df = None

        if st.button("🚀 Execute Neural Batch", key="batch_proc_btn"):
            st.session_state.trace_buffer = []

            async def run_batch():
                # 🚀 CLEANED: No 'delay' argument passed here.
                # The 1.5s sleep is now hardcoded inside BatchProcessor.execute
                return await self.controller.run_ui_batch(
                    n_samples=ui_params['batch_size'],
                    params=ui_params,
                    callback=update_live_console,
                )

            try:
                df = asyncio.run(run_batch())

                if df is not None and not df.empty:
                    # Fix Scientific Notation on CC
                    if 'CC' in df.columns:
                        df['CC'] = df['CC'].astype(str).str.replace(r'\.0$', '', regex=True)

                    # Ensure numeric consistency for the PerformanceEvaluator
                    for col in ['BASE', 'GOLD', 'N_RAW', 'MATH', 'ACT']:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

                    df['ACT'] = df['ACT'].astype(int)
                    st.session_state.last_batch_df = df
                    st.success("Batch Complete.")
            except Exception as e:
                # This will no longer trigger 'unexpected keyword argument delay'
                st.error(f"Execution Failed: {e}")

        # --- 3. RESULTS RENDERING ---
        if st.session_state.last_batch_df is not None:
            df = st.session_state.last_batch_df

            # Trust Weights Component
            MetricsComponent.render_trust_weights(st.session_state.pipeline.scorer.adapter.get_weights())

            st.divider()
            st.subheader("Detailed Audit Trail")

            # Updated Column Config to include Names
            st.dataframe(
                df,
                column_config={
                    "CC": st.column_config.TextColumn("Full Card", width="medium"),
                    "first": "First Name",
                    "last": "Last Name",
                    "merchant": "Merchant",
                    "ACT": st.column_config.NumberColumn("Actual", format="%d"),
                    "MATH": st.column_config.ProgressColumn("Risk", min_value=0, max_value=1, format="%.3f"),
                    "explanation": st.column_config.TextColumn("Narrative", width="large"),
                },
                use_container_width=True,
                hide_index=True
            )

            # Performance Plots
            MetricsComponent.render_performance_comparison(df)

            # --- 4. LOCAL RECORD INSPECTOR ---
            st.divider()
            st.subheader("🔍 Local Record Inspector")

            selected_cc = st.selectbox("Inspect specific record:", options=df['CC'].unique())

            if selected_cc:
                row = df[df['CC'] == selected_cc].iloc[0]
                st.session_state.search_cc_query = str(selected_cc)

                with st.container():
                    c1, c2, c3 = st.columns([1.5, 1.5, 1])
                    with c1:
                        # Pulling Nathan Massey correctly using your CSV headers
                        fname = row.get('first', 'Unknown')
                        lname = row.get('last', 'User')
                        st.markdown(f"**Customer:**\n### {fname} {lname}")
                    with c2:
                        st.markdown(f"**Merchant:**\n### {row.get('merchant', 'Unknown')}")
                    with c3:
                        st.metric("Risk Score", f"{row['MATH']:.4f}")

                    st.markdown("---")
                    col_text, col_stats = st.columns([2, 1])

                    with col_text:
                        st.markdown("**Auditor's Narrative**")
                        st.info(row['explanation'])

                    with col_stats:
                        st.markdown("**Pillar Breakdown**")
                        st.write(f"🎯 Supervised (GOLD): `{row['GOLD']:.3f}`")
                        st.write(f"🧠 Neural (N_RAW): `{row['N_RAW']:.4f}`")
                        st.write(f"📊 Baseline: `{row['BASE']:.3f}`")

                    with st.expander("View Full Multi-Agent Communication Trace"):
                        st.json(row.get('trace', "[]"))
        else:
            st.info("👆 Click 'Execute' to begin.")
# class BatchView:
#     def __init__(self, controller):
#         self.controller = controller
#
#     def render(self, ui_params: dict):
#         st.header("📊 Batch Fraud Audit")
#         st.caption("Process balanced samples from the 2026 transaction ledger.")
#
#         # --- 1. THE LIVE CONSOLE ---
#         st.subheader("🕵️ Live Agentic Reasoning Stream")
#         console_placeholder = st.empty()
#
#         # FIXED: Indented to be a local function inside render
#         def update_live_console(agent, message):
#             if 'trace_buffer' not in st.session_state:
#                 st.session_state.trace_buffer = []
#             log_line = f"**[{agent}]**: {message[:120]}..."
#             st.session_state.trace_buffer.insert(0, log_line)
#             st.session_state.trace_buffer = st.session_state.trace_buffer[:5]
#
#             with console_placeholder.container():
#                 st.markdown(
#                     f"""<div style="background-color: #0f172a; color: #38bdf8; padding: 15px; border-radius: 8px; border-left: 5px solid #d946ef; font-family: monospace;">
#                     {"<br>".join(st.session_state.trace_buffer)}
#                     </div>""", unsafe_allow_html=True
#                 )
#
#         # --- 2. EXECUTION LOGIC (Indented inside render) ---
#         if "last_batch_df" not in st.session_state:
#             st.session_state.last_batch_df = None
#
#         if st.button("🚀 Execute Neural Batch", key="batch_proc_btn"):
#             st.session_state.trace_buffer = []
#
#             try:
#                 df = asyncio.run(self.controller.run_ui_batch(
#                     n_samples=ui_params['batch_size'],
#                     params=ui_params,
#                     callback=update_live_console
#                 ))
#
#                 if 'CC' in df.columns:
#                     df['CC'] = df['CC'].astype(str)
#
#                 if 'trace' in df.columns:
#                     df['trace'] = df['trace'].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else str(x))
#
#                 st.session_state.last_batch_df = df
#                 st.success("Batch Complete.")
#             except Exception as e:
#                 st.error(f"Execution Failed: {e}")
#
#         # --- 3. RESULTS & SELECTION (Indented inside render) ---
#         if st.session_state.last_batch_df is not None:
#             df = st.session_state.last_batch_df
#
#             st.divider()
#             st.subheader("Detailed Audit Trail")
#
#             event = st.dataframe(
#                 df,
#                 column_config={
#                     "CC": st.column_config.TextColumn("Full Credit Card", width="medium"),
#                     "trace": st.column_config.TextColumn("Agent Trace", width="large"),
#                     "MATH": st.column_config.NumberColumn("Risk Score", format="%.4f")
#                 },
#                 use_container_width=True,
#                 hide_index=True,
#                 on_select="rerun",
#                 selection_mode="single-row" # Fixed hyphen from previous error
#             )
#
#             if event.selection.rows:
#                 selected_idx = event.selection.rows[0]
#                 selected_cc = str(df.iloc[selected_idx]['CC'])
#
#                 st.session_state.target_search_cc = selected_cc
#                 st.session_state.current_view = "Single Audit"
#                 st.rerun()
#
# class BatchView:
#     def __init__(self, controller):
#         self.controller = controller
#
#     def render(self, ui_params: dict):
#         st.header("📊 Batch Fraud Audit")
#         st.caption("Process balanced samples from the 2026 transaction ledger.")
#
#         # --- 1. THE LIVE CONSOLE ---
#         st.subheader("🕵️ Live Agentic Reasoning Stream")
#         console_placeholder = st.empty()
#
#         def update_live_console(agent, message):
#             if 'trace_buffer' not in st.session_state:
#                 st.session_state.trace_buffer = []
#             log_line = f"**[{agent}]**: {message[:120]}..."
#             st.session_state.trace_buffer.insert(0, log_line)
#             st.session_state.trace_buffer = st.session_state.trace_buffer[:5]
#
#             with console_placeholder.container():
#                 st.markdown(
#                     f"""<div style="background-color: #0f172a; color: #38bdf8; padding: 15px; border-radius: 8px; border-left: 5px solid #d946ef; font-family: monospace;">
#                     {"<br>".join(st.session_state.trace_buffer)}
#                     </div>""", unsafe_allow_html=True
#                 )
#
#         # --- 2. EXECUTION LOGIC ---
#         if "last_batch_df" not in st.session_state:
#             st.session_state.last_batch_df = None
#
#         if st.button("🚀 Execute Neural Batch", key="batch_proc_btn"):
#             st.session_state.trace_buffer = []
#
#             try:
#                 df = asyncio.run(self.controller.run_ui_batch(
#                     n_samples=ui_params['batch_size'],
#                     params=ui_params,
#                     callback=update_live_console
#                 ))
#
#                 # --- CRITICAL FIX FOR FULL CC NUMBER ---
#                 # 1. Ensure the column is explicitly string-type to prevent scientific notation
#                 # 2. If your controller/mock generator was masking (e.g. ****),
#                 #    we ensure here we are dealing with the raw string.
#                 if 'CC' in df.columns:
#                     df['CC'] = df['CC'].astype(str)
#
#                 # --- FIX FOR [object Object] ---
#                 # Recursively convert any dicts/lists in the 'trace' column to strings
#                 if 'trace' in df.columns:
#                     def clean_trace(x):
#                         if isinstance(x, (dict, list)):
#                             return json.dumps(x)
#                         return str(x)
#                     df['trace'] = df['trace'].apply(clean_trace)
#
#                 st.session_state.last_batch_df = df
#                 st.success("Batch Complete.")
#             except Exception as e:
#                 st.error(f"Execution Failed: {e}")
#
#         # --- 3. RESULTS RENDERING ---
#         if st.session_state.last_batch_df is not None:
#             df = st.session_state.last_batch_df
#
#             MetricsComponent.render_trust_weights(
#                 st.session_state.pipeline.scorer.adapter.get_weights()
#             )
#
#             st.divider()
#             st.subheader("Detailed Audit Trail")
#
#             # --- COLUMN CONFIGURATION FOR FULL VISIBILITY ---
#             # Using st.column_config.TextColumn ensures the UI doesn't format it as a number
#             st.dataframe(
#                 df,
#                 column_config={
#                     "CC": st.column_config.TextColumn(
#                         "Full Credit Card",
#                         help="Complete 16-digit card number",
#                         width="medium"
#                     ),
#                     "trace": st.column_config.TextColumn("Agent Trace", width="large"),
#                     "MATH": st.column_config.NumberColumn("Risk Score", format="%.4f")
#                 },
#                 use_container_width=True,
#                 hide_index=True
#             )
#
#             MetricsComponent.render_performance_comparison(df)
#
#             with st.expander("🔍 Examine Specific Agent Histories"):
#                 for idx, row in df.head(10).iterrows():
#                     # Display the full CC string here
#                     st.markdown(f"### Card: `{row['CC']}`")
#                     st.text(f"Risk: {row['MATH']:.4f}")
#                     st.info(f"**Explanation:** {row['explanation']}")
#                     if 'trace' in row:
#                         st.code(row['trace'], language="json")
#         else:
#             st.info("👆 Adjust calibration in the sidebar and click 'Execute' to begin.")