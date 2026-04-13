import streamlit as st
import asyncio
import pandas as pd


class SingleAuditView:
    def __init__(self, controller):
        self.controller = controller

    def render(self, ui_params):
        st.header("🔎 Targeted Compliance Audit")

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

        # 2. Search Interface
        search_cc = st.text_input("Enter Full Credit Card Number", placeholder="e.g. 377895991033232")

        if not search_cc:
            st.info("Enter a CC number above to begin.")
            return

        # RESET LOGIC: Clear results if the search term changes
        if "last_search" not in st.session_state or st.session_state.last_search != search_cc:
            st.session_state.current_audit_result = None
            st.session_state.last_search = search_cc

        # 3. Retrieval Logic (Unified)
        target_data = None
        # Check cached batch results first
        if st.session_state.get('last_batch_results') is not None:
            df = st.session_state['last_batch_results']
            match = df[df['CC'].astype(str).str.endswith(str(search_cc)[-4:])]
            if not match.empty:
                target_data = match.iloc[0].to_dict()

        # Fallback to Database lookup
        if not target_data:
            with st.spinner("Searching database..."):
                target_data = self.controller.get_transaction_by_cc(search_cc)

        # 4. Main View Logic
        if not target_data:
            st.error(f"CC '{search_cc}' not found in the ledger or database.")
            return

        # Success Message (Only once)
        st.success(f"✅ Record Found: {target_data.get('merchant', 'Unknown Merchant')}")

        # --- 5. EXECUTION ---
        if st.button("🛡️ Run Agentic Audit"):
            st.session_state.trace_buffer = []
            try:
                audit_res = asyncio.run(self.controller.run_single_inference(
                    target_data, ui_params, callback=update_live_console
                ))
                st.session_state.current_audit_result = audit_res
            except Exception as e:
                st.error(f"Audit Execution Failed: {str(e)}")

        # --- 6. CONDITIONAL RESULTS DISPLAY ---
        if st.session_state.get('current_audit_result'):
            result = st.session_state.current_audit_result
            
            merchant_name = target_data.get('merchant', 'Unknown Merchant')
            customer_name = f"{target_data.get('first', 'Unknown')} {target_data.get('last', 'User')}"

            st.divider()
            st.subheader(f"🕵️ Audit Results for: {merchant_name}")

            # Top-Level Metrics
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("Final Audit Score", f"{result['MATH']:.3f}")
            with m_col2:
                actual = target_data.get('actual_label', result.get('ACT', 0))
                st.metric("Ground Truth", "FRAUD" if actual == 1 else "LEGIT")
            with m_col3:
                status = "🚩 HIGH RISK" if result['MATH'] > 0.3 else "✅ CLEAR"
                st.markdown(f"### {status}")

            # Pillar Breakdown
            st.subheader("📊 Mathematical Pillar Values")
            p_col1, p_col2, p_col3 = st.columns(3)
            p_col1.metric("Supervised (GOLD)", f"{result.get('GOLD', 0.0):.3f}")
            p_col2.metric("Neural (N_RAW)", f"{result.get('N_RAW', 0.0):.3f}")
            p_col3.metric("Baseline (BASE)", f"{result.get('BASE', 0.0):.3f}")

            # Narrative & Trace
            st.divider()
            res_col1, res_col2 = st.columns([2, 1])
            with res_col1:
                st.subheader("🤖 Auditor's Narrative")
                st.info(result['explanation'])
            with res_col2:
                st.subheader("🕵️ Agentic Trace")
                with st.expander("View Thought Process"):
                    for log in result.get('trace', []):
                        st.caption(f"**{log['agent']}**")
                        st.write(log['content'])

            # --- 7. DRILL DEEPER SECTION ---
            st.divider()
            st.subheader("👤 Customer Management")
            if st.button(f"🔍 Drill Deeper: View {result.get('first', 'User')}'s Profile", key="drill_deeper_btn"):
                st.session_state.selected_customer = {**target_data, **result}
                st.session_state.current_page = "CustomerProfile"
                st.rerun()

# class SingleAuditView:
#     def __init__(self, controller):
#         self.controller = controller
#
#     def render(self, ui_params):
#         st.header("🔎 Targeted Compliance Audit")
#
#         # --- 1. THE LIVE CONSOLE (Shared Logic with Batch View) ---
#         st.subheader("🕵️ Live Agentic Reasoning Stream")
#         console_placeholder = st.empty()
#
#         def update_live_console(agent, message):
#             if 'trace_buffer' not in st.session_state:
#                 st.session_state.trace_buffer = []
#
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
#         # 2. Simple Search Interface
#         search_cc = st.text_input(
#             "Enter Full Credit Card Number for Investigation",
#             placeholder="e.g. 377895991033232"
#         )
#
#         if not search_cc:
#             st.info("Enter a CC number above to begin the audit.")
#             return
#
#         # 3. Retrieval Logic
#         target_data = None
#         if 'last_batch_results' in st.session_state and st.session_state['last_batch_results'] is not None:
#             df = st.session_state['last_batch_results']
#             # Search logic for CC match
#             match = df[df['CC'].astype(str).str.endswith(str(search_cc)[-4:])]
#             if not match.empty:
#                 target_data = match.iloc[0].to_dict()
#
#         if not target_data:
#             with st.spinner("Searching database for record..."):
#                 target_data = self.controller.get_transaction_by_cc(search_cc)
#
#         # 4. Execution Display
#         if target_data:
#             st.success(f"✅ Record Found: {target_data.get('merchant', 'Unknown Merchant')}")
#
#             # Inside SingleAuditView.render
#             if st.button(f"🛡️ Run Agentic Audit"):
#                 st.session_state.trace_buffer = []
#
#                 try:
#                     # 1. The Controller handles the DataFrame -> Dict conversion
#                     result = asyncio.run(self.controller.run_single_inference(
#                         target_data,
#                         ui_params,
#                         callback=update_live_console
#                     ))
#
#                     # 2. Results Display (NO .iloc here!)
#                     st.divider()
#                     res_col1, res_col2 = st.columns([1, 2])
#
#                     with res_col1:
#                         # result is a DICT, so we use ['KEY']
#                         st.metric("Final Audit Score", f"{result['MATH']:.3f}")
#                         status = "🚩 HIGH RISK" if result['MATH'] > 0.3 else "✅ CLEAR"
#                         st.markdown(f"### {status}")
#
#                     with res_col2:
#                         st.subheader("🤖 Auditor's Narrative")
#                         # This will now show the real text once Step 1 is applied
#                         st.info(result['explanation'])
#
#                 except Exception as e:
#                     st.error(f"Audit Execution Failed: {str(e)}")
#         else:
#             st.error(f"CC '{search_cc}' not found in the ledger or database.")

