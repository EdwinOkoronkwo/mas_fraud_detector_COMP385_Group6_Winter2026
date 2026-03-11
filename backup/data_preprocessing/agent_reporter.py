import os
from datetime import datetime


class AgentReporter:
    def __init__(self, settings, logger):
        self.settings = settings
        self.logger = logger
        os.makedirs(self.settings.REPORT_DIR, exist_ok=True)

    def process_results(self, chat_result):
        """
        The main entry point: Prints to terminal AND saves to files.
        """
        self._stream_to_terminal_and_log(chat_result)
        self._save_critic_markdown(chat_result)

    def _stream_to_terminal_and_log(self, chat_result):
        """Prints the conversation and saves the raw transcript."""
        log_path = os.path.join(self.settings.REPORT_DIR, "phase1_agent_chat.log")

        print("\n" + "=" * 60)
        print("💬 AGENT COLLABORATION LOG (Terminal View)")
        print("=" * 60)

        with open(log_path, "w", encoding="utf-8") as f:
            for msg in chat_result.messages:
                sender = getattr(msg, 'source', 'System').upper()
                content = msg.content or ""

                # --- FIXED TOOL CALL EXTRACTION ---
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for call in msg.tool_calls:
                        # AutoGen FunctionCall objects have 'name' and 'arguments' directly
                        call_name = getattr(call, 'name', 'unknown_function')
                        call_args = getattr(call, 'arguments', '{}')
                        content += f"\n[TOOL CALL]: {call_name}({call_args})"

                # Construct the output string
                output = f"\n[{sender}]:\n{content}\n" + "-" * 30

                # 1. Show it to you in the terminal
                print(output)

                # 2. Save it to the log file
                f.write(output + "\n")

        self.logger.info(f"📜 Full Agent Chat Log saved to: {log_path}")

    def _save_critic_markdown(self, chat_result):
        """Extracts the Critic's final word and saves it as a Markdown report."""
        critic_output = chat_result.messages[-1].content
        report_path = os.path.join(self.settings.REPORT_DIR, "phase1_critic_report.md")

        header = f"# Agentic Quality Audit: Phase 1\n**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(critic_output)

        self.logger.info(f"📑 Critic's Markdown report saved to: {report_path}")