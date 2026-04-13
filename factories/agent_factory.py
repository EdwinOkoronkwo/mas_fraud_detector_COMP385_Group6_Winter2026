# mas_fraud_detector/factories/agent_factory.py
# mas_fraud_detector/factories/agent_factory.py
from agentic_inference.agents.rag_team import RagAuditTeam
from agentic_inference.agents.sql_researcher import SQLResearcher
from agentic_inference.agents.synthesis_engine import SynthesisEngine
from agentic_inference.agents.vector_researcher import VectorResearcher
from core.decision_agent import DecisionAggregator
from strategies.anomaly.anomaly_team import create_anomaly_team
from strategies.anomaly.clustering.clustering_team import create_clustering_team
from strategies.anomaly.critic_agent import AnomalyCritic
from strategies.anomaly.clustering.dbscan_agent import DBSCANAgent
from strategies.anomaly.clustering.iso_forest import IsolationForestAgent
from strategies.anomaly.clustering.kmeans_agent import KMeansAgent
from strategies.anomaly.clustering.som_agent import SOMAgent
from strategies.anomaly.neuro_pattern.ae_agent import AEAgent
from strategies.anomaly.neuro_pattern.ae_rnn_agent import RNNAgent
from strategies.anomaly.neuro_pattern.neuro_critic import NeuroCritic
from strategies.anomaly.neuro_pattern.neuro_pattern_team import create_neuro_team
from strategies.anomaly.neuro_pattern.vae_agent import VAEAgent
from strategies.data_preprocessing.feature_engineer import FeatureEngineerAgent
from strategies.data_preprocessing.sql_ingestor_agent import SQLIngestorAgent
from strategies.data_preprocessing.eda_agent import EDAAgent
from strategies.data_preprocessing.preprocess_agent import PreprocessAgent
from strategies.data_preprocessing.preprocessing_team import create_preprocessing_team
from strategies.supervised.ann_agent import ANNAgent
from strategies.supervised.dynamic_xgb_agent import DynamicXGBAgent
from strategies.supervised.lr_agent import LRAgent
from strategies.supervised.dynamic_rf_agent import DynamicRFAgent
from strategies.supervised.sampling_agent import SamplingAgent
from strategies.supervised.static_xgb_agent import StaticXGBAgent
from strategies.supervised.supervised_critic import SupervisedCritic
from strategies.supervised.supervised_team import create_supervised_team
from strategies.supervised.xgb_agent import XGBAgent


import os

class AgentFactory:
    def __init__(self, model_client, settings):
        self.model_client = model_client
        self.settings = settings

        # Mapping key paths for easy access across the factory using the settings object
        self.temp_data_path = getattr(self.settings, "TEMP_SPLIT_PATH", None)
        self.db_path = getattr(self.settings, "DB_PATH", None)
        self.report_dir = getattr(self.settings, "REPORT_DIR", "reports")

        if self.temp_data_path is None:
            print("⚠️ WARNING: AgentFactory could not find TEMP_SPLIT_PATH in settings!")

    def get_sql_ingestor(self):
        # Create the specific config dictionary the Ingestor needs
        config = {
            "KAGGLE_PATH": getattr(self.settings, "KAGGLE_PATH", "data/kaggle"),
            "DB_PATH": self.db_path
        }
        return SQLIngestorAgent(config, self.model_client)

    def get_eda_agent(self):
        return EDAAgent(self.settings, self.model_client)

    def get_preprocess_agent(self):
        # PreprocessAgent EXPECTS a Dict, so we build it from settings
        config_dict = {
            "DB_PATH": self.db_path,
            "REPORT_DIR": self.report_dir,
            "MODELS_DIR": getattr(self.settings, "MODELS_DIR", "models")
        }
        return PreprocessAgent(config_dict, self.model_client)

    def get_feature_engineer(self):
        # 1. Prepare the path string
        models_dir = getattr(self.settings, "MODELS_DIR", "models")
        path_str = str(os.path.join(models_dir, "preprocessor.joblib"))

        # 2. Instantiate using the code YOU provided
        # It matches: (settings, model_client, path_string)
        return FeatureEngineerAgent(
            settings=self.settings,
            model_client=self.model_client,
            preprocessor_path=path_str
        )

    def get_preprocessing_team(self):
        sql_ingestor = self.get_sql_ingestor()
        eda_specialist = self.get_eda_agent()
        feature_engineer = self.get_feature_engineer()
        preprocess_agent = self.get_preprocess_agent()

        return create_preprocessing_team(
            sql_ingestor,
            eda_specialist,
            feature_engineer,
            preprocess_agent,
            self.model_client
        )

    def get_sampling_agent(self):
        database_path = self.settings.DB_PATH
        return SamplingAgent(
            model_client=self.model_client,
            db_path=database_path
        ).agent

    def get_lr_agent(self):
        return LRAgent(self.model_client, self.temp_data_path).agent

    def get_rf_agent(self):
        return DynamicRFAgent(self.model_client, self.temp_data_path).agent

    def get_xgb_agent(self):
        return XGBAgent(self.model_client, self.temp_data_path).agent

    def get_ann_agent(self):
        return ANNAgent(self.model_client, self.temp_data_path).agent

    def get_static_xgb_agent(self):
        return StaticXGBAgent(self.model_client, self.temp_data_path).agent

    def get_dynamic_xgb_agent(self):
        return DynamicXGBAgent(self.model_client, self.temp_data_path).agent

    def get_supervised_critic_agent(self):
        return SupervisedCritic(
            model_client=self.model_client,
            db_path=self.settings.DB_PATH
        ).agent

    def get_supervised_championship_team(self):
        # 1. Fetch all participants
        sample = self.get_sampling_agent()
        s_xgb = self.get_static_xgb_agent()
        d_xgb = self.get_dynamic_xgb_agent()
        rf_agent = self.get_rf_agent()  # 🟢 DON'T FORGET THIS
        ann_agent = self.get_ann_agent()
        critic = self.get_supervised_critic_agent()

        # 2. Pass them ALL to the factory
        return create_supervised_team(
            sample, s_xgb, d_xgb, rf_agent, ann_agent, critic,  # 🟢 Ensure 6 participants
            self.model_client,
            self.settings.DB_PATH
        )
    def get_kmeans_agent(self):
        return KMeansAgent(self.model_client, self.settings).agent

    def get_dbscan_agent(self):
        return DBSCANAgent(self.model_client, self.settings).agent

    def get_som_agent(self):
        return SOMAgent(self.model_client, self.temp_data_path).agent

    def get_iso_forest_agent(self):
        return IsolationForestAgent(self.model_client, self.temp_data_path).agent

    def get_anomaly_critic_agent(self):
        return AnomalyCritic(self.model_client, self.settings).agent

    def get_clustering_challenge_team(self):
        sql_db_path = self.settings.DB_PATH
        self.temp_data_path = sql_db_path

        kmeans = self.get_kmeans_agent()
        dbscan = self.get_dbscan_agent()
        som = self.get_som_agent()
        iso_forest = self.get_iso_forest_agent()
        critic = self.get_anomaly_critic_agent()

        return create_clustering_team(
            agents_list=[kmeans, dbscan, som, iso_forest, critic],
            model_client=self.model_client
        )

    def get_ae_agent(self):
        return AEAgent(
            model_client=self.model_client,
            db_path=self.settings.DB_PATH
        ).agent

    def get_vae_agent(self):
        return VAEAgent(self.model_client, self.settings).agent

    def get_rnn_agent(self):
        return RNNAgent(self.model_client, self.settings).agent

    def get_neuro_critic_agent(self):
        return NeuroCritic(self.model_client, self.settings.DB_PATH).agent

    def get_neuro_pattern_challenge_team(self):
        sql_db_path = self.settings.DB_PATH
        self.temp_data_path = sql_db_path

        ae_agent = self.get_ae_agent()
        vae_agent = self.get_vae_agent()
        rnn_agent = self.get_rnn_agent()
        critic = self.get_neuro_critic_agent()

        return create_neuro_team(
            agents_list=[ae_agent, vae_agent, rnn_agent, critic],
            model_client=self.model_client
        )

    def get_decision_aggregator_agent(self, save_method):
        return DecisionAggregator(
            model_client=self.model_client,
            settings=self.settings,
            save_tool=save_method
        ).agent

    def get_decision_aggregation_team(self, save_method):
        aggregator = self.get_decision_aggregator_agent(save_method)
        return aggregator

    def get_anomaly_discovery_team(self):
        rnn = RNNAgent(self.model_client, self.settings)
        vae = VAEAgent(self.model_client, self.settings)
        dbscan = DBSCANAgent(self.model_client, self.settings)
        kmeans = KMeansAgent(self.model_client, self.settings)
        critic = AnomalyCritic(self.model_client, self.settings)

        return create_anomaly_team(
            [rnn.agent, vae.agent, dbscan.agent, kmeans.agent, critic.agent],
            self.model_client
        )

    def get_sql_researcher(self):
        """Manufactures the SQL Specialist agent."""
        return SQLResearcher(
            model_client=self.model_client,
            db_path=self.settings.DB_PATH
        ).agent

    def get_vector_researcher(self, vector_service):
        """Manufactures the Policy/Handbook Specialist agent."""
        return VectorResearcher(
            model_client=self.model_client,
            vector_service=vector_service
        ).agent

    def get_synthesis_engine(self):
        """Manufactures the Final Auditor/Synthesizer agent."""
        return SynthesisEngine(self.model_client).agent

    # --- The RAG Team Assembly ---

    def get_rag_audit_team(self, vector_service):
        sql_agent = self.get_sql_researcher()
        vector_agent = self.get_vector_researcher(vector_service)
        synth_agent = self.get_synthesis_engine()  # <--- Use the Narrator version we modified

        return RagAuditTeam(
            model_client=self.model_client,
            sql_agent=sql_agent,
            vector_agent=vector_agent,
            synth_agent=synth_agent
        )
#
