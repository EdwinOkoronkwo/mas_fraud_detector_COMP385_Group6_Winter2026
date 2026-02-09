# mas_fraud_detector/factories/agent_factory.py
# mas_fraud_detector/factories/agent_factory.py
from core.decision_agent import DecisionAggregator
from strategies.anomaly.anomaly_team import create_anomaly_team
from strategies.anomaly.clustering.clustering_team import create_clustering_team
from strategies.anomaly.clustering.critic_agent import AnomalyCritic
from strategies.anomaly.clustering.dbscan_agent import DBSCANAgent
from strategies.anomaly.clustering.iso_forest import IsolationForestAgent
from strategies.anomaly.clustering.kmeans_agent import KMeansAgent
from strategies.anomaly.clustering.som_agent import SOMAgent
from strategies.anomaly.neuro_pattern.ae_agent import AEAgent
from strategies.anomaly.neuro_pattern.ae_rnn_agent import RNNAgent
from strategies.anomaly.neuro_pattern.neuro_critic import NeuroCritic
from strategies.anomaly.neuro_pattern.neuro_pattern_team import create_neuro_team
from strategies.anomaly.neuro_pattern.vae_agent import VAEAgent
from strategies.data_preprocessing.sql_ingestor_agent import SQLIngestorAgent
from strategies.data_preprocessing.eda_agent import EDAAgent
from strategies.data_preprocessing.preprocess_agent import PreprocessAgent
from strategies.data_preprocessing.preprocessing_team import create_preprocessing_team
from strategies.supervised.ann_agent import ANNAgent
from strategies.supervised.lr_agent import LRAgent
from strategies.supervised.rf_agent import RFAgent
from strategies.supervised.sampling_agent import SamplingAgent
from strategies.supervised.supervised_team import create_supervised_team
from strategies.supervised.xgb_agent import XGBAgent


class AgentFactory:
    def __init__(self, model_client, config):
        self.model_client = model_client
        self.config = config
        # This path is passed to agents so they know where the temp data is
        self.temp_data_path = config.get("TEMP_DATA_PATH")

    def get_sql_ingestor(self):
        return SQLIngestorAgent(self.config, self.model_client)

    def get_eda_agent(self):
        return EDAAgent(self.config, self.model_client)

    def get_preprocess_agent(self):
        return PreprocessAgent(self.config, self.model_client)

    def get_preprocessing_team(self):
        """
        Orchestrates the creation of all required agents and assembles the team.
        """
        # 1. Instantiate all concrete agents
        sql_ingestor = self.get_sql_ingestor()
        eda_specialist = self.get_eda_agent()
        preprocess_agent = self.get_preprocess_agent()
        
        # 2. Assemble and return the team
        return create_preprocessing_team(
            sql_ingestor, 
            eda_specialist, 
            preprocess_agent, 
            self.model_client
        )


    # Inside AgentFactory class
    def get_sampling_agent(self):
        # Grab the path we defined in settings.py
        database_path = self.config.get("DB_PATH")

        # Pass it to the constructor
        return SamplingAgent(
            model_client=self.model_client,
            db_path=database_path
        ).agent

    def get_lr_agent(self):
        return LRAgent(self.model_client, self.temp_data_path).agent

    def get_rf_agent(self):
        return RFAgent(self.model_client, self.temp_data_path).agent

    def get_xgb_agent(self):
        return XGBAgent(self.model_client, self.temp_data_path).agent

    def get_ann_agent(self):
        return ANNAgent(self.model_client, self.temp_data_path).agent

    # Inside AgentFactory class
    def get_supervised_championship_team(self):
        # 1. Create the model agents
        sample = self.get_sampling_agent()
        lr = self.get_lr_agent()
        rf = self.get_rf_agent()
        xgb = self.get_xgb_agent()
        ann = self.get_ann_agent()

        # 2. Pass them as a list to the team creator
        print(f"DEBUG: Sampling Agent is {sample}")

        return create_supervised_team(
            sample, lr, rf, xgb, ann,
            self.model_client,
            self.config.get("DB_PATH")
        )

    # mas_fraud_detector/factories/agent_factory.py

    def get_kmeans_agent(self):
        return KMeansAgent(self.model_client, self.temp_data_path).agent

    def get_dbscan_agent(self):
        return DBSCANAgent(self.model_client, self.temp_data_path).agent

    def get_som_agent(self):
        return SOMAgent(self.model_client, self.temp_data_path).agent

    def get_iso_forest_agent(self):
        return IsolationForestAgent(self.model_client, self.temp_data_path).agent

    def get_anomaly_critic_agent(self):
        return AnomalyCritic(self.model_client, self.temp_data_path).agent

    def get_clustering_challenge_team(self):
        """
        Assembles the 4-model Anomaly Detection team plus the Anomaly Critic.
        """
        # 1. Update internal state to use SQLite
        sql_db_path = self.config.get("DB_PATH")
        self.temp_data_path = sql_db_path

        # 2. Instantiate the "Squad"
        kmeans = self.get_kmeans_agent()
        dbscan = self.get_dbscan_agent()
        som = self.get_som_agent()
        iso_forest = self.get_iso_forest_agent()

        # 3. Instantiate the Critic (Ensure this method exists in your factory)
        # The critic needs the db_path to run the cross-check tool
        critic = self.get_anomaly_critic_agent()

        # 4. Pass the full list to the team creator
        # The selector will now be able to route the final model output to the critic
        return create_clustering_team(
            agents_list=[kmeans, dbscan, som, iso_forest, critic],
            model_client=self.model_client
        )

    # Helper for AE
    def get_ae_agent(self):
        return AEAgent(self.model_client).agent

    # Helper for VAE
    def get_vae_agent(self):
        return VAEAgent(self.model_client).agent

    def get_rnn_agent(self):
        return RNNAgent(self.model_client).agent

    # Helper for Neuro Critic
    def get_neuro_critic_agent(self):
        # Ensure settings.DB_PATH is available here
        return NeuroCritic(self.model_client, self.config.get("DB_PATH")).agent

    def get_neuro_pattern_challenge_team(self):
        """
        Assembles the Neural Reconstruction Triple-Threat team:
        Standard AE, Variational AE, and RNN (LSTM) AE plus the Neuro Critic.
        """
        # 1. Update internal path to the SQLite database
        sql_db_path = self.config.get("DB_PATH")
        self.temp_data_path = sql_db_path

        # 2. Instantiate the Neural Agents (AE, VAE, and the new RNN)
        # Ensure these helper methods return the .agent property of the classes
        ae_agent = self.get_ae_agent()
        vae_agent = self.get_vae_agent()
        rnn_agent = self.get_rnn_agent()  # Added the RNN sequence expert

        # 3. Instantiate the specialized Neuro Critic
        # (Configured with verify_anomaly_labels and persist_champion_model)
        critic = self.get_neuro_critic_agent()

        # 4. Create the team using the updated neuro_selector logic
        # This now handles the sequence: AE -> VAE -> RNN -> Critic
        return create_neuro_team(
            agents_list=[ae_agent, vae_agent, rnn_agent, critic],
            model_client=self.model_client
        )

    # Helper for Decision Aggregator

    # agent_factory.py
    def get_decision_aggregator_agent(self, save_method):
        return DecisionAggregator(
            model_client=self.model_client,
            save_tool=save_method
        ).agent


    # Helper for the Aggregation Team
    def get_decision_aggregation_team(self):
        """
        Assembles the final Decision Aggregation team.
        For the final phase, this is typically the Aggregator and a Critic
        to ensure the Jurgovsky (2018) logic is applied correctly.
        """
        aggregator = self.get_decision_aggregator_agent()

        # We can reuse the Neuro Critic or a specialized 'Strategy Critic'
        # For now, let's keep it lean with a direct aggregator run or a simple team
        return aggregator

    def get_anomaly_discovery_team(self):
        # Instantiate the 4 discovery agents + 1 critic
        agents = [
            self.get_kmeans_agent(),
            self.get_dbscan_agent(),
            self.get_vae_agent(),
            self.get_rnn_agent(),
            self.get_anomaly_critic_agent()  # Ensure system_message asks for "Dual Champion"
        ]
        return create_anomaly_team(agents, self.model_client)