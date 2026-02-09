mas_fraud_detector/
├── config/
│   ├── settings.py            # Global paths & hyper-parameters
│   └── llm_config.py          # Mistral/OpenAI Client setup
├── core/
│   ├── planner.py             # THE BRAIN: Task decomposition & recruitment
│   ├── critic.py              # THE JUDGE: Validates outputs & forces retries
│   ├── orchestrator.py        # THE MANAGER: GroupChat/RoundRobin logic
│   └── state_registry.py      # THE MEMORY: Shared blackboard for results
├── factories/
│   └── agent_factory.py       # THE RECRUITER: Instantiates specialists
├── interfaces/
│   ├── i_agent.py             # Base Agent contract
│   ├── i_supervised.py        # Supervised tournament contract
│   └── i_anomaly.py           # Anomaly challenge contract
├── strategies/
│   ├── supervised/
│   │   ├── rf_agent.py        # Random Forest
│   │   ├── xgb_agent.py       # XGBoost
│   │   ├── ann_agent.py       # Neural Network
│   │   └── lr_agent.py        # Logistic Regression (Baseline)
│   └── anomaly/
│       ├── clustering/
│       │   ├── dbscan_agent.py
│       │   ├── som_agent.py
│       │   ├── kmeans_agent.py
│       │   └── iforest_agent.py
│       └── neuro_pattern/
│           ├── ae_standard_agent.py
│           ├── ae_rnn_agent.py
│           └── vae_agent.py   # Variational Autoencoder
├── tools/
│   ├── data_prep/
│   │   ├── cleaner.py
│   │   └── sequence_builder.py
│   ├── modeling/
│   │   ├── trainers.py        # Shared logic for XGB, RF, etc.
│   │   └── evaluators.py      # Standardized metric tool
│   └── reporting/
│       └── reporter.py        # Consensus report generator
├── utils/
│   ├── logger.py              # Centralized trace & metric logging
│   └── formatter.py           # Cleans LLM strings into JSON/DataFrames
├── data/                      # CSVs & Model checkpoints
├── logs/                      # System & Agent logs
└── main.py                    # Entry point