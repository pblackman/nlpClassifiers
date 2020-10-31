from os.path import dirname, join

ROOT = dirname(dirname(__file__))

PATH_TO_VIRTUAL_OPERATOR_DATA = join(ROOT, "data/virtual-operator")
PATH_TO_AGENT_BENCHMARK_DATA = join(ROOT, "data/agent-benchmark")
PATH_TO_ML_PT_DATA = join(ROOT, "data/mercado-livre-pt-only")

PATH_TO_VIRTUAL_OPERATOR_MODELS = join(ROOT, "models/virtual-operator")
PATH_TO_AGENT_BENCHMARK_MODELS = join(ROOT, "models/agent-benchmark")
PATH_TO_ML_PT_MODELS = join(ROOT, "models/mercado-livre-pt-only")

# BERT variables
PATH_TO_BERT_MODELS_FOLDER = join(ROOT, "models", "bert")
PATH_TO_BASE_BERT = join(ROOT, "models", "bert", "base")
PATH_TO_LARGE_BERT = join(ROOT, "models", "bert", "large")
PATH_TO_BERT = {"large": PATH_TO_LARGE_BERT, "base": PATH_TO_BASE_BERT}
PATH_TO_BERT_FINE_TUNING_DATA = join(ROOT, "models", "bert_fine_tuning_data")