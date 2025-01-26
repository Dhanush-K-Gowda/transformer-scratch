from pathlib import Path

def get_config():
    return {
        "batch_size":8,
        "num_epochs":3,
        "lr":10**-4,
        "seq_len":128,
        "d_model":128,
        "lang_src":"en",
        "lang_tgt":"kn",
        "model_folder":"weights",
        "model_filename":"tmodel_",
        "tokenizer_src": "ai4bharat/indic-bert",  # Use Indic tokenizer
        "tokenizer_tgt": "ai4bharat/indic-bert",
        "tokenizer_file":"tokenizer_{0}.json",
        "experiment_name":"runs/tmodel"

    }

def get_weights_file_path(config,epoch:str):
    model_folder = config['model_folder']
    model_basename=config['model_basename']
    model_filename=f"{model_basename}{epoch}.pt"
    return str(Path('.')/model_folder/model_filename)
