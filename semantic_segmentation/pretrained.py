import keras

from models.all_models import model_from_name


def model_from_checkpoint_path(model_config, latest_weights):

    model = model_from_name[model_config['model_class']](
        model_config['n_classes'], input_height=model_config['input_height'],
        input_width=model_config['input_width'])
    model.load_weights(latest_weights)
    return model

