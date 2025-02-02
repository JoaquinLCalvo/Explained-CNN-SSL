from utils.logger import setup_logger
from training.train_simclr import train_simclr
from training.train_logistic_regression import train_logreg
from training.train_resnet import train_resnet18
from scripts.run_explainability import run_explainability
from data.datasets import get_stl10_datasets, get_stl10_datasets_clr, prepare_data_features, get_transformations
from torchvision import transforms
from models.resnet_for_xai import ResNetForXai
from models.logistic_regression_for_xai import LogisticRegressionForXai
from utils.explainability import get_examples, get_attributions, show_attributions
import os

def run_pipeline():
    logger = setup_logger(log_dir='logs/', log_file='pipeline.log')

    EXAMPLES_FOLDER = "examples"
    EXAMPLE_NUMBER = "ig_gaussian_baseline_contrast"
    examples_image_folder_path = os.path.join(EXAMPLES_FOLDER, EXAMPLE_NUMBER)
    os.makedirs(examples_image_folder_path, exist_ok=True)
    EXAMPLE_IMAGE_NAME = "gaussian_baseline_contrast_attributions.png"
    examples_image_path = os.path.join(examples_image_folder_path, EXAMPLE_IMAGE_NAME)
    EXAMPLES_COUNT = 5

    num_epochs = 25
    batch_size = 256
    lr = 1e-3
    temperature = 0.07
    hidden_dim = 128
    weight_decay = 1e-4
    num_classes = 10
    data_path = "data/"
    num_workers = os.cpu_count()

    unlabeled_data, train_data_contrast = get_stl10_datasets_clr(data_path=data_path)

    simclr_model = train_simclr(batch_size=batch_size, unlabeled_data=unlabeled_data,
                                train_data_contrast=train_data_contrast, num_workers=num_workers,
                                temperature=temperature, hidden_dim=hidden_dim, weight_decay=weight_decay,
                                lr=lr)
    
    normalize = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_img_data, test_img_data = get_stl10_datasets(data_path=data_path, train_transforms=normalize, test_transforms=normalize)
    train_feats_data = prepare_data_features(simclr_model, train_img_data)
    test_feats_data = prepare_data_features(simclr_model, test_img_data)

    simclr_logreg = train_logreg(batch_size=64, train_feats_data=train_feats_data, test_feats_data=test_feats_data,
                                 lr=lr, weight_decay=weight_decay, feature_dim=train_feats_data.tensors[0].shape[1],
                                 num_classes=num_classes)
    
    contrast_transforms, mixup_transforms = get_transformations()
    train_img_data_resnet, test_img_data_resnet = get_stl10_datasets(data_path=data_path,
                                                                     train_transforms=contrast_transforms,
                                                                     test_transforms=normalize)
    resnet_model = train_resnet18(train_dataset=train_img_data_resnet, val_dataset=test_img_data_resnet,
                                  batch_size=batch_size, lr=lr, weight_decay=weight_decay, num_workers=num_workers)
    
    simclr_logreg_for_xai = LogisticRegressionForXai(
        lr=lr,
        weight_decay=weight_decay,
        feature_dim=train_feats_data.tensors[0].shape[1],
        num_classes=num_classes,
        simclr_model=simclr_model
    )
    simclr_logreg_for_xai.load_state_dict(simclr_logreg.state_dict())
    simclr_logreg_for_xai.eval()

    resnet_model_for_xai = ResNetForXai(
        lr=lr,
        weight_decay=weight_decay,
        num_classes=10,
    )
    resnet_model_for_xai.load_state_dict(resnet_model.state_dict())
    resnet_model_for_xai.eval()

    models = {
        'simclr': simclr_logreg_for_xai,
        'resnet': resnet_model_for_xai
    }

    examples = get_examples(images=test_img_data, size=EXAMPLES_COUNT)
    models_attributions = {}
    for model_name in models:
        models_attributions[model_name] = get_attributions(model=models[model_name], examples=examples)
        for idx in range(len(examples)):
            examples[idx][model_name] = {}
            examples[idx][model_name]['prediction'] = models_attributions[model_name][idx]['prediction']
            examples[idx][model_name]['attr'] = models_attributions[model_name][idx]['attr']
    
    show_attributions(attributions=examples, model_names=models.keys(), save_fig=True, examples_image_path=examples_image_path)

    logger.info("pipeline completed successfully.")


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        print(f"An error occurred during the pipeline: {e}")
        raise