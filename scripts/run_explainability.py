import torch
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
from utils import setup_logger
from configs.config import Config
from data.datasets import get_stl10_datasets

def run_explainability():
    logger = setup_logger(Config.saved_models_path, log_file="explainability.log")
    logger.info("Loading model...")
    model = torch.load(f=f'{Config.saved_models_path}simclr_model.pth', weights_only=False)
    # model.eval()
    # Select the target class for which you want to explain predictions
    dataset = get_stl10_datasets(Config.data_path)
    input_tensor = torch.from_numpy(dataset[0].data)
    # target_class = torch.argmax(model(input_tensor)).item()
    logger.info("Got an example predicted with class 0")
    ig = IntegratedGradients(model)
    attributions, delta = ig.attribute(
        input_tensor, target=0, return_convergence_delta=True
    )
    attributions_np = attributions[0].cpu().detach().numpy()

    attributions_np = (attributions_np - attributions_np.min()) / (attributions_np.max() - attributions_np.min())

    logger.info("Preparing image with attributions")
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(input_tensor[0].cpu().permute(1, 2, 0))
    ax[0].set_title('Original Image')
    ax[1].imshow(attributions_np.transpose(1, 2, 0))
    ax[1].set_title('Attributions')
    plt.show()

# if __name__ == "main":
#     logger = setup_logger(Config.saved_models_path, log_file="explainability.log")
#     logger.info("Starting explainability pipeline...")

#     try:
#         run_explainability()
#         logger.info("Explainability pipeline completed successfully.")
#     except Exception as e:
#         logger.error(f"An error occurred during explainablity pipeline: {e}")
#         raise