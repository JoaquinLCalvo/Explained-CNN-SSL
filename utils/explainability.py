from captum.attr import IntegratedGradients
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

# Getting some example images from the dataset
def get_examples(images, size):
    examples = []
    for i in range(size):
        image, label = images[i]
        examples.append({
            'img': image,
            'label': label
        })
    return examples

def get_attributions(model, examples: list):

    def forward_fn(inputs):
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)
        return model(inputs)

    method = IntegratedGradients(forward_fn)

    results = []
    for example in examples:
        input_tensor = example['img']
        input_tensor = input_tensor.unsqueeze(0)

        with torch.no_grad():
            logits = model(input_tensor)
            target_class = logits.argmax(dim=-1).item()
        
        gaussian_blur = transforms.GaussianBlur(kernel_size=13, sigma=2)
        baseline_image = gaussian_blur(input_tensor)

        attributions = method.attribute(
            inputs=input_tensor, target=target_class,
            baselines=baseline_image
        )
        attributions_np = attributions[0].cpu().detach().numpy()
        attributions_np = (attributions_np - attributions_np.min()) / (attributions_np.max() - attributions_np.min())

        if attributions_np.ndim == 3:
            attributions_np = attributions_np.transpose(1, 2, 0)
        results.append({
            'prediction': target_class,
            'attr': attributions_np
        })
    return results

def show_attributions(attributions, model_names, examples_image_path, save_fig=False):
    class_names = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]
    if not isinstance(attributions, list):
        attributions = [attributions]
    fig, ax = plt.subplots(len(attributions), len(model_names) + 1, figsize=(12, 5 * len(attributions)))
    for idx, ex in enumerate(attributions):
        img = ex['img']
        label = ex['label']

        img = img.detach()
        img = FT.to_pil_image(img)

        ax[idx, 0].imshow(np.asarray(img))
        ax[idx, 0].set_title(f'Original image class is {class_names[label]}')

        for jdx, model_name in enumerate(model_names):
            prediction = ex[model_name]['prediction']
            attr = ex[model_name]['attr']
            ax[idx, jdx + 1].imshow(attr)
            ax[idx, jdx + 1].set_title(f'Attributions of {model_name} (predicted: {class_names[prediction]})')
    plt.tight_layout()
    if save_fig:
        plt.savefig(examples_image_path)
    plt.show()