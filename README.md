#Fine-tuning of a CNN Model for Monument Image Classification

This project uses Deep Learning to identify monuments through transfer learning and fine-tuning.

##Workflow
1. Load a pretrained network.
2. Remove the last layer.
3. Freeze the backbone.
4. Add new layers specific to the problem.
5. Adjust hyperparameters.
6. Train the model with the dataset.
7. Unfreeze the model and fine-tune it.
8. Evaluate the model, modify hyperparameters if necessary and re-evaluate it.

##Technologies Used
- Python
- PyTorch
- Timm
- NumPy
- Scikit-learn
- Matplotlib
