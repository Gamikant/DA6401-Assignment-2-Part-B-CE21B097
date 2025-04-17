# Fine-Tuning Pre-Trained Models for iNaturalist Classification

This project explores the application of transfer learning techniques to classify images from the iNaturalist dataset, which contains various biological specimens across 10 classes. By leveraging pre-trained models from the ImageNet dataset, we demonstrate how fine-tuning can significantly improve performance compared to training models from scratch.

## Project Overview

The iNaturalist dataset presents a challenging image classification task with natural images of biological specimens. This project implements and evaluates different strategies for adapting pre-trained ImageNet models to this specific domain, addressing key challenges in transfer learning:

1. Adapting input dimensions to match pre-trained model requirements
2. Modifying output layers to accommodate the 10 iNaturalist classes
3. Implementing efficient fine-tuning strategies to balance performance and computational cost

## Implementation Details
```
├── Q1.md                  # Addressing input/output adaptations for pre-trained models
├── Q2.md                  # Discussion of fine-tuning strategies
├── Q3.py                  # Implementation of fine-tuning with ResNet50
├── dataset_split.py       # Stratified dataset splitting with subset options
└── README.md              # Detailed comparison of fine-tuning vs. training 
└── inaturalist_12k/       # Dataset directory
                    ├── train/       # Training data into 10 class folders
                    └── val/         # Validation data into 10 class folders
```

### Pre-Trained Model Selection

We selected ResNet50 pre-trained on ImageNet as our base model due to its strong performance on image classification tasks and its well-established architecture that balances depth and computational efficiency.

### Adaptation Strategy

To adapt the pre-trained model to our task, we:

1. Resized all iNaturalist images to 224×224 pixels to match ResNet50's expected input dimensions
2. Replaced the final fully connected layer (originally for 1000 ImageNet classes) with a new layer for our 10 iNaturalist classes
3. Implemented a partial freezing strategy, where the first 6 children modules (early layers) were frozen while later layers were fine-tuned

### Fine-Tuning Approach

We implemented the "Freezing Base Layers" strategy, which:

- Preserves the general feature extraction capabilities in early layers
- Allows adaptation of higher-level features to the specific characteristics of biological specimens
- Uses different learning rates for different parts of the network (0.001 for the new classification layer, 0.0001 for the unfrozen pre-trained layers)
- How to run
```
python Q3.py
```


## Experimental Results

### Fine-Tuning Performance

The fine-tuned ResNet50 model achieved impressive results with minimal training:


| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
| :-- | :-- | :-- | :-- | :-- |
| 1 | 1.6650 | 0.5078 | 0.8515 | 0.7340 |
| 2 | 0.5509 | 0.8389 | 0.6150 | 0.7900 |
| 3 | 0.1967 | 0.9530 | 0.5899 | 0.8100 |
| 4 | 0.0748 | 0.9880 | 0.6260 | 0.8020 |
| 5 | 0.0280 | 0.9970 | 0.6278 | 0.8180 |

### Comparison with Training from Scratch

When comparing with our custom CNN model trained from scratch in Part A, the differences are striking:


| Approach | Initial Val Acc | Final Val Acc | Training Time | Convergence Speed |
| :-- | :-- | :-- | :-- | :-- |
| From Scratch | ~16% | ~26% | High | Slow (10+ epochs) |
| Fine-Tuning | 73.4% | 81.8% | Low | Fast (5 epochs) |

## Key Insights: Fine-Tuning vs. Training from Scratch

### 1. Convergence Speed and Efficiency

The fine-tuned model demonstrated remarkably faster convergence, achieving 73.4% validation accuracy in the very first epoch. In contrast, the model trained from scratch required 10+ epochs to reach just 26% accuracy. This dramatic difference highlights how pre-trained models leverage existing knowledge of visual features, providing a much stronger starting point than random initialization.

The computational efficiency was also substantially better with fine-tuning. Despite ResNet50 being a much larger model than our custom CNN, the training time per epoch was significantly reduced due to freezing early layers, which limits the number of parameters requiring gradient computation.

### 2. Performance with Limited Data

One of the most striking advantages of fine-tuning was its ability to perform well with limited training data. Using only 25% of the iNaturalist dataset, the fine-tuned model achieved over 80% validation accuracy, while the model trained from scratch struggled to reach even 30% accuracy with the full dataset.

This demonstrates how transfer learning effectively leverages knowledge gained from large datasets (ImageNet) to perform well even when task-specific data is scarce. The pre-trained model already understands fundamental visual concepts, requiring less data to adapt to a new domain.

### 3. Feature Representation Quality

The fine-tuned model produced more semantically meaningful representations of iNaturalist images from the very first epoch. This was evident in the model's ability to distinguish between visually similar biological classes much earlier in training.

The model trained from scratch needed many more epochs to develop useful feature representations, and even then struggled with certain fine-grained distinctions. This highlights how pre-trained models benefit from transferable feature hierarchies learned from diverse image datasets.

### 4. Generalization Capabilities

While the fine-tuned model showed some signs of overfitting (gap between 99.7% training accuracy and 81.8% validation accuracy in the final epoch), its generalization capabilities were still far superior to the model trained from scratch. This improved generalization can be attributed to the robust, general-purpose features learned by the pre-trained model on ImageNet, which provide a form of implicit regularization.

## Conclusion

Our experiments clearly demonstrate the substantial advantages of fine-tuning pre-trained models over training from scratch for the iNaturalist classification task. The fine-tuned ResNet50 model achieved significantly higher accuracy (81.8% vs. 26%) with much less training time and computational resources.

These findings align with the broader understanding in the field that fine-tuning leverages rich feature representations learned from large datasets, making it particularly valuable when working with specialized domains where labeled data may be limited. For the iNaturalist dataset specifically, the fine-tuning approach proved to be the superior choice, demonstrating how transfer learning can effectively bridge the gap between general visual understanding and specialized biological classification tasks.
