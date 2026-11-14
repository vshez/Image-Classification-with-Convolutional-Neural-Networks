# Image-Classification-with-Convolutional-Neural-Networks
To design, implement, and evaluate a Long Short-Term Memory (LSTM) model (in combination with CNN layers) for image classification tasks.
# OrganAMNIST-CNN-LSTM

Anatomical structure classification using Python, TensorFlow, and Keras. This project compares a convolutional neural network (CNN) baseline with CNN–LSTM models on OrganAMNIST, including image preprocessing, hyperparameter tuning, learning curves, and held-out test evaluation.

The selected 128-unit CNN–LSTM achieved **89.26% test accuracy** and **0.8863 macro F1**, compared with **84.86% accuracy** for the CNN baseline.

## Dataset

Source: [OrganAMNIST in the MedMNIST collection](https://medmnist.com/). Download instructions and the Python package are available in the [official MedMNIST repository](https://github.com/MedMNIST/MedMNIST).

The exercise uses the **64 × 64 grayscale** version of OrganAMNIST, containing **58,830 axial CT image crops** across 11 anatomical categories:

- Bladder
- Left and right femur
- Heart
- Left and right kidney
- Liver
- Left and right lung
- Pancreas
- Spleen

The official dataset partitions are retained:

| Partition | Images | Purpose |
| --- | ---: | --- |
| Training | 34,561 | Learn model weights |
| Validation | 6,491 | Select checkpoints and hyperparameters |
| Test | 17,778 | Evaluate the selected models |

The script downloads the images through `medmnist.OrganAMNIST(..., size=64, download=True)` and caches them in `organamnist_64.npz`.

## Setup in PyCharm

1. Open the project and select its Python interpreter or virtual environment. The recorded exercise used Python 3.12.4.
2. Place `analysis.py` in the project folder.
3. Open PyCharm's Terminal and install the dependencies:

   ```bash
   python -m pip install tensorflow==2.21.0 medmnist==3.0.2 numpy pandas matplotlib seaborn scikit-learn
   ```

4. Check the dataset, model, and output paths configured in the script. Make sure these locations are writable.
5. Set the run configuration's working directory to the project folder. Internet access is needed for the first dataset download.

TensorFlow 2.21.0 and MedMNIST 3.0.2 match the recorded exercise. The other packages above are not an exact environment lock. Training can run on the CPU; the original native Windows environment reported no available GPU.

## How to Run

Right-click `analysis.py` and select **Run**, or execute:

```bash
python analysis.py
```

Read the console metrics and check the configured `outputs/` folder for CSV summaries and figures. Model checkpoints are stored in the configured `models/` folder.

The script can load completed models and their histories when those files already exist. A saved-model evaluation does not represent a new training run. To repeat training, use a separate copy of the project with a fresh model/output location and retain the original results for comparison.

## Method

1. Inspect image dimensions, pixel ranges, labels, and class counts while preserving the official partitions.
2. Convert pixel values to `float32`, divide by 255, and add a channel dimension to produce `64 × 64 × 1` inputs.
3. Apply small rotations of approximately ±5.4 degrees and translations of up to 3% during training. Do not flip images, because labels distinguish left and right anatomy. Validation and test images receive intensity scaling only.
4. Train a CNN baseline with two convolutional blocks: bias-free `3 × 3` convolution, batch normalization, ReLU, and `2 × 2` max pooling. The blocks use 32 and 64 filters. Flatten the resulting `16 × 16 × 64` feature map before a 64-neuron dense layer, dropout, and an 11-class softmax output.
5. Train CNN–LSTM variants using the same convolutional design. Reshape each feature map into **16 spatial steps with 1,024 features per step**, followed by an LSTM, output dropout, and softmax classification. Each sequence contains rows within one image.
6. Use Adam, sparse categorical cross-entropy, batches of 64, a maximum of 20 epochs, and early stopping with patience of five epochs. Retain the checkpoint with the lowest validation loss.
7. Compare four CNN–LSTM configurations using validation loss. After selection, evaluate the CNN baseline, initial CNN–LSTM, and selected CNN–LSTM on the test set using accuracy, precision, recall, F1, log loss, and confusion matrices.

Random seeds were set to 42. Each architecture learned its own convolutional weights. The LSTM models spatial relationships within a crop; neighboring CT slices are not combined into a sequence.

### Hyperparameter configurations

| Configuration | LSTM units | Output dropout | Learning rate | Epochs completed | Best epoch | Best validation loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Initial CNN–LSTM | 64 | 0.3 | 0.001 | 7 | 2 | 0.138684 |
| Lower learning rate | 64 | 0.3 | 0.0003 | 15 | 10 | 0.153720 |
| Larger LSTM — selected | 128 | 0.3 | 0.001 | 12 | 7 | 0.055469 |
| Higher dropout | 64 | 0.5 | 0.001 | 9 | 4 | 0.144188 |

The larger LSTM achieved **98.06% validation accuracy** at its minimum-loss checkpoint, epoch 7. The selected model is the `larger_lstm` experiment. The `lower_learning_rate` experiment is a separate configuration that was not selected.

### Observed test results

All three models were evaluated on the same 17,778 test images.

| Model | Model parameters | Correct predictions | Test accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 | Log loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CNN baseline | 1,068,459 | 15,087 | 84.86% | 0.8506 | 0.8123 | 0.8243 | 0.8461 | 0.5038 |
| Initial CNN–LSTM | 298,603 | 15,266 | 85.87% | 0.8582 | 0.8449 | 0.8498 | 0.8569 | 0.4187 |
| Selected CNN–LSTM | 610,859 | 15,868 | 89.26% | 0.8970 | 0.8826 | 0.8863 | 0.8897 | 0.3881 |

Model parameter counts include trainable weights and non-trainable batch-normalization statistics, excluding optimizer state.

Compared with the CNN baseline, the selected hybrid improved accuracy by **4.39 percentage points**, produced **781 fewer errors**, and used **42.83% fewer model parameters**. This corresponds to a **29.02% reduction in classification errors**.

### Learning curves

**CNN baseline**
Training and validation accuracy and loss for the CNN baseline.
<img width="2400" height="800" alt="cnn_baseline_training_curves" src="https://github.com/user-attachments/assets/d138603d-3c77-4d8f-9122-b54707d5698b" />

**Initial 64-unit CNN–LSTM**
Training and validation accuracy and loss for the initial CNN–LSTM.
<img width="2400" height="800" alt="cnn_lstm_initial_training_curves" src="https://github.com/user-attachments/assets/f620b798-8538-4cdb-bc68-000734ca51fc" />

**Selected 128-unit CNN–LSTM — larger LSTM experiment**
Training and validation accuracy and loss for the selected CNN–LSTM.
<img width="2400" height="800" alt="cnn_lstm_larger_lstm_curves" src="https://github.com/user-attachments/assets/d4cfc7d0-d510-47e8-8ea7-95701d1ae2a0" />

Training performance continued improving after the best validation checkpoints. Both hybrids showed substantial validation fluctuations. For the selected model, validation accuracy fell to 64.49% at epoch 9 before recovering. Reported evaluation uses the retained epoch 7 checkpoint, rather than the final epoch. The cause of the abrupt fluctuations was not isolated.

### Confusion matrix and error analysis

![Selected CNN–LSTM confusion matrices showing raw counts and proportions within each actual class](outputs/cnn_lstm_selected_test_confusion_matrices.png)

Rows represent actual categories and columns represent predicted categories. The left panel shows image counts; the right panel shows proportions within each actual class. Diagonal values in the normalized matrix therefore represent class recall.

- Left-lung recall was 100.00%, right-lung recall was 99.17%, and liver recall was 97.72%.
- Left-kidney recall was 65.46%, making it the weakest category by recall.
- The two kidney classes accounted for 1,048 of the 1,910 errors, or 54.87%.
- Common errors included left kidney predicted as spleen (262 images) and left kidney predicted as right kidney (236 images).
- Perfect left-lung recall coexisted with 91.13% precision because 170 images from other categories were assigned that label.

## Limitations

These results describe one completed training run per configuration on one dataset. The selected model's validation accuracy of 98.06% exceeded its test accuracy of 89.26%, a gap of 8.80 percentage points. Variation across seeds, confidence intervals, external validation, and controlled timing benchmarks were not assessed.

The architectures have different classification heads and parameter counts, so the comparison does not isolate the effect of recurrence. The initial model also used an earlier data-pipeline state; identical augmented inputs were not established across all configurations. Images from the same scan should not be treated as independent patient outcomes. The findings concern anatomical crop classification and do not establish diagnostic performance.

## Troubleshooting

- **Dataset download fails:** Check internet access and the configured cache path. Use the official MedMNIST download guidance if an automatic download is unavailable.
- **TensorFlow reports no GPU:** CPU execution is supported. TensorFlow versions after 2.10 do not support CUDA GPU execution on native Windows; consult the [TensorFlow installation guide](https://www.tensorflow.org/install/pip) for WSL2 setup.
- **The script loads models immediately:** Completed checkpoints already exist. Use a fresh model/output location when repeating training.
- **A trained summary shows more parameters:** Keras may include optimizer variables in its displayed total. Use model parameters without optimizer state for the comparison above.
- **Results differ:** Check dataset resolution, partitions, preprocessing, seeds, package versions, and saved checkpoints. Retraining can produce different results.
- **A README image does not appear:** Keep the four referenced PNG files in `outputs/` and preserve their filenames and capitalization.

## Reference

Yang, J., Shi, R., Wei, D., Liu, Z., Zhao, L., Ke, B., Pfister, H., & Ni, B. (2023). MedMNIST v2—A large-scale lightweight benchmark for 2D and 3D biomedical image classification. *Scientific Data, 10*, Article 41. https://doi.org/10.1038/s41597-022-01721-8

