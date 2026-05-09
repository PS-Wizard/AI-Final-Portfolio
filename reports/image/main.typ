// Compile from this directory with: typst compile --root ../.. main.typ main.pdf

#import "@preview/versatile-apa:7.2.0": abstract-page, title-page, versatile-apa as apa-style

#set document(
  title: [Task II: Traffic Sign Classification ],
  keywords: ("traffic sign classification", "CNN", "MobileNetV2", "transfer learning", "computer vision"),
)

#show: apa-style.with(
  font-size: 12pt,
  running-head: [ ],
)

#set par(
  first-line-indent: 0pt,
  justify: true,
)

#let imgroot = "../../outputs/figures/vision"

#let panel(path, title) = block[
  #image(path, width: 100%)
  #v(-2pt)
  #align(center)[#text(8pt, weight: "semibold")[#title]]
]

#let compact-panel(path, title) = block[
  #image(path, width: 100%, height: 82pt, fit: "contain")
  #v(-2pt)
  #align(center)[#text(8pt, weight: "semibold")[#title]]
]

#let image-figure(number, caption, body) = block[
  #counter(figure).step()
  #set par(first-line-indent: 0pt)
  #text(weight: "bold")[Figure #number]
  #v(3pt)
  #emph[#caption]
  #v(6pt)
  #block(
    width: 100%,
    fill: luma(248),
    inset: 8pt,
    radius: 2pt,
  )[
    #align(center)[#body]
  ]
]

#let table-figure(number, caption, body) = block[
  #set par(first-line-indent: 0pt)
  #text(weight: "bold")[Table #number]
  #v(3pt)
  #emph[#caption]
  #v(6pt)
  #body
]

#let two-up(a, b) = grid(
  columns: (1fr, 1fr),
  column-gutter: 9pt,
  row-gutter: 9pt,
  a, b,
)
#let three-up(a, b, c) = grid(
  columns: (1fr, 1fr, 1fr),
  column-gutter: 7pt,
  row-gutter: 8pt,
  a, b, c,
)

#let metric-table(..rows) = text(size: 10pt)[
  #table(
    columns: (2.5fr, 0.9fr, 1fr, 0.8fr, 0.8fr, 1fr, 0.9fr),
    inset: (x: 3pt, y: 3.2pt),
    align: (left + horizon, right + horizon, right + horizon, right + horizon, right + horizon, right + horizon, right + horizon),
    stroke: 0.35pt + luma(180),
    table.header([Model], [Acc.], [Macro F1], [Errors], [Epochs], [Time], [Params]),
    ..rows,
  )
]

#title-page(
  title: [Task II: Traffic Sign Classification],
  authors: (
    (
      name: [Swoyam Pokharel],
      affiliations: "herald",
    ),
  ),
  affiliations: (
    "herald": [Herald College Kathmandu],
  ),
  course: [6CS012: Artificial Intelligence and Machine Learning],
  instructor: [Jinu Nyachhyon],
  due-date: [2026],
)

#abstract-page([
  This report presents Task II of the final assessment. It is a five-class image classification problem classifying the `Traffic_Sign_2` dataset. This report documents the findings of the various experiments that were produced as a result of answering Task II. The entire project was treated as a controlled model-design study rather than a single model run. The experiment chanin includes baseline CNN variants, two transfer-learning strategies and a deeper CNN ablation study.

  The strongest raw results came from `deeper_cnn_augmentation` which reached `1.00` accuracy, `1.00` macro F1, and `0` misclassified samples, on the test split.

  The strongest transfer learning result came from MobileNetV2's fine tuning, which reached `.9963` accuracy and `.9936` macro F1, while using about ~5x less trainable parameters than the scratch CNN family.

  Overall, the dataset was highly learnable; even just the baseline CNN performed extremely well reaching `.997` accuracy, and `.995` macro F1. The experiment sequence showed that extra components as class weighting, batch normalization, dropout and SDG don't automatically help improve a model.
])

#outline(title: [Contents])
#pagebreak()

= Introduction

Traffic sign classification is more of a practical problem because road-sign recognition is a core part of how driver-assistance and autonomous-vehicle systems understand road environments. A useful classifier has to separate signs that may share similar colours, shapes, and backgrounds, while still responding correctly to the specific visual symbol inside each sign.

The assignment requires two main modelling approaches: a convolutional neural network built from scratch and a transfer-learning model adapted from a pretrained network, with the main question being *which modelling choices improve traffic sign classification most effectively, and what do those choices cost in training time, model size, and generalization?* To answer that, the experiments compare a baseline CNN family, a deeper CNN ablation study, an optimizer trial, and MobileNetV2 transfer learning.

Accuracy is reported throughout the report, but it is not used alone. The dataset is imbalanced, with `SpeedLimit` containing far more images than the smaller classes such as `Cautions` and `Crossings`. Because of that, macro F1 is used to keep minority-class performance visible, while confusion matrices are used to show the actual error patterns behind the headline scores.

The report follows the experiment sequence used in the notebook. First, the baseline CNN is tested with and without augmentation and class weighting. Second, a deeper CNN is evaluated through augmentation, batch normalization, dropout, and their combination. Third, Adam is compared against SGD with momentum on the same deeper architecture. Finally, MobileNetV2 is tested through feature extraction and fine-tuning. All in all, this report covers 10 total experiments and the final sections combine all results, compare model complexity against performance.

#pagebreak()

= Dataset and Preprocessing

\

== Dataset overview

The labelled part of the dataset contained `16,065` usable images after the integrity scan. The scan also found `35` corrupted images, which were recorded but ignored. The separate `Test` folder contained only `10` unlabeled images, so those images were reserved for qualitative inference instead of formal scoring.

#line(length: 100%, stroke: 0.4pt + luma(160))

\

#image-figure(
  1,
  [Dataset overview and representative traffic sign examples.],
  two-up(
    panel(imgroot + "/eda/dataset_eda_summary.png", [Dataset summary and raw distributions]),
    panel(imgroot + "/eda/sample_images_per_class.png", [Representative image samples from each class]),
  ),
)

#pagebreak()
\
The dataset has five classes:
- `Cautions`,
- `Crossings`,
- `Direction`,
- `No Entry`,
- `SpeedLimit`.

The imbalance is visible immediately. `SpeedLimit` has `6,681` images, while `Cautions` has only `1,671`. Although this does not make the dataset immediately unusable, it does mean that accuracy alone can hide weak minority-class behaviour, thus macro F1 is a better metric for evaluation.

#table-figure(
  1,
  [Class distribution after dataset scan.],
  table(
    columns: (1fr, 0.8fr, 2fr),
    inset: 4pt,
    align: left,
    stroke: 0.35pt + luma(180),
    table.header([Class], [Images], [Role in dataset]),
    [Cautions], [1,671], [smallest labelled class],
    [Crossings], [1,821], [minority class],
    [Direction], [2,961], [medium class],
    [No Entry], [2,931], [medium class],
    [SpeedLimit], [6,681], [majority class],
  ),
)

== Split strategy

The labelled images were split into train, validation, and test sets using a *stratified* `70/15/15` split. Stratified matters because every model is compared on the same class-balanced split structure rather than on a random split that might accidentally make one experiment easier than another.

#image-figure(
  2,
  [Class distribution across train, validation, and test splits.],
  align(center)[#image(imgroot + "/eda/split_distribution.png", width: 82%)],
)
\
The final split was consistent across classes:
- `Cautions` had `1170/250/251` train/validation/test images,
- `Crossings` had `1275/273/273`,
- `Direction` had `2072/445/444`,
- `No Entry` had `2052/439/440`,
- `SpeedLimit` had `4676/1003/1002`.

#pagebreak()

== Preprocessing and augmentation

The scratch CNN models used `128 x 128` RGB images normalized to `[0, 1]`, however the MobileNetV2 used `224 x 224` inputs and the expected MobileNetV2 preprocessing function instead of the `[0,1]` normalization.


#image-figure(
  3,
  [Augmentation examples using rotation, zoom, and translation.],
  align(center)[#image(imgroot + "/eda/augmentation_examples.png", width: 82%)],
)

#pagebreak()
= Experimental Setup

All experiments were implemented in TensorFlow/Keras with a fixed seed of `42`. Everything was run locally while utilizing the system's GPU. The codebase was split into separate Python modules rather than being placed entirely inside the notebook.  This structure made the notebook more of an experiment driver rather than the only place where the actual logic lived.

```text
src/vision/
├── data.py          # scan, split, preprocess, and build tf.data pipelines
├── eda.py           # dataset summaries, split plots, and augmentation figures
├── experiments.py   # train, evaluate, save metrics, predictions, and models
├── inference.py     # qualitative inference and error-analysis galleries
└── models.py        # scratch CNN and MobileNetV2 model definitions

outputs/
├── figures/
│   ├── eda/                 # dataset scan, samples, split, augmentation
│   ├── training_curves/     # accuracy/loss curves for each experiment
│   ├── confusion_matrices/  # test-set confusion matrices
│   ├── comparisons/         # global model comparison plots
│   ├── error_analysis/      # misclassified sample galleries
│   └── inference_examples/  # qualitative predictions on held-out images
├── histories/               # per-experiment training history files
├── model_summaries/         # saved architecture summaries
├── models/                  # saved trained models
├── predictions/             # per-sample predictions and confidence outputs
└── tables/                  # metrics and comparison tables
```

\

Every scored model used the same labelled test split. Training histories, metrics, confusion matrices, prediction files, model summaries, and saved model files were written under `outputs/`. This was done so that each run had it's evidence that could be checked again later, rather than relying on screenshots, notebook memory, or manually copied numbers.

Most experiments used Adam. A separate optimizer comparison tested SGD with momentum on the same deeper architecture.

\

= Baseline CNN Study

== Architecture

The baseline CNN establishes the starting point. It uses three convolutional blocks followed by max pooling, then a dense classifier head and a five-class softmax output.

#image-figure(
  4,
  [Baseline CNN architecture used for the scratch-model starting point.],
  align(center)[#image("diagrams/baseline_cnn.svg", width: 92%)],
)

== Baseline experiments

#par(first-line-indent: 0pt)[
  Three baseline-family experiments were run:
]
- The plain baseline CNN,
- The same baseline with augmentation,
- The same baseline with class weights.

#par(first-line-indent: 0pt)[
  This was done to test if the model benefits from extra regularization or imbalance correction before moving to deeper models.
]


#image-figure(
  5,
  [Training curves for the baseline-family experiments.],
  three-up(
    panel(imgroot + "/training_curves/baseline_cnn_curves.png", [Baseline CNN]),
    panel(imgroot + "/training_curves/baseline_cnn_augmentation_curves.png", [Baseline + augmentation]),
    panel(imgroot + "/training_curves/baseline_cnn_class_weights_curves.png", [Baseline + class weights]),
  ),
)

#image-figure(
  6,
  [Confusion matrices for the baseline-family experiments.],
  three-up(
    panel(imgroot + "/confusion_matrices/baseline_cnn_cm.png", [Baseline CNN]),
    panel(imgroot + "/confusion_matrices/baseline_cnn_augmentation_cm.png", [Baseline + augmentation]),
    panel(imgroot + "/confusion_matrices/baseline_cnn_class_weights_cm.png", [Baseline + class weights]),
  ),
)

#table-figure(
  2,
  [Baseline-family comparison.],
  metric-table(
    [Baseline],
    [0.9971],
    [0.9953],
    [7],
    [15],
    [151.32s],
    [8.52M],
    [Baseline + Aug.],
    [0.9967],
    [0.9952],
    [8],
    [14],
    [145.22s],
    [8.52M],
    [Baseline + CW],
    [0.9946],
    [0.9926],
    [13],
    [9],
    [89.47s],
    [8.52M],
  ),
)
\
The plain baseline CNN was the strongest model in this family, reaching `0.9971` accuracy, `0.9953` macro F1, and only `7` misclassified samples. Augmentation was almost identical but slightly lower, with `8` misclassified samples. Class weighting performed worst in the baseline family, with `13` errors and a lower macro F1.

The takeaway then is that the baseline CNN already learned the dataset extremely well. Class imbalance exists, but class weighting does not help this image model. Augmentation also was not better at the shallow baseline stage.

\
#line(length: 100%, stroke: 0.4pt + luma(160))
\

= Deeper CNN Experiments

== Architecture and rationale

The deeper CNN increases capacity by using two convolutional layers per block instead of one. Although augmentation was not strictly better in the shallow baseline family, it was kept as the common base for the deeper CNN experiments so that the ablation could test what happens when batch normalization and dropout are added to the same deeper augmented pipeline?

#image-figure(
  7,
  [Deeper CNN architecture and the ablation points for batch normalization and dropout.],
  align(center)[#image("diagrams/deeper_cnn.svg", width: 94%)],
)


We need to recall that the baseline was already strong, if a deeper model improves performance, we need to answer whether the improvement comes from extra depth, augmentation, dropout, batch normalization, or some other specific combination.

#pagebreak()

== Results

#par(first-line-indent: 0pt)[
  The deeper CNN family contained four experiments:
]
- `deeper_cnn_augmentation`,
- `deeper_cnn_augmentation_batchnorm`,
- `deeper_cnn_augmentation_dropout`,
- `deeper_cnn_augmentation_batchnorm_dropout`.
\

#image-figure(
  8,
  [Training curves for the deeper CNN ablation variants.],
  grid(
    columns: (1fr, 1fr),
    column-gutter: 9pt,
    row-gutter: 9pt,
    compact-panel(imgroot + "/training_curves/deeper_cnn_augmentation_curves.png", [Deeper + augmentation]),
    compact-panel(imgroot + "/training_curves/deeper_cnn_augmentation_dropout_curves.png", [Deeper + augmentation + dropout]),
    compact-panel(
      imgroot + "/training_curves/deeper_cnn_augmentation_batchnorm_curves.png",
      [Deeper + augmentation + batch normalization],
    ),
    compact-panel(
      imgroot + "/training_curves/deeper_cnn_augmentation_batchnorm_dropout_curves.png",
      [Deeper + augmentation + batch normalization + dropout],
    ),
  ),
)
\
These figures show the training and validation behaviour for the four deeper CNN ablation variants. An important comparison is not just which curve reaches the highest accuracy, but how stable the validation curve remains as extra components are added. The plain deeper augmented model shows the cleanest learning pattern, while the dropout and batch-normalized variants introduce more variation without producing a better final result.

\
#image-figure(
  9,
  [Confusion-matrix comparing the best and worst Deeper-CNN variants.],
  two-up(
    panel(imgroot + "/confusion_matrices/deeper_cnn_augmentation_cm.png", [Best model: deeper + augmentation]),
    panel(imgroot + "/confusion_matrices/deeper_cnn_augmentation_batchnorm_dropout_cm.png", [Weakest Adam ablation]),
  ),
)
\
#table-figure(
  3,
  [Deeper CNN ablation results.],
  metric-table(
    [Deep + Aug.],
    [1.0000],
    [1.0000],
    [0],
    [15],
    [270.61s],
    [8.72M],
    [Deep + Aug. + Drop.],
    [0.9983],
    [0.9977],
    [4],
    [15],
    [315.33s],
    [8.72M],
    [Deep + Aug. + BN],
    [0.9946],
    [0.9926],
    [13],
    [10],
    [378.34s],
    [8.72M],
    [Deep + Aug. + BN + Drop.],
    [0.9809],
    [0.9659],
    [46],
    [15],
    [556.53s],
    [8.72M],
  ),
)

The strongest model in the entire Vision study was `deeper_cnn_augmentation`. It reached `1.0000` accuracy, `1.0000` macro F1, and `0` misclassified samples on the held-out labelled test split.

The deeper dropout variant was also strong, but it did not beat the plain deeper augmented model. Batch normalization alone was weaker, and the batch-normalization-plus-dropout combination was much worse. Thus the takewaway from this ablation is that extra regularization is not automatically useful. In this dataset and architecture, the simplest deeper augmented variant performs better.

= Optimizer Comparison

The optimizer comparison used the same deeper batch-normalization-plus-dropout architecture and changed only the optimizer. Adam was compared against SGD with momentum. This keeps the interpretation narrow: the result is about optimizer behaviour under this configuration.

#image-figure(
  10,
  [Optimizer comparison training curves using the same deeper BN+dropout architecture.],
  two-up(
    panel(imgroot + "/training_curves/deeper_cnn_augmentation_batchnorm_dropout_curves.png", [Adam]),
    panel(imgroot + "/training_curves/deeper_cnn_augmentation_batchnorm_dropout_sgd_curves.png", [SGD + momentum]),
  ),
)

#image-figure(
  11,
  [Optimizer comparison confusion matrices.],
  two-up(
    panel(imgroot + "/confusion_matrices/deeper_cnn_augmentation_batchnorm_dropout_cm.png", []),
    panel(
      imgroot + "/confusion_matrices/deeper_cnn_augmentation_batchnorm_dropout_sgd_cm.png",
      [],
    ),
  ),
)

#table-figure(
  4,
  [Optimizer comparison.],
  table(
    columns: (1.8fr, 1fr, 1fr, 0.8fr, 0.8fr, 1fr),
    inset: (x: 3pt, y: 3.2pt),
    align: (left + horizon, right + horizon, right + horizon, right + horizon, right + horizon, right + horizon),
    stroke: 0.35pt + luma(180),
    table.header([Optimizer], [Accuracy], [Macro F1], [Errors], [Epochs], [Time]),
    [Adam], [0.9809], [0.9659], [46], [15], [556.53s],
    [SGD + momentum], [0.4158], [0.1175], [1,408], [7], [241.50s],
  ),
)

Adam was far more stable in this setup. SGD with momentum used a learning rate of `1e-2` with momentum set to `0.9`, but it collapsed to `0.4158` accuracy and `0.1175` macro F1, with `1,408` misclassified samples.

The conclusion is not that SGD is universally bad but that the SGD configuration was unsuitable for this architecture and dataset. In later review, I've realized that the `1e-2` learning rate was likely too large for this SGD setup. Optimizer configuration was a real bottleneck here.

#pagebreak()
= Transfer Learning with MobileNetV2

== Architecture and rationale

MobileNetV2 was used as the transfer-learning backend because it comparetively lighter amongst the larger ImageNet models while still providing pretrained visual features. Two strategies were tested: feature extraction and fine-tuning. 

#image-figure(
  12,
  [MobileNetV2 transfer-learning experiment setup],
  align(center)[#image("diagrams/mobilenetv2_transfer.svg", width: 94%)],
)

== Results

#image-figure(
  13,
  [Training curves for MobileNetV2 transfer-learning experiments.],
  two-up(
    panel(imgroot + "/training_curves/mobilenetv2_feature_extraction_curves.png", [Feature extraction]),
    panel(imgroot + "/training_curves/mobilenetv2_finetuning_curves.png", [Fine-tuning]),
  ),
)

#image-figure(
  14,
  [Confusion matrices for MobileNetV2 transfer-learning experiments.],
  two-up(
    panel(imgroot + "/confusion_matrices/mobilenetv2_feature_extraction_cm.png", [Feature extraction]),
    panel(imgroot + "/confusion_matrices/mobilenetv2_finetuning_cm.png", [Fine-tuning]),
  ),
)

#table-figure(
  5,
  [Transfer-learning comparison.],
  metric-table(
    [MobileNetV2 FE],
    [0.9950],
    [0.9913],
    [12],
    [10],
    [231.12s],
    [0.33M],
    [MobileNetV2 FT],
    [0.9963],
    [0.9936],
    [9],
    [5],
    [132.50s],
    [1.68M],
  ),
)

Fine-tuning outperformed feature extraction. It reached `0.9963` accuracy, `0.9936` macro F1, and only `9` misclassified samples. Feature extraction was slightly weaker but still strong, reaching `0.9950` accuracy and `0.9913` macro F1.

The important point is efficiency. The scratch CNN models used around `8.5M--8.7M` trainable parameters, while MobileNetV2 fine-tuning used around `1.68M`. It did not beat the best scratch CNN, but it came close with a much smaller trainable footprint. That makes it the strongest practical model if parameter efficiency matters.

= Global Comparative Analysis

All in all, the experiment registry proves that the dataset is highly learnable, but it also shows that the path to the best model isn't obvious . Some additions helped, some were neutral, and some made the result worse.

#image-figure(
  15,
  [Macro F1 ranking across all Vision experiments.],
  align(center)[#image(imgroot + "/comparisons/macro_f1_comparison.png", width: 88%)],
)

#pagebreak()
#image-figure(
  16,
  [Training time versus accuracy across all Vision experiments.],
  image(imgroot + "/comparisons/training_time_vs_accuracy_clean.png", width: 100%),
)

#table-figure(
  6,
  [Top five Vision experiments ranked by macro F1.],
  table(
    columns: (0.7fr, 2fr, 1fr, 1fr, 0.8fr),
    inset: (x: 3pt, y: 3.2pt),
    align: (center + horizon, left + horizon, right + horizon, right + horizon, right + horizon),
    stroke: 0.35pt + luma(180),
    table.header([Rank], [Model], [Accuracy], [Macro F1], [Errors]),
    [1], [Deep + Aug.], [1.0000], [1.0000], [0],
    [2], [Deep + Aug. + Dropout], [0.9983], [0.9977], [4],
    [3], [Baseline CNN], [0.9971], [0.9953], [7],
    [4], [Baseline + Aug.], [0.9967], [0.9952], [8],
    [5], [MobileNetV2 FT], [0.9963], [0.9936], [9],
  ),
)

The best raw model was `deeper_cnn_augmentation`. The best transfer-learning model was `mobilenetv2_finetuning`. The baseline CNN was already very strong, which is why the later improvements are small in the absolute sense. However, the experiments still gave us meaningful results because it shows that class weighting, batch normalization, and the tested SGD setup were not useful in this specific workflow.

The efficiency picture is also worth re-mentioning. The highest-scoring scratch model used more than eight million trainable parameters. MobileNetV2 fine-tuning used far fewer trainable parameters and still reached near-baseline performance. 

Therefore, the choice choice therefore really depends on what we are aiming for, maximum held-out test score or a smaller trainable model with strong practical performance.

\

= Error Analysis and Qualitative Inference

The best overall model made zero errors on the held-out labelled test set, so its error analysis is not very informative. For inspection, MobileNetV2 fine-tuning is more useful because it is still strong but has a small number of mistakes and low-confidence cases to inspect.

#image-figure(
  17,
  [error analysis and inference examples.],
  two-up(
    panel(
      imgroot + "/error_analysis/mobilenetv2_finetuning_misclassified_gallery.png",
      [MobileNetV2 fine-tuning misclassified examples],
    ),
    panel(imgroot + "/inference_examples/unlabeled_test_inference_gallery.png", [Unlabeled test-folder predictions]),
  ),
)

#image-figure(
  18,
  [Confidence inspection of MobileNetV2 fine-tuning.],
  two-up(
    panel(
      imgroot + "/inference_examples/mobilenetv2_finetuning_high_confidence_correct.png",
      [High-confidence correct predictions],
    ),
    panel(
      imgroot + "/inference_examples/mobilenetv2_finetuning_low_confidence_predictions.png",
      [Low-confidence predictions],
    ),
  ),
)

Here, the unlabeled test images are not formal evaluation because ground-truth labels are unavailable. They only show inference behaviour on unseen files. 

The low-confidence labelled examples also help explain why the remaining mistakes are not only caused by blur. The bicycle-crossing image specifically, is visually clearer than several other samples, but it still has the same red triangular warning-sign structure that appears strongly in the `Cautions` class. The model therefore appears to confuse the broad sign shape and warning-sign context with the more specific bicycle-crossing symbol, which is why it predicts `Cautions` with only moderate confidence rather than confidently recognizing it as `Crossings`.

#pagebreak()
= Model Complexity and Efficiency

The model-complexity and performance is inheretly a tradeoff, and we need to find a balance between the two. The scratch CNN family achieved the best score, but it also carried the largest trainable parameter count. The MobileNetV2 models were smaller in trainable terms because most of the representation came from the pretrained backend.

Thus the takeaway is pretty straightforward:

- If the goal is the best held-out score, `deeper_cnn_augmentation` wins.
- If the goal is a smaller trainable model with strong performance, `mobilenetv2_finetuning` is the better practical option.

#image-figure(
  19,
  [Saved model disk size versus test accuracy.],
  image(imgroot + "/comparisons/model_size_vs_accuracy.png", width: 100%),
)

The saved model files show the same efficiency trade-off in a more deployment-facing way. The best scratch CNN models were around `98--100 MB` on disk, while `mobilenetv2_finetuning` was only about `23.24 MB` and still reached `0.9963` accuracy. This does not make MobileNetV2 the best raw scorer, but it makes the storage-performance compromise much stronger.

Training time also did not perfectly predict performance. Some slower variants were weaker than faster ones. The `batch-normalization-plus-dropout` model took the longest in the deeper family but was not competitive with the simpler deeper augmented model. More computation was not automatically buying better generalization.

#table-figure(
  7,
  [Training time versus performance evidence.],
  table(
    columns: (2.4fr, 1fr, 1fr, 0.9fr, 1fr),
    inset: (x: 3pt, y: 3.2pt),
    align: (left + horizon, right + horizon, right + horizon, right + horizon, right + horizon),
    stroke: 0.35pt + luma(180),
    table.header([Model], [Accuracy], [Macro F1], [Errors], [Time]),
    [Deep + Aug.], [1.0000], [1.0000], [0], [270.61s],
    [Deep + Aug. + Drop.], [0.9983], [0.9977], [4], [315.33s],
    [MobileNetV2 FT], [0.9963], [0.9936], [9], [132.50s],
    [Deep + Aug. + BN], [0.9946], [0.9926], [13], [378.34s],
    [Deep + Aug. + BN + Drop.], [0.9809], [0.9659], [46], [556.53s],
  ),
)

= Limitations

The experiments have several limitations. First, the perfect test-set result should be interpreted carefully. It shows that the selected model solved the held out, labelled, test split of `2,410` images, not that it would remain perfect under real-world road conditions. If this same model was deployed in a real world scenario, it would almost certainly perform much wrose. Deployment's data would contain more variation in camera quality, lighting, blur, occlusion, weather, viewing angle, and sign damage.

There are also experimental limits. 
- Only one transfer-learning backbone was tested.
- The SGD comparison used one learning-rate and momentum setup, so it only shows that this configuration failed. 
- The unlabeled test folder could not be scored because labels were not provided. Finally, the results are tied to this dataset, preprocessing pipeline, and split.


Although these limitations do not invalidate the result, it's worth mentioning them as they define the scope.

= Conclusion

The baseline CNN already performed extremely well, but the deeper CNN with augmentation produced the strongest overall result: `1.0000` accuracy, `1.0000` macro F1, and `0` misclassified samples on the held-out labelled test split.

The main lesson is that extra complexity needs evidence. Class weighting did not help the image models. Batch normalization and dropout together underperformed. SGD with momentum failed under the tested configuration. 

In contrast, deeper convolutional capacity with augmentation worked extremely well, and MobileNetV2 fine-tuning provided a strong efficiency trade-off.

The final choice depends on the priority. For maximum score on this dataset, `deeper_cnn_augmentation` is the best model. For a smaller trainable model with strong practical performance, `mobilenetv2_finetuning` is the stronger compromise. 
