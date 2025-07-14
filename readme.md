# <img src='https://github.com/fmhoward/TIGER/blob/main/TIGER.png?raw=true' height = 80px>
TIGER (<b>T</b>ransformer-based h<b>I</b>stology-driven <b>G</b>ene <b>E</b>xpression <b>R</b>egressor) is a pipeline to accurately reconstruct gene expression and gene expression signatures from digital histology. These histology derived gene expression signatures can be used to preidct prognosis, chemotherapy response, and response to specific therapies.


## Attribution
If you use this code in your work or find it helpful, please consider citing our paper in <a href=''>bioRxiv</a>.
```
```

## Installation
This github repository should be downloaded to a project directory. Installation takes < 5 minutes on a standard desktop computer. Runtime for training our signature models on the TCGA training dataset requires approximately 5 minutes for 8 epochs on 1 A100 GPU. Subsequent predictions for inference can be run in approximately 1 minute. All software was tested on CentOS 8 with an AMD EPYC 7302 16-Core Processor and 4x A100 SXM 40 GB GPUs.

Requirements:
* python 3.8
* pytorch 1.11
* opencv 4.6.0.66
* scipy 1.9.3
* scikit-image 0.19.3
* OpenSlide
* Libvips 8.9
* pandas 1.5.1
* numpy 1.21.6

For full environment used for model testing please see the environment.yml file

## Comparison of TIGER to Recent Methodologies for Gene Expression Prediction. 
Visualizing synthetic histology from a feature vector is easily performed with HistoXGAN; the included models allow for visualization of CTransPath and RetCCL feature vectors.
Trained models used in this work are available at https://doi.org/10.5281/zenodo.10892176. The trained HistoXGAN models alone can be downloaded from the FINAL_MODELS.rar folder in this Zenodo repository; or the trained models in conjunction with other supplemental data used to evaluate HistoXGAN can be downloaded from the HistoXGAN.rar folder.


Features can be extracted as described in the <a href = 'https://slideflow.dev/'>slideflow source documentation</a>.
```
import slideflow as sf
from slideflow.model import build_feature_extractor

P = sf.Project(".../PROJECT/TIGER/")
dataset = P.dataset(tile_px=224, tile_um=224)
dataset.extract_tiles()
uni = build_feature_extractor('uni', tile_px=224)
P.generate_feature_bags(uni, dataset, outdir = '.../FEATEURES/UNI/', normalizer = normalizer)
```

These features can be converted via an autoencoder (used in <a href='https://github.com/PangeaResearch/enlight-deeppt-data/tree/main'>DEEP-PT</a>) or via k-means clustering (as in <a href='https://github.com/gevaertlab/sequoia-pub'>SEQUOIA</a>) to compare these respective models to our approach.
```
from slideflow.mil.convert_features import *
convert_to_deeppt(bags = ".../FEATEURES/UNI/", outdir = ".../FEATURES/DEEPPT/", model_path = ".../model_AE.pth")
convert_to_sequoia((bags = ".../FEATEURES/UNI/", outdir = ".../FEATURES/SEQUOIA/"
```

To train models with cross-validaiton for gene expression prediction, the following code would be used, replacing model type with 'deeppt', 'sequoia', or 'bistro.transformer' (the optimal model found in our analysis), and specifying the bag directory to foundational model features, or the features generated for DEEP-PT / SEQUOIA as above.
```
model_type = ***
bag_dir = ***
gene_list = ***

from slideflow.mil import mil_config
config = mil_config(model_type, outcome_type="continuous")
splits = dataset.kfold_split(k=3)
for train, val in splits:
	P.train_mil(
		config=config,
		outcomes=gene_list,
		train_dataset=train,
		val_dataset=val,
		bags=bag_dir
	)
```

The trained models can subsequently be used for gene expression inference on external datasets:

```
predict_mil(model = "/path/to/trained_model/",config=config, dataset=external_dataset, outcomes = gene_list, bags = external_bag_dir) 
```

## Validation for Response Prediction in the University of Chicago Cohort and Other Cohorts
### Patch Generation and Feature Extraction
We have provided extracted features from University of Chicago neoadjuvant cohort to illustrate associations of gene signature predictions with response in the /UCMC_NAC/ directory. Tile images can be extracted in an identical fashion

Tile images were extracted from this cohort with the Slideflow architecture using Otsu thresholding and Gaussian Blur based filtering at an effective magnification of 10X. 
```
import slideflow as sf
P = sf.Project("".../PROJECTS/SIGNATURE_PREDICTIONS/") # A slideflow project directory as per <a href = 'https://slideflow.dev/project_setup/'>https://slideflow.dev/project_setup/</a>
P.annotations = ".../TIGER/UCMC_NAC/uc_anon_annotations.csv"
P.sources = ["UCMC_NAC"] # specifies the dataset location
dataset = P.dataset(tile_px=224, tile_um=224)
qc = [
  qc.Otsu(),
  qc.GaussianV2()
]
dataset.extract_tiles(qc=qc, roi_method = 'ignore')
```

UNI features were then extracted and uploaded.
```
from slideflow.model import build_feature_extractor
ctranspath = build_feature_extractor('uni', weights = "...directory of uni weights...")
P.generate_feature_bags(ctranspath, dataset, outdir = 'UCMC_NAC')
```

### Application of Gene Signatures in Breast Cancer for Outcome Prediction
A trained model for gene signature prediction is available <a href='github.com/fmhoward/TIGER/tree/main/signature_QC_model'>here</a>. Predictions can then be made on a per patient basis for these signatures as illustrated below for the UCMC cohort:

```
config = mil_config('bistro.transformer', outcome_type="continuous", aggregation_level="patient", apply_softmax = False)
dataset = P.dataset(tile_px=224, tile_um=224)	
eval_mil(weights = ".../MODELS/signature_QC_model/",
	dataset= dataset
	outcomes = ['pCR'],
	bags = ".../UCMC_NAC/",
	config = config)
```

The predictions can then be loaded from the generated prediction parquet file and correlated with outcomes of interest.

```
pd = read_parquet(".../mil/00001-bistro.transformer/predictions.parquet")
```
