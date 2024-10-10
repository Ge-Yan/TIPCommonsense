# Implicit Knowledge-Augmented Prompting for Commonsense Explanation Generation
## Setup Instructions

Before starting, make sure you have installed all the necessary dependencies and follow the steps below to configure the project.

### 1. Install Requirements

First, install the necessary dependencies for the project by running:
```bash
pip install -r requirements.txt
```
### 2. Download Dataset

Go to [SemEval 2020 Task 4: Commonsense Validation and Explanation](https://github.com/wangcunxiang/SemEval2020-Task4-Commonsense-Validation-and-Explanation) and download the dataset for the subtask. Place the downloaded data in the `data` folder at the project root.

### 3. Configure `config.yaml`

Open the `conf/config.yaml` file and modify the `data_path` field to point to the path of the dataset you just downloaded. For example:

```yaml
data_path: "data/your_dataset_folder"
```
### 4. Extract Concepts

Run the `concept_extractor.py` script to generate the necessary concept file:

```bash
python concept_extractor.py
```

Then update the \texttt{concepts\_path} field in \texttt{conf/config.yaml} to point the concept file. 

### 5. Set Model Name

In `conf/config.yaml`, locate the `model_name` field and set the name of the model you want to use. For example:

```yaml
model_name: "facebook/opt-1.3b"
```
### 6. Run the Main Program

Once the above steps are complete, you can run the main program by executing:

```bash
python run.py
