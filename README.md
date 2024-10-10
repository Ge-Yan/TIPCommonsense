# Implicit Knowledge-Augmented Prompting for Commonsense Explanation Generation
## Setup Instructions

Before starting, make sure you have installed all the necessary dependencies and follow the steps below to configure the project.

### 1. Install Requirements

First, install the necessary dependencies for the project by running:
```bash
pip install -r requirements.txt

### 2. Download Dataset

Go to [SemEval 2020 Task 4: Commonsense Validation and Explanation](https://github.com/wangcunxiang/SemEval2020-Task4-Commonsense-Validation-and-Explanation) and download the dataset for the subtask. Place the downloaded data in the `data` folder at the project root.

### 3. Configure `config.yaml`

Open the `conf/config.yaml` file and modify the `data_path` field to point to the path of the dataset you just downloaded. For example:

```yaml
data_path: "data/your_dataset_folder"
