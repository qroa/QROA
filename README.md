# 🛠️ QROA: A Black-Box Query-Response Optimization Attack on LLMs

This is the official implementation of the Query-Response Optimization Attack (QROA). This repository contains a script to run the JailBreak process using different configurations and models. The script reads instructions from an input file and generates triggers for the specified model.

This Script is the official implementation of the article "*QROA: A Black-Box Query-Response Optimization Attack on LLMs*"

Paper: https://arxiv.org/abs/2406.02044

## 📄 Citation
```
@article{jawad2024qroa,
  title={QROA: A Black-Box Query-Response Optimization Attack on LLMs},
  author={Jawad, Hussein and BRUNEL, Nicolas J-B},
  journal={arXiv preprint arXiv:2406.02044},
  year={2024}
}
```

## 📜 Abstract

Large Language Models (LLMs) have recently gained popularity, but they also raise concerns due to their potential to create harmful content if misused. This study in- troduces the Query-Response Optimization Attack (QROA), an optimization-based strategy designed to exploit LLMs through a black-box, query-only interaction. QROA adds an optimized trigger to a malicious instruction to compel the LLM to generate harmful content. Unlike previous approaches, QROA does not require access to the model’s logit information or any other internal data and operates solely through the standard query-response interface of LLMs. Inspired by deep Q-learning and Greedy coordinate descent, the method iteratively updates tokens to maximize a designed reward function. We tested our method on various LLMs such as Vicuna, Falcon, and Mistral, achieving an Attack Success Rate (ASR) over 80%. We also tested the model against Llama2-chat, the fine-tuned version of Llama2 designed to resist Jailbreak attacks, achieving good ASR with a suboptimal initial trigger seed. This study demonstrates the feasibility of generating jailbreak attacks against deployed LLMs in the public domain using black-box optimization methods, enabling more comprehensive safety testing of LLMs


![QROA](img/qroa.png)

## ⚙️ Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/qroa/qroa.git
    cd qroa
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage


### Running the Attack

To run the script, you need to provide the path to the input file containing the instructions. The input file should be in CSV format.
Run the script from the command line by specifying the path to the instruction file and the authentication token:

```bash
python main.py data/instructions.csv [API_AUTH_TOKEN]
```

Replace `instructions.csv` with the path to your text file containing the instructions, and `API_AUTH_TOKEN` with the actual authentication token.

### 🧪 Demo and Testing Model Generation
- **Notebook Demo:** Run `demo.ipynb` to see a demonstration of the process.
- **Notebook Analysis Experiement:** Run `analysis.ipynb` to analyse results and calculate metrics value (ASR).
- **Testing Model:** Generation: Execute `generate.py` to test the generation process on custom instructions and triggers.

This script can be run from the command line as follows:

```bash
python generate.py -auth_token [API_AUTH_TOKEN] -instruction [THE INSTRUCTION HERE] -suffix [THE SUFFIX HERE]
```

Where: 

- **auth_token:** Authentication token required for accessing the model.
- **instruction:** The specific instruction you want the model to follow.
- **suffix:** The adversarial trigger that, when appended to the instruction, causes the LLM to obey the instruction.

### 📁 Output Files
The following output files are generated during the execution of the script:

Generated and validated triggers are saved in JSON format:

- **Generated Triggers:** `./results/[MODEL_NAME]/triggers.json`  : Contains the triggers generated by the model.
- **Validated Triggers:** `./results/[MODEL_NAME]/triggers_validate.json` : Contains the triggers validated after applying the z test.

Logs for generation and validation processes are also available:

- **Trigger Generation Logs:** `./logs/[MODEL_NAME]/logging_generator.csv` : Logs the process of trigger generation.
- **Trigger Validation Logs:** `./logs/[MODEL_NAME]/logging_validator.csv` : Logs the process of validating the triggers with the z test.

## 🔧Configuration Settings

The following table outlines the configuration settings for the JailBreak process. Each parameter plays a role in the setup and execution of the process:

| Parameter              | Description |
|------------------------|-------------|
| `model`                | Specifies the Large Language Model (LLM) to be used for the attack, such as 'vicuna_hf', 'falcon_hf', etc.|
| `apply_defense_methods`| A boolean parameter that determines whether defense methods are activated to protect the model during the JailBreak process.|
| `auth_token`           | Authentication token required for accessing the model. This token could be from Hugging Face for accessing their models or other providers like OpenAI for closed source models. |
| `system_prompt`        | The initial message or command that initiates interaction with the LLM. |
| `embedding_model_path` | Path to the surrogate model's embedding layer. |
| `len_coordinates`      | Specifies the number of tokens in the generated trigger, defining the length of the attack vector. |
| `learning_rate`        | Learning rate for the optimizer. |
| `weight_decay`         | Weight decay (L2 penalty) for the optimizer. |
| `nb_epochs`            | The total number of training cycles through the dataset where the model learns by adjusting internal parameters. |
| `batch_size`           | Number of training examples used to calculate gradient and update internal model parameters per iteration. |
| `scoring_type`         | Method used to evaluate the effectiveness of triggers. For example, 'hm' could refer to a scoring model that uses a fine-tuned RoBERTa model for detecting harmful content. |
| `max_generations_tokens` | Maximum number of tokens that the LLM is allowed to generate in response to a query during the attack. |
| `topk`                 | The top K value triggers identified in each epoch, equivalent to the number of queries sent to the target LLM. |
| `max_d`                | The maximum size of the memory buffer. |
| `ucb_c`                | The exploration-exploitation parameter for the Upper Confidence Bound (UCB) algorithm. A higher value encourages exploration of less certain actions. |
| `triggers_init`        | Initial triggers used as a starting point for the algorithm; these triggers are used to pre-fill the memory buffer to avoid starting from scratch. |
| `threshold`            | The statistical significance threshold used when validating triggers. |
| `nb_samples_per_trigger` | Number of samples per trigger for statistically validating the efficiency of the trigger. |
| `logging_path`         | Path to the logging directory. |
| `results_path`         | Path to the results directory. |
| `temperature`          | Sampling temperature used by the LLM. |
| `top_p`                | Top P value for nucleus sampling used by the LLM. |
| `p_value`              | P-value for statistical testing. |