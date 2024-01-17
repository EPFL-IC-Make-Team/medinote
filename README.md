<img src="figures/medinote.png" width="30%">

>
MediNote is a suite of open-source medical Large Language Models (LLMs) trained for **clinical note generation**.

[MediNote-7B](https://huggingface.co/AGBonnet/medinote-7b) and [MediNote-13B](https://huggingface.co/AGBonnet/medinote-13b) are fine-tuned from the [Meditron](https://arxiv.org/abs/2311.16079) foundation model to generate clinical notes from doctor-patient conversations.

### Models

We fine-tune MediTron models, variants of Llama-2 whose pre-training was extended to PubMed articles, abstracts and clinical practice guidelines. 

<p align="center">
    <img src="figures/model_pipeline.png" width="80%"> 
</p>

### Data

Our primary source of clinical notes is the [**PMC-patients**](https://arxiv.org/abs/2202.13876) dataset. This large-scale dataset contains 167K patient summaries extracted from open-access case studies published in PubMed Central. 

Distribution of confidential patient-doctor conversations is forbidden, so no large scale dataset is publicly available for training.
We circumvent the lack of real dialogue data by building upon [**NoteChat**](https://huggingface.co/datasets/akemiH/NoteChat), an extension of PMC-Patients with synthetic patient-doctor conversations generated with ChatGPT.

We augment the NoteChat dataset by extracting structured patient information from clinical notes as fine-tuning data for the intermediary step in chained CNG models. To do so, we prompt GPT-4 with zero-shot instructions, providing clinical notes and a structured template of patient medical information with feature definition. This template encapsulates crucial aspects of a clinical note such as the patient's admission to a care center, medical history, current symptoms, as well as the doctor's diagnosis and treatment plan. We release the resulting [**Augmented Clinical Notes**](https://huggingface.co/datasets/AGBonnet/augmented-clinical-notes) shown below.

<p align="center">
    <img src="figures/data_pipeline.png" width="90%"> 
</p>


### **Write your own clinical notes**

You can use our [MediNote-7B](https://huggingface.co/AGBonnet/medinote-7b) and [MediNote-13B](https://huggingface.co/AGBonnet/medinote-13b) generator models to writes notes directly from patient-doctor conversations.

You can load either model directly from Huggingface to a chosen path as follows:

````python
# Load model and tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("AGBonnet/medinote-7b")
model = AutoModelForCausalLM.from_pretrained("AGBonnet/medinote-7b"

# Save to local directory
model_path = "model/medinote-13b"
tokenizer.save_pretrained(model_path)
model.save_pretrained(model_path)
````

Note that dialogues be formatted as follows:

```
Doctor: Hello, how are you doing today?
Patient: Not so well! I have a headache.
```

You can then generate your own note with `utils/infer.py` (which uses [vLLM](https://github.com/vllm-project/vllm)):

```bash
./utils/infer.sh \
    --model_path model/medinote-7b \
    --dialogue "Doctor: Hello, how are you doing today?\nPatient: Not so well! I have a headache."
```

### **Reproducibility**

To replicate our experiments, you can run our data preprocessing script by first creating the Augmented Clinical Notes dataset as follows: 

```bash
python utils/data.py
```

Alternatively, you can directly download the [Augmented Clinical Notes](https://huggingface.co/datasets/AGBonnet/augmented-clinical-notes) dataset directly from Huggingface as follows: 

````python
from datasets import load_dataset
dataset = load_dataset("AGBonnet/augmented-clinical-notes")
``````

You can fine-tune your own MediNote-7B by loading the [MediTron-7B](https://huggingface.co/epfl-llm/meditron-70b) from Huggingface and using the [Megatron-LLM](https://github.com/epfLLM/Megatron-LLM) distributed trainer code. Note that the MediTron-13B we used as base for MediNote-13B is not publicly available. See the documentation for a detailed description of the training procedure. 

<p align="center">
    <img src="figures/eval_pipeline.png" width="80%"> 
</p>

Finally, you can run the full inference for all models shown above using `infer.sh`: 

```bash
./utils/infer.sh all
```


Once inference is done, you can evaluate generated patient summaries using the `eval.ipynb` notebook.

### Supplementary material

Here is the template we used to generate patient summaries from the NoteChat dataset (available in JSON format in `generation/templates/template_definitions.json`)
<p align="center">
    <img src="figures/template.png" width="80%"> 
</p>

Here are the prompts used for training and inference:
<p align="center">
    <img src="figures/prompts.png" width="70%"> 
</p>

### Acknowledgments

This project is a contribution to the [2023 MAKE Initiative for Generative AI](https://make.epfl.ch/projects/generative-ai) at the Swiss Federal Institute of Technology (EPFL). This project was initiated and funded by Prof. Antoine Bosselut of NLP lab and Prof Martin Jaggi of MLO lab. 
We also thank Alexandre Sallinen for his advice on chaining specialized LLMs and Prof. Mary-Anne Hartley for her advice on the appropriate template for medical patient summaries. 

### Citation

If you use this code or our models, please cite the following paper:

```
ADD PAPER
```