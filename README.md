<img src="figures/medinote.png" width="30%">

### **MediNote**: Automatic Clinical Note Generation

MediNote is a suite of open-source medical Large Language Models (LLMs) fine-tuned for clinical note generation from the [Meditron](https://arxiv.org/abs/2311.16079) foundation model. 

MediNote-7B and MediNote-13B are trained to generate clinical notes from doctor-patient conversations.

<img src="figures/model_pipeline.pdf" width="100%">

#### *Meditron*: Base model
For both of these tasks, we will fine-tune [Meditron](https://huggingface.co/epfl-llm/meditron-70b), a version of Llama-2-13B whose pre-training was extended to PubMed articles, abstracts and clinical practice guidelines. **Note**: *Whether we fine-tune two separate versions (one for each task) or a single version for both tasks is still to be determined.*


#### *NoteChat*: Fine-tuning data

For our fine-tuning data, we will use the [NoteChat](https://arxiv.org/abs/2310.15959) dataset [Huggingface link](https://huggingface.co/datasets/akemiH/NoteChat). This dataset contains 167K pairs of real clinical notes extracted from [PMC-Patients](https://arxiv.org/abs/2202.13876) with generated patient-doctor conversations. These pairs are generated using three `GPT-3.5-Turbo` submodules (Planning, Roleplaying and Polishing). The authors show that this approach leads to better dialogue quality than using GPT-4 or ChatGPT alone. 

<img src="figures/notechat.png" width="100%">

We extend this dataset of pairs to triplets with GPT-3.5: 

(`clinical note`, `dialogue transcript`) $\to$ (`clinical note`, `dialogue transcript`, `patient summary`)

We extract patient summaries by prompting GPT-3.5 with a clinical note, its corresponding dialogue transcript and a comprehensive template of patient features. 

### Model 1: Summarizer

We fine-tune Meditron on the (`dialogue transcript`, `patient summary`) pairs extracted from NoteChat.


<p align="center">
<img src="figures/extractor.png" width="50%">
</p>

### Model 2: Generator

We fine-tune Meditron on the (`patient summary`, `clinical note`) pairs extracted from NoteChat.

<p align="center">
<img src="figures/generator.png" width="50%">
</p>

**Few-shot prompting**: During generation, we may use a few samples of the clinician's own notes to condition the model to the clinician's writing style.


### **Running our code**

First prepare the NoteChat data and generate triplets using:

```bash
python utils/data.py
```

Running inference is done as follows:

```bash
python3 utils/inference.py \
    --model_name meditron-7b-summarizer \
    --model_path /pure-mlo-scratch/make_project/trial-runs/meditron-7b-summarizer/hf_checkpoint/ \
    --input_path /pure-mlo-scratch/make_project/data/inference/summaries_30K.jsonl \
    --output_path /pure-mlo-scratch/make_project/data/inference/generation.jsonl \
    --num_samples 1000 \
    --mode summarizer \
    --template_path /pure-mlo-scratch/make_project/ClinicalNotes/generation/templates/template.json \
```

Alternatively, you can run the full inference for all models using `infer.sh`: 

```bash
./infer.sh
```
Once inference is done, you can evaluate generated patient summaries as follows: 

```bash
python utils/eval.py \
    --mode summary \
    --path data/generated/summaries.jsonl \
    --score_types bleu, rouge, bert
```

Evaluating clinical notes is done as follows:

```bash
python utils/eval.py \
    --mode clinical_note \
    --path data/generated/notes.jsonl \
    --score_types all
```


### Acknowledgments

This project is part of the [EPFL MAKE initiative for Generative AI](https://make.epfl.ch/projects/generative-ai). 
