import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline

model_id = "/pure-mlo-scratch/make_project/trial-runs/meditron-7b-summarizer/hf_checkpoint/"

# load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_id, use_cache=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id, use_cache=False)

# Set the prompt
prompt = "Given the provided patient-doctor dialogue, write the corresponding patient information summary in JSON format.\nMake sure to extract all the information from the dialogue into the template, but do not add any new information. \nIf a field is not mentioned, simply write \"feature\": \"None\".\n\nDoctor: Good morning, sir. How are you feeling today?\nPatient: Not so good, doctor. I'm in a lot of pain.\nDoctor: I understand. Can you tell me what happened to you?\nPatient: I was rescued after being jammed between an excavator basket and a wall with some stacked material for 5 minutes.\nDoctor: I see. When you arrived at the emergency room, what were your vital signs like?\nPatient: I think my blood pressure was 121\/71 mmHg, heart rate was 90 beats per minute, respiratory rate was 30 breaths per minute, body temperature was 36.9\u00b0C, and O2 saturation was 94%.\nDoctor: Okay. And what did you complain of when you arrived?\nPatient: I complained of severe pain on the whole left side of my body down to the waist and on the right chest wall.\nDoctor: I see. We performed a plain radiography and a computed tomography on you in the emergency room. The results showed that you have multiple rib fractures on your right 7th and 8th ribs and your left 3rd to 12th ribs, as well as a hemothorax on both sides with the left side being dominant, a left scapular fracture, a liver laceration, a retro-peritoneal hematoma, and a transverse process fracture of your thoracic and lumbar spine.\nPatient: Hmm. That's a lot of injuries.\nDoctor: Yes, it is. You were initially admitted to the intensive care unit after a closed thoracostomy. There was no special complication during the chest tube insertion, and the initial drainage was approximately 200 mL.\nPatient: Okay.\nDoctor: There was bloody drainage during the chest tube insertion, but it became serous and you were transferred to the general ward the next day. On the 4th post-trauma day, you complained of acute pain radiating in the left rear direction. We found about 400 mL of bloody drainage at the same time.\nPatient: Yes, I remember that.\nDoctor: When you complained of pain, you became drowsy as your blood pressure decreased to 70\/50 mmHg. However, your blood pressure became 100\/70 mmHg after the infusion of 300 mL of isotonic saline. Although your vital signs became stable with a heart rate of 71 beats per minute, respiratory rate of 24 breaths per minute, body temperature of 36.6\u00b0C, and O2 saturation of 99%, bloody drainage of 100 mL or more hourly continued.\nPatient: That's concerning.\nDoctor: Yes, it is. We will need to monitor your condition closely and continue to provide you with appropriate treatment.\nPatient: Okay.\nDoctor: Is there anything else you would like to ask me, sir?\nPatient: No, I think that's all for now.\nDoctor: Alright, please let me know if you have any questions or concerns in the future.\nPatient: I will, thank you, doctor."

# use a pipeline to generate text
pipe = pipeline("text-generation",
                model=model,
                tokenizer= tokenizer,    
                max_new_tokens = 200,
                do_sample=True,
                top_k=10,
                num_return_sequences=2,
                eos_token_id=tokenizer.eos_token_id,
                return_full_text=False
                )

# generate from prompt
generated = pipe(prompt)
print(generated['generated_text'])