# Kurtis-E1.1-Qwen2.5-3B-Instruct

Kurtis E1.1 fine-tuned with [flower](https://flower.ai/)

## Eval Results

Evaluation tasks were performed with the [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) on an NVIDIA A40.


### hellaswag

```
lm_eval --model hf --model_args pretrained=ethicalabs/Kurtis-E1.1-Qwen2.5-3B-Instruct  --tasks hellaswag --device cuda:0 --batch_size 8
```

|  Tasks  |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|---------|------:|------|-----:|--------|---|-----:|---|-----:|
|hellaswag|      1|none  |     0|acc     |↑  |0.5555|±  |0.0050|
|         |       |none  |     0|acc_norm|↑  |0.7412|±  |0.0044|

### arc_easy

```
lm_eval --model hf --model_args pretrained=ethicalabs/Kurtis-E1.1-Qwen2.5-3B-Instruct  --tasks arc_easy --device cuda:0 --batch_size 8
```

| Tasks  |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|--------|------:|------|-----:|--------|---|-----:|---|-----:|
|arc_easy|      1|none  |     0|acc     |↑  |0.7710|±  |0.0086|
|        |       |none  |     0|acc_norm|↑  |0.6789|±  |0.0096|


### arc_challenge

```
lm_eval --model hf --model_args pretrained=ethicalabs/Kurtis-E1.1-Qwen2.5-3B-Instruct  --tasks arc_challenge --device cuda:0 --batch_size 8
```

|    Tasks    |Version|Filter|n-shot| Metric |   |Value|   |Stderr|
|-------------|------:|------|-----:|--------|---|----:|---|-----:|
|arc_challenge|      1|none  |     0|acc     |↑  |0.436|±  |0.0145|
|             |       |none  |     0|acc_norm|↑  |0.448|±  |0.0145|

### mmlu

```
lm_eval --model hf --model_args pretrained=ethicalabs/Kurtis-E1.1-Qwen2.5-3B-Instruct  --tasks mmlu --device cuda:0 --batch_size 8
```

|                 Tasks                 |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|---------------------------------------|------:|------|-----:|------|---|-----:|---|-----:|
|mmlu                                   |      2|none  |      |acc   |↑  |0.6522|±  |0.0038|
| - humanities                          |      2|none  |      |acc   |↑  |0.5734|±  |0.0066|
|  - formal_logic                       |      1|none  |     0|acc   |↑  |0.4603|±  |0.0446|
|  - high_school_european_history       |      1|none  |     0|acc   |↑  |0.7939|±  |0.0316|
|  - high_school_us_history             |      1|none  |     0|acc   |↑  |0.8333|±  |0.0262|
|  - high_school_world_history          |      1|none  |     0|acc   |↑  |0.8397|±  |0.0239|
|  - international_law                  |      1|none  |     0|acc   |↑  |0.7769|±  |0.0380|
|  - jurisprudence                      |      1|none  |     0|acc   |↑  |0.7963|±  |0.0389|
|  - logical_fallacies                  |      1|none  |     0|acc   |↑  |0.7975|±  |0.0316|
|  - moral_disputes                     |      1|none  |     0|acc   |↑  |0.6850|±  |0.0250|
|  - moral_scenarios                    |      1|none  |     0|acc   |↑  |0.2905|±  |0.0152|
|  - philosophy                         |      1|none  |     0|acc   |↑  |0.7106|±  |0.0258|
|  - prehistory                         |      1|none  |     0|acc   |↑  |0.7438|±  |0.0243|
|  - professional_law                   |      1|none  |     0|acc   |↑  |0.4759|±  |0.0128|
|  - world_religions                    |      1|none  |     0|acc   |↑  |0.8246|±  |0.0292|
| - other                               |      2|none  |      |acc   |↑  |0.7087|±  |0.0079|
|  - business_ethics                    |      1|none  |     0|acc   |↑  |0.7300|±  |0.0446|
|  - clinical_knowledge                 |      1|none  |     0|acc   |↑  |0.7321|±  |0.0273|
|  - college_medicine                   |      1|none  |     0|acc   |↑  |0.6705|±  |0.0358|
|  - global_facts                       |      1|none  |     0|acc   |↑  |0.3900|±  |0.0490|
|  - human_aging                        |      1|none  |     0|acc   |↑  |0.7130|±  |0.0304|
|  - management                         |      1|none  |     0|acc   |↑  |0.7961|±  |0.0399|
|  - marketing                          |      1|none  |     0|acc   |↑  |0.8803|±  |0.0213|
|  - medical_genetics                   |      1|none  |     0|acc   |↑  |0.7600|±  |0.0429|
|  - miscellaneous                      |      1|none  |     0|acc   |↑  |0.7957|±  |0.0144|
|  - nutrition                          |      1|none  |     0|acc   |↑  |0.7353|±  |0.0253|
|  - professional_accounting            |      1|none  |     0|acc   |↑  |0.5426|±  |0.0297|
|  - professional_medicine              |      1|none  |     0|acc   |↑  |0.6434|±  |0.0291|
|  - virology                           |      1|none  |     0|acc   |↑  |0.4880|±  |0.0389|
| - social sciences                     |      2|none  |      |acc   |↑  |0.7618|±  |0.0076|
|  - econometrics                       |      1|none  |     0|acc   |↑  |0.5439|±  |0.0469|
|  - high_school_geography              |      1|none  |     0|acc   |↑  |0.7677|±  |0.0301|
|  - high_school_government_and_politics|      1|none  |     0|acc   |↑  |0.8860|±  |0.0229|
|  - high_school_macroeconomics         |      1|none  |     0|acc   |↑  |0.6949|±  |0.0233|
|  - high_school_microeconomics         |      1|none  |     0|acc   |↑  |0.7773|±  |0.0270|
|  - high_school_psychology             |      1|none  |     0|acc   |↑  |0.8477|±  |0.0154|
|  - human_sexuality                    |      1|none  |     0|acc   |↑  |0.7786|±  |0.0364|
|  - professional_psychology            |      1|none  |     0|acc   |↑  |0.7075|±  |0.0184|
|  - public_relations                   |      1|none  |     0|acc   |↑  |0.6818|±  |0.0446|
|  - security_studies                   |      1|none  |     0|acc   |↑  |0.7224|±  |0.0287|
|  - sociology                          |      1|none  |     0|acc   |↑  |0.8458|±  |0.0255|
|  - us_foreign_policy                  |      1|none  |     0|acc   |↑  |0.8400|±  |0.0368|
| - stem                                |      2|none  |      |acc   |↑  |0.6070|±  |0.0085|
|  - abstract_algebra                   |      1|none  |     0|acc   |↑  |0.4700|±  |0.0502|
|  - anatomy                            |      1|none  |     0|acc   |↑  |0.6667|±  |0.0407|
|  - astronomy                          |      1|none  |     0|acc   |↑  |0.6776|±  |0.0380|
|  - college_biology                    |      1|none  |     0|acc   |↑  |0.7222|±  |0.0375|
|  - college_chemistry                  |      1|none  |     0|acc   |↑  |0.5000|±  |0.0503|
|  - college_computer_science           |      1|none  |     0|acc   |↑  |0.6000|±  |0.0492|
|  - college_mathematics                |      1|none  |     0|acc   |↑  |0.3400|±  |0.0476|
|  - college_physics                    |      1|none  |     0|acc   |↑  |0.4902|±  |0.0497|
|  - computer_security                  |      1|none  |     0|acc   |↑  |0.7000|±  |0.0461|
|  - conceptual_physics                 |      1|none  |     0|acc   |↑  |0.6468|±  |0.0312|
|  - electrical_engineering             |      1|none  |     0|acc   |↑  |0.6690|±  |0.0392|
|  - elementary_mathematics             |      1|none  |     0|acc   |↑  |0.5979|±  |0.0253|
|  - high_school_biology                |      1|none  |     0|acc   |↑  |0.8129|±  |0.0222|
|  - high_school_chemistry              |      1|none  |     0|acc   |↑  |0.5813|±  |0.0347|
|  - high_school_computer_science       |      1|none  |     0|acc   |↑  |0.7800|±  |0.0416|
|  - high_school_mathematics            |      1|none  |     0|acc   |↑  |0.5037|±  |0.0305|
|  - high_school_physics                |      1|none  |     0|acc   |↑  |0.4437|±  |0.0406|
|  - high_school_statistics             |      1|none  |     0|acc   |↑  |0.5972|±  |0.0334|
|  - machine_learning                   |      1|none  |     0|acc   |↑  |0.4554|±  |0.0473|

|      Groups      |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|------------------|------:|------|------|------|---|-----:|---|-----:|
|mmlu              |      2|none  |      |acc   |↑  |0.6522|±  |0.0038|
| - humanities     |      2|none  |      |acc   |↑  |0.5734|±  |0.0066|
| - other          |      2|none  |      |acc   |↑  |0.7087|±  |0.0079|
| - social sciences|      2|none  |      |acc   |↑  |0.7618|±  |0.0076|
| - stem           |      2|none  |      |acc   |↑  |0.6070|±  |0.0085|


### mmlu (5-shot)

```
lm_eval   --model hf   --model_args pretrained=ethicalabs/Kurtis-E1.1-Qwen2.5-3B-Instruct   --tasks mmlu   --device cuda:0   --batch_size 8 --num_fewshot 5
```

|                 Tasks                 |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|---------------------------------------|------:|------|-----:|------|---|-----:|---|-----:|
|mmlu                                   |      2|none  |      |acc   |↑  |0.6629|±  |0.0038|
| - humanities                          |      2|none  |      |acc   |↑  |0.5862|±  |0.0067|
|  - formal_logic                       |      1|none  |     5|acc   |↑  |0.4683|±  |0.0446|
|  - high_school_european_history       |      1|none  |     5|acc   |↑  |0.7818|±  |0.0323|
|  - high_school_us_history             |      1|none  |     5|acc   |↑  |0.8284|±  |0.0265|
|  - high_school_world_history          |      1|none  |     5|acc   |↑  |0.8692|±  |0.0219|
|  - international_law                  |      1|none  |     5|acc   |↑  |0.7769|±  |0.0380|
|  - jurisprudence                      |      1|none  |     5|acc   |↑  |0.7963|±  |0.0389|
|  - logical_fallacies                  |      1|none  |     5|acc   |↑  |0.8098|±  |0.0308|
|  - moral_disputes                     |      1|none  |     5|acc   |↑  |0.7110|±  |0.0244|
|  - moral_scenarios                    |      1|none  |     5|acc   |↑  |0.3464|±  |0.0159|
|  - philosophy                         |      1|none  |     5|acc   |↑  |0.7042|±  |0.0259|
|  - prehistory                         |      1|none  |     5|acc   |↑  |0.7284|±  |0.0247|
|  - professional_law                   |      1|none  |     5|acc   |↑  |0.4759|±  |0.0128|
|  - world_religions                    |      1|none  |     5|acc   |↑  |0.8304|±  |0.0288|
| - other                               |      2|none  |      |acc   |↑  |0.7171|±  |0.0078|
|  - business_ethics                    |      1|none  |     5|acc   |↑  |0.7400|±  |0.0441|
|  - clinical_knowledge                 |      1|none  |     5|acc   |↑  |0.7321|±  |0.0273|
|  - college_medicine                   |      1|none  |     5|acc   |↑  |0.6647|±  |0.0360|
|  - global_facts                       |      1|none  |     5|acc   |↑  |0.4100|±  |0.0494|
|  - human_aging                        |      1|none  |     5|acc   |↑  |0.7220|±  |0.0301|
|  - management                         |      1|none  |     5|acc   |↑  |0.7864|±  |0.0406|
|  - marketing                          |      1|none  |     5|acc   |↑  |0.8889|±  |0.0206|
|  - medical_genetics                   |      1|none  |     5|acc   |↑  |0.7900|±  |0.0409|
|  - miscellaneous                      |      1|none  |     5|acc   |↑  |0.7957|±  |0.0144|
|  - nutrition                          |      1|none  |     5|acc   |↑  |0.7680|±  |0.0242|
|  - professional_accounting            |      1|none  |     5|acc   |↑  |0.5532|±  |0.0297|
|  - professional_medicine              |      1|none  |     5|acc   |↑  |0.6471|±  |0.0290|
|  - virology                           |      1|none  |     5|acc   |↑  |0.5120|±  |0.0389|
| - social sciences                     |      2|none  |      |acc   |↑  |0.7735|±  |0.0075|
|  - econometrics                       |      1|none  |     5|acc   |↑  |0.5877|±  |0.0463|
|  - high_school_geography              |      1|none  |     5|acc   |↑  |0.7828|±  |0.0294|
|  - high_school_government_and_politics|      1|none  |     5|acc   |↑  |0.8756|±  |0.0238|
|  - high_school_macroeconomics         |      1|none  |     5|acc   |↑  |0.7051|±  |0.0231|
|  - high_school_microeconomics         |      1|none  |     5|acc   |↑  |0.7773|±  |0.0270|
|  - high_school_psychology             |      1|none  |     5|acc   |↑  |0.8550|±  |0.0151|
|  - human_sexuality                    |      1|none  |     5|acc   |↑  |0.8092|±  |0.0345|
|  - professional_psychology            |      1|none  |     5|acc   |↑  |0.7288|±  |0.0180|
|  - public_relations                   |      1|none  |     5|acc   |↑  |0.6909|±  |0.0443|
|  - security_studies                   |      1|none  |     5|acc   |↑  |0.7551|±  |0.0275|
|  - sociology                          |      1|none  |     5|acc   |↑  |0.8308|±  |0.0265|
|  - us_foreign_policy                  |      1|none  |     5|acc   |↑  |0.8300|±  |0.0378|
| - stem                                |      2|none  |      |acc   |↑  |0.6159|±  |0.0084|
|  - abstract_algebra                   |      1|none  |     5|acc   |↑  |0.5000|±  |0.0503|
|  - anatomy                            |      1|none  |     5|acc   |↑  |0.6222|±  |0.0419|
|  - astronomy                          |      1|none  |     5|acc   |↑  |0.7500|±  |0.0352|
|  - college_biology                    |      1|none  |     5|acc   |↑  |0.7083|±  |0.0380|
|  - college_chemistry                  |      1|none  |     5|acc   |↑  |0.4700|±  |0.0502|
|  - college_computer_science           |      1|none  |     5|acc   |↑  |0.6200|±  |0.0488|
|  - college_mathematics                |      1|none  |     5|acc   |↑  |0.4000|±  |0.0492|
|  - college_physics                    |      1|none  |     5|acc   |↑  |0.4902|±  |0.0497|
|  - computer_security                  |      1|none  |     5|acc   |↑  |0.8200|±  |0.0386|
|  - conceptual_physics                 |      1|none  |     5|acc   |↑  |0.6383|±  |0.0314|
|  - electrical_engineering             |      1|none  |     5|acc   |↑  |0.6483|±  |0.0398|
|  - elementary_mathematics             |      1|none  |     5|acc   |↑  |0.5820|±  |0.0254|
|  - high_school_biology                |      1|none  |     5|acc   |↑  |0.8161|±  |0.0220|
|  - high_school_chemistry              |      1|none  |     5|acc   |↑  |0.6059|±  |0.0344|
|  - high_school_computer_science       |      1|none  |     5|acc   |↑  |0.7500|±  |0.0435|
|  - high_school_mathematics            |      1|none  |     5|acc   |↑  |0.4926|±  |0.0305|
|  - high_school_physics                |      1|none  |     5|acc   |↑  |0.4702|±  |0.0408|
|  - high_school_statistics             |      1|none  |     5|acc   |↑  |0.6343|±  |0.0328|
|  - machine_learning                   |      1|none  |     5|acc   |↑  |0.4911|±  |0.0475|

|      Groups      |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|------------------|------:|------|------|------|---|-----:|---|-----:|
|mmlu              |      2|none  |      |acc   |↑  |0.6629|±  |0.0038|
| - humanities     |      2|none  |      |acc   |↑  |0.5862|±  |0.0067|
| - other          |      2|none  |      |acc   |↑  |0.7171|±  |0.0078|
| - social sciences|      2|none  |      |acc   |↑  |0.7735|±  |0.0075|
| - stem           |      2|none  |      |acc   |↑  |0.6159|±  |0.0084|
