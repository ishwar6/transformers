# why? : to enhance their performance (LLMs) on specific tasks by adapting them to domain-specific data. 

# Fine-tuning LLM involves the additional training of a pre-existing model, which has previously acquired patterns and features from an extensive dataset, using a smaller, domain-specific dataset.
# Utilizing the existing knowledge embedded in the pre-trained model allows for achieving high performance on specific tasks with substantially reduced data and computational requirements.


# key steps involved in LLM Fine-tuning:

# Select a pre-trained model: For LLM Fine-tuning first step is to carefully select a base pre-trained model that aligns with our desired architecture and functionalities. 

# Gather relevant Dataset: Then we need to gather a dataset that is relevant to our task. The dataset should be labeled or structured in a way that the model can learn from it.

# Preprocess Dataset: Once the dataset is ready, we need to do some preprocessing for fine-tuning by cleaning it, splitting it into training, validation, and test sets, and ensuring it’s compatible with the model on which we want to fine-tune.

# Fine-tuning: After selecting a pre-trained model we need to fine tune it on our preprocessed relevant dataset which is more specific to the task at hand. 
The dataset which we will select might be related to a particular domain or application, allowing the model to adapt and specialize for that context.

# Task-specific adaptation: During fine-tuning, the model’s parameters are adjusted based on the new dataset, helping it better understand and generate content relevant to the specific task. 
This process retains the general language knowledge gained during pre-training while tailoring the model to the nuances of the target domain.


# #Fine-tuning methods: 
#   1. Full Fine Tuning (Instruction fine-tuning) : updates all model weights, creating a new version with improved capabilities. However, it demands sufficient memory and computational resources, similar to pre-training,
#   2. Parameter Efficient Fine-Tuning (PEFT) is a form of instruction fine-tuning that is much more efficient than full fine-tuning. PEFT addresses this by updating only a subset of parameters, effectively “freezing” the rest. 
      # This reduces the number of trainable parameters, making memory requirements more manageable and preventing catastrophic forgetting.
# There are various ways of achieving Parameter efficient fine-tuning. Low-Rank Adaptation LoRA & QLoRA are the most widely used and effective.

# LoRA: Low-Rank Adaptation of Large Language Models
# Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen
