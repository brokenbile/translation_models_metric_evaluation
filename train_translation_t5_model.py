import transformers
import datasets
import torch

if __name__ == '__main__':
    from datasets import load_dataset
    dataset = load_dataset('csv', data_files = 'en-fr-train.csv')
    dataset = dataset['train'].train_test_split(test_size = 0.1)
    print(dataset['train'][1])

    from transformers import AutoTokenizer, T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    print(device)


    source_lang = "fr"
    target_lang = "en"
    prefix = "translate French to English:"

    def preprocess_function(example):
        inputs = prefix + str(example[source_lang])
        targets = str(example[target_lang])
        model_inputs = tokenizer(inputs, text_target=targets, max_length=1000, truncation=True, padding = True)
        return model_inputs

    tokenized_translations = dataset.map(preprocess_function)

    print(tokenized_translations)
    print(tokenized_translations['train'][1])

    #tokenized_translations = tokenized_translations.remove_columns(dataset['train'].column_names)

    from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
    from transformers import AutoModelForCausalLM, T5ForConditionalGeneration
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

    from transformers import DataCollatorForSeq2Seq
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    from transformers import TrainingArguments, Trainer
    
    training_args = TrainingArguments(
        output_dir = "t5-small-model",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        weight_decay=0.02,
        save_total_limit=3,
        num_train_epochs=20,
        per_device_train_batch_size=4, 
        gradient_accumulation_steps=4,
        fp16=True,
        save_strategy = "no",
    )
    #train_batch_size and gradient_accumulation_steps were set to 4 to allow for the model to successfully train without running out of memory

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset = tokenized_translations["train"],
        eval_dataset = tokenized_translations["test"],
        tokenizer = tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    from transformers import pipeline

    #if you want to save the model to a local directory on your computer
    trainer.save_model("translation_ai_model")

    list_of_examples = ["Ah, si je pouvais vivre dans l'eau Le monde serait-il plus beau? ", "Nous pardonneras-tu, ô chère mère?", "L'eau dans son courant fait danser nos vies Et la cité, elle nourrit Ainsi que toi, mon doux amour", "Non, le grand amour ne suffit pas Seul un adieu fleurira", "C'est notre histoire de vie, douce et amère", "Moi, je suis et serai toujours là À voir le monde et sa beauté Et ça ne changera jamais, jamais"]

    for text in list_of_examples:
        text_to_translate = "translate French to English: " + text #this assumes that the examples that you want to translate do not contain the prefix to translate
        print(text_to_translate)
        output = translator(text_to_translate)
        for translation in output:
            translated_text = translation['translation_text']
            print("Translated text: " + translated_text)
        print()