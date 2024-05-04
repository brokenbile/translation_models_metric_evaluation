import transformers
import datasets
import torch

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = "cuda:0" 
    else:
        device = "cpu"
    print(device)

    from transformers import AutoTokenizer, T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    from transformers import AutoModelForSeq2SeqLM
    model = AutoModelForSeq2SeqLM.from_pretrained("translation_ai_model/.")

    list_of_examples = ["Ah, si je pouvais vivre dans l'eau Le monde serait-il plus beau? ", "Nous pardonneras-tu, ô chère mère?", "L'eau dans son courant fait danser nos vies Et la cité, elle nourrit Ainsi que toi, mon doux amour", "Non, le grand amour ne suffit pas Seul un adieu fleurira", "C'est notre histoire de vie, douce et amère", "Moi, je suis et serai toujours là À voir le monde et sa beauté Et ça ne changera jamais, jamais"]
    
    from transformers import pipeline
    translator = pipeline("translation_fr_to_en", model = model, tokenizer = tokenizer, device="cuda", max_length = 512, repetition_penalty = 1.1)
    
    for text in list_of_examples:
        text_to_translate = "translate French to English: " + text #this assumes that the examples that you want to translate do not contain the prefix to translate
        print(text_to_translate)
        output = translator(text_to_translate)
        for translation in output:
            translated_text = translation['translation_text']
            print("Translated text: " + translated_text)
        print()

    