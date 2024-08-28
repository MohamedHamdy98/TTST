# Translation

from googletrans import Translator, LANGUAGES

translator = Translator()

def translate_text(text, src_lang, dest_lang):
    try:
        # Translate the text
        translated = translator.translate(text, src=src_lang, dest=dest_lang)
        return translated.text
    except Exception as e:
        return f"Error: {str(e)}"

# Get user input for the text, source language, and destination language
with open('mic_to_text.txt', 'r') as f:
    text = f.read()

src_lang = input("Enter the source language code: ")
dest_lang = input("Enter the destination language code: ")

# Translate the text
translated_text = translate_text(text, src_lang, dest_lang)
print(f"\nTranslated text: {translated_text}")

with open('text_after_translation.txt', 'w') as f:
    f.write(translated_text)





