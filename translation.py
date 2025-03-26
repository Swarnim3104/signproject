from googletrans import Translator

def translate_file(input_file, output_file, target_language):
    translator = Translator()

    # Read the text from the file
    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()

    # Translate the text
    translation = translator.translate(text, dest=target_language)

    # Write the translated text to a new file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(translation.text)

    print(f"Translation completed! Saved to {output_file}")

# Example usage:
translate_file("output.txt", "output_translated.txt", "hi")  # Translates to Hindi
