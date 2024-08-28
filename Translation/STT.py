#Speech To Text

import speech_recognition as sr


def mic_to_text(recognizer, language):
    with sr.Microphone() as source:
        print('noises: ')
        recognizer.adjust_for_ambient_noise(source, duration=5)
        print("speak now: ")
        try:
            audio = recognizer.listen(source)
            print("processing.....wait")
            speech_text = recognizer.recognize_google(audio, language=language)
            print("text:" + speech_text)

            with open('mic_to_text.txt', "w") as file:
                file.write(speech_text)
        except sr.WaitTimeoutError:
            print('no audio coming from mic')
        except sr.UnknownValueError:
            print("google does not understand the input audio langauage")
        except sr.RequestError as e:
            print('request failed to google service {e}')
        except Exception as e:
            print(f'Error {e}')



import speech_recognition as sr
recognizer = sr.Recognizer()

def file_to_text(recognizer, language, fileName):
    try:
        with sr.AudioFile(fileName) as source:
            # Record the audio from the file
            audio = recognizer.record(source)
            # print("Processing... Please wait.")
            
            # Recognize speech using Google Speech Recognition
            speech_text = recognizer.recognize_google(audio, language=language)
            # print("Text: " + speech_text)
            
            output_file = 'speech_to_text_file.txt'
            # Write the recognized text to a file
            with open(output_file, "w", encoding='utf-8') as file:
                file.write(speech_text)
            # print("Text successfully written to mic_to_text.txt")
            return speech_text

    except sr.UnknownValueError:
        print("Google could not understand the input audio language.")
        return None
    except sr.RequestError as e:
        print(f'Request failed to Google service: {e}')
        return None
    except Exception as e:
        print(f'An error occurred: {e}')
        return None

# t = file_to_text(recognizer, 'en', r'H:\avatar_veem\Coding\xtts2-hf-main\xtts2-hf-main\Translation\result.wav')
# with open(t, 'r') as f:
#     prompt = f.read()
# print(prompt)


