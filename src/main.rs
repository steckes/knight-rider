use std::error::Error;

use llama::BlockingLlama;

use crate::{
    speech_to_text::{SpeechToText, Vad},
    system_audio::{AudioConfig, SystemAudio},
    text_to_speech::TextToSpeech,
};

mod llama;
mod speech_to_text;
mod system_audio;
mod text_to_speech;

fn main() -> Result<(), Box<dyn Error>> {
    // If you want to select your device type, comment out the following
    system_audio::list_device_names();

    // Choose which models you want to use
    let mut vad = Vad::new()?;
    let mut stt = SpeechToText::new_moonshine()?;
    let mut tts = TextToSpeech::new_matcha(0);

    let audio_config = AudioConfig {
        input_device: None,  // None = default
        output_device: None, // None = default
        system_sample_rate: 48000,
        num_frames: 512,
        vad_sample_rate: vad.sample_rate(),
        tts_sample_rate: tts.sample_rate(),
    };

    let mut system_audio = SystemAudio::new(audio_config)?;

    // Start Llama Client
    let mut llama = match BlockingLlama::new() {
        Ok(llama) => llama,
        Err(_) => {
            eprintln!("Make sure `llama-server` is running and start again!");
            return Ok(());
        }
    };

    // Say something so we know the system is ready
    println!("K.I.T.T. is ready for your requests..");
    let generated_speech = tts.create("All systems ready!");
    system_audio.send_audio(&generated_speech);

    // Main AI Loop
    loop {
        if system_audio.num_samples_available() >= vad.window_size() {
            let input_audio = system_audio.receive_audio(vad.window_size());

            vad.process_audio(input_audio);

            while vad.speech_detected() {
                // do not accept new speech input while processing
                system_audio.set_ready_to_receive(false);

                let transcript = stt.transcribe(&vad.speech_segment());

                if !transcript.is_empty() {
                    println!("User: {}", &transcript);

                    let mut answer = match llama.chat(&transcript) {
                        Ok(anwser) => anwser,
                        Err(_) => {
                            eprintln!("Error: Llama failed to produce an answer...");
                            "Oh no, I could not produce an answer, there must be an issue with the connection."
                                .to_string()
                        }
                    };

                    println!("KITT: {}", &answer);

                    // Limit the prompt so it does not take too long to generate the speech
                    if answer.len() > 200 {
                        println!("Answer too long, cutting off...");
                        answer = answer[0..200].to_string();
                    }

                    // Remove symbols that should not be said
                    answer = answer
                        .chars()
                        .filter(|c| {
                            c.is_alphanumeric()
                                || c.is_whitespace()
                                || matches!(c, '.' | '?' | '!' | ',' | ':' | ';')
                        })
                        .collect();
                    let generated_speech = tts.create(&answer);

                    system_audio.send_audio(&generated_speech);
                }
                vad.delete_speech_segment();
            }
            // now new input speech can be accepted
            system_audio.set_ready_to_receive(true);
        }
    }
}
