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
    // system_audio::list_device_names();

    let vad_sample_rate = 16000;
    let tts_sample_rate = 22050;
    let audio_config = AudioConfig {
        input_device: None,  // None = default
        output_device: None, // None = default
        system_sample_rate: 48000,
        vad_sample_rate,
        generated_speech_sample_rate: tts_sample_rate,
        num_frames: 512,
    };

    let mut system_audio = SystemAudio::new(audio_config)?;

    let mut vad = Vad::new(vad_sample_rate)?;
    let mut stt = SpeechToText::new_moonshine(vad_sample_rate)?;
    let mut tts = TextToSpeech::new_matcha(tts_sample_rate, 0);

    // LlamaClient to talk to LlamaServer
    let mut llama = match BlockingLlama::new() {
        Ok(llama) => llama,
        Err(_) => {
            eprintln!("Make sure `llama-server` is running and start again!");
            return Ok(());
        }
    };

    println!("K.I.T.T. is ready for your requests..");

    // AI Loop
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

                    let answer = match llama.chat(&transcript) {
                        Ok(anwser) => anwser,
                        Err(_) => {
                            eprintln!("Error: Llama failed to produce an answer...");
                            "Oh no, I could not produce an answer, there must be an issue with the connection."
                                .to_string()
                        }
                    };

                    println!("KITT: {}", &answer);

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
