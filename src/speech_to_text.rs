use std::error::Error;

use sherpa_rs::{
    moonshine::{MoonshineConfig, MoonshineRecognizer},
    silero_vad::{SileroVad, SileroVadConfig},
    whisper::{WhisperConfig, WhisperRecognizer},
};

pub struct Vad {
    vad: SileroVad,
    window_size: usize,
    sample_rate: u32,
}

impl Vad {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        let sample_rate = 16000;
        let window_size = 512;
        let vad_config = SileroVadConfig {
            model: "silero_vad.onnx".into(),
            threshold: 0.25,
            min_silence_duration: 0.5,
            min_speech_duration: 0.25,
            max_speech_duration: 5.0,
            sample_rate,
            window_size: window_size as i32,
            ..Default::default()
        };
        let vad = SileroVad::new(vad_config, 10.0)?;
        Ok(Self {
            vad,
            window_size,
            sample_rate,
        })
    }

    pub fn process_audio(&mut self, audio: Vec<f32>) {
        self.vad.accept_waveform(audio);
    }

    pub fn speech_detected(&mut self) -> bool {
        !self.vad.is_empty()
    }

    pub fn speech_segment(&mut self) -> Vec<f32> {
        self.vad.front().samples
    }

    pub fn delete_speech_segment(&mut self) {
        self.vad.pop();
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn window_size(&self) -> usize {
        self.window_size
    }
}

#[allow(unused)]
pub enum SpeechToText {
    Moonshine(MoonshineRecognizer),
    Whisper(WhisperRecognizer),
}

#[allow(unused)]
impl SpeechToText {
    pub fn new_moonshine() -> Result<Self, Box<dyn Error>> {
        // Speech To Text
        let config = MoonshineConfig {
            preprocessor: "./sherpa-onnx-moonshine-tiny-en-int8/preprocess.onnx".into(),
            encoder: "./sherpa-onnx-moonshine-tiny-en-int8/encode.int8.onnx".into(),
            uncached_decoder: "./sherpa-onnx-moonshine-tiny-en-int8/uncached_decode.int8.onnx"
                .into(),
            cached_decoder: "./sherpa-onnx-moonshine-tiny-en-int8/cached_decode.int8.onnx".into(),
            tokens: "./sherpa-onnx-moonshine-tiny-en-int8/tokens.txt".into(),
            num_threads: None,
            ..Default::default()
        };
        let stt = MoonshineRecognizer::new(config)?;
        Ok(SpeechToText::Moonshine(stt))
    }

    pub fn new_whisper(sample_rate: u32) -> Result<Self, Box<dyn Error>> {
        // Speech To Text
        let config = WhisperConfig {
            decoder: "sherpa-onnx-whisper-tiny/tiny-decoder.onnx".into(),
            encoder: "sherpa-onnx-whisper-tiny/tiny-encoder.onnx".into(),
            tokens: "sherpa-onnx-whisper-tiny/tiny-tokens.txt".into(),
            language: "en".into(),
            ..Default::default()
        };
        let stt = WhisperRecognizer::new(config)?;
        Ok(SpeechToText::Whisper(stt))
    }

    pub fn transcribe(&mut self, audio: &[f32]) -> String {
        match self {
            SpeechToText::Moonshine(moonshine_recognizer) => {
                moonshine_recognizer.transcribe(16000, audio).text
            }
            SpeechToText::Whisper(whisper_recognizer) => {
                whisper_recognizer.transcribe(16000, audio).text
            }
        }
    }
}
