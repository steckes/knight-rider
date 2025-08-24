use std::error::Error;

use sherpa_rs::{
    moonshine::{MoonshineConfig, MoonshineRecognizer},
    silero_vad::{SileroVad, SileroVadConfig},
};

pub struct Vad {
    vad: SileroVad,
    window_size: usize,
}

impl Vad {
    pub fn new(sample_rate: u32) -> Result<Self, Box<dyn Error>> {
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
        Ok(Self { vad, window_size })
    }

    pub fn window_size(&self) -> usize {
        self.window_size
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
}

pub struct SpeechToText {
    stt: MoonshineRecognizer,
    sample_rate: u32,
}

impl SpeechToText {
    pub fn new(sample_rate: u32) -> Result<Self, Box<dyn Error>> {
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
        Ok(Self { stt, sample_rate })
    }

    pub fn transcribe(&mut self, audio: &[f32]) -> String {
        self.stt.transcribe(self.sample_rate, audio).text
    }
}
