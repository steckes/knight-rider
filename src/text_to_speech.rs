use sherpa_rs::tts::{KittenTts, KittenTtsConfig, MatchaTts, MatchaTtsConfig};

#[allow(unused)]
pub enum TextToSpeech {
    Matcha(MatchaTts, u32),
    Kitten(KittenTts, u32, u8),
}

#[allow(unused)]
impl TextToSpeech {
    pub fn new_matcha(sample_rate: u32) -> Self {
        let config = MatchaTtsConfig {
            acoustic_model: "./matcha-icefall-en_US-ljspeech/model-steps-3.onnx".into(),
            vocoder: "./hifigan_v2.onnx".into(),
            tokens: "./matcha-icefall-en_US-ljspeech/tokens.txt".into(),
            data_dir: "./matcha-icefall-en_US-ljspeech/espeak-ng-data".into(),
            ..Default::default()
        };
        TextToSpeech::Matcha(MatchaTts::new(config), sample_rate)
    }

    pub fn new_kitten(sample_rate: u32, voice: u8) -> Self {
        let config = KittenTtsConfig {
            model: "./kitten-nano-en-v0_2-fp16/model.fp16.onnx".to_string(),
            voices: "./kitten-nano-en-v0_2-fp16/voices.bin".into(),
            tokens: "./kitten-nano-en-v0_2-fp16/tokens.txt".into(),
            data_dir: "./kitten-nano-en-v0_2-fp16/espeak-ng-data".into(),
            length_scale: 1.0,
            ..Default::default()
        };
        TextToSpeech::Kitten(KittenTts::new(config), sample_rate, voice)
    }

    pub fn create(&mut self, text: &str) -> Vec<f32> {
        let result = match self {
            TextToSpeech::Matcha(matcha_tts, sr) => (matcha_tts.create(text, 0, 1.0), *sr),
            TextToSpeech::Kitten(kitten_tts, sr, voice) => {
                (kitten_tts.create(text, *voice as i32, 1.0), *sr)
            }
        };
        if let (Ok(audio), sr) = result {
            assert_eq!(audio.sample_rate, sr, "Wrong `tts_sample_rate` configured");
            audio.samples
        } else {
            Vec::new()
        }
    }
}
