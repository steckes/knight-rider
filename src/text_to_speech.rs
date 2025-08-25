use sherpa_rs::tts::{
    KittenTts, KittenTtsConfig, KokoroTts, KokoroTtsConfig, MatchaTts, MatchaTtsConfig,
};

#[allow(unused)]
pub enum TextToSpeech {
    Matcha(MatchaTts, u32, u8),
    Kitten(KittenTts, u32, u8),
    Kokoro(KokoroTts, u32, u8),
}

#[allow(unused)]
impl TextToSpeech {
    pub fn new_matcha(sample_rate: u32, voice: u8) -> Self {
        let config = MatchaTtsConfig {
            acoustic_model: "./matcha-icefall-en_US-ljspeech/model-steps-3.onnx".into(),
            vocoder: "./hifigan_v2.onnx".into(),
            tokens: "./matcha-icefall-en_US-ljspeech/tokens.txt".into(),
            data_dir: "./matcha-icefall-en_US-ljspeech/espeak-ng-data".into(),
            ..Default::default()
        };
        TextToSpeech::Matcha(MatchaTts::new(config), sample_rate, voice)
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

    pub fn new_kokoro(sample_rate: u32, voice: u8) -> Self {
        let config = KokoroTtsConfig {
            model: "./kokoro-multi-lang-v1_0/model.onnx".to_string(),
            voices: "./kokoro-multi-lang-v1_0/voices.bin".into(),
            tokens: "./kokoro-multi-lang-v1_0/tokens.txt".into(),
            data_dir: "./kokoro-multi-lang-v1_0/espeak-ng-data".into(),
            dict_dir: "./kokoro-multi-lang-v1_0/dict".into(),
            length_scale: 1.0,
            ..Default::default()
        };
        TextToSpeech::Kokoro(KokoroTts::new(config), sample_rate, voice)
    }

    pub fn create(&mut self, text: &str) -> Vec<f32> {
        let result = match self {
            TextToSpeech::Matcha(matcha_tts, sr, voice) => {
                (matcha_tts.create(text, *voice as i32, 1.0), *sr)
            }
            TextToSpeech::Kitten(kitten_tts, sr, voice) => {
                (kitten_tts.create(text, *voice as i32, 1.0), *sr)
            }
            TextToSpeech::Kokoro(kokoro_tts, sr, voice) => {
                (kokoro_tts.create(text, *voice as i32, 1.0), *sr)
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
