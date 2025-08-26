use sherpa_rs::tts::{
    KittenTts, KittenTtsConfig, KokoroTts, KokoroTtsConfig, MatchaTts, MatchaTtsConfig,
};

#[allow(unused)]
pub enum TextToSpeech {
    Matcha {
        tts: MatchaTts,
        voice_id: i32,
        sample_rate: u32,
    },
    Kitten {
        tts: KittenTts,
        voice_id: i32,
        sample_rate: u32,
    },
    Kokoro {
        tts: KokoroTts,
        voice_id: i32,
        sample_rate: u32,
    },
}

#[allow(unused)]
impl TextToSpeech {
    pub fn new_matcha(voice_id: i32) -> Self {
        let config = MatchaTtsConfig {
            acoustic_model: "./matcha-icefall-en_US-ljspeech/model-steps-3.onnx".into(),
            vocoder: "./hifigan_v2.onnx".into(),
            tokens: "./matcha-icefall-en_US-ljspeech/tokens.txt".into(),
            data_dir: "./matcha-icefall-en_US-ljspeech/espeak-ng-data".into(),
            ..Default::default()
        };
        TextToSpeech::Matcha {
            tts: MatchaTts::new(config),
            voice_id,
            sample_rate: 22050,
        }
    }

    pub fn new_kitten(voice_id: i32) -> Self {
        let config = KittenTtsConfig {
            model: "./kitten-nano-en-v0_2-fp16/model.fp16.onnx".to_string(),
            voices: "./kitten-nano-en-v0_2-fp16/voices.bin".into(),
            tokens: "./kitten-nano-en-v0_2-fp16/tokens.txt".into(),
            data_dir: "./kitten-nano-en-v0_2-fp16/espeak-ng-data".into(),
            length_scale: 1.0,
            ..Default::default()
        };
        TextToSpeech::Kitten {
            tts: KittenTts::new(config),
            voice_id,
            sample_rate: 24000,
        }
    }

    pub fn new_kokoro(voice_id: i32) -> Self {
        let config = KokoroTtsConfig {
            model: "./kokoro-en-v0_19/model.onnx".to_string(),
            voices: "./kokoro-en-v0_19/voices.bin".into(),
            tokens: "./kokoro-en-v0_19/tokens.txt".into(),
            data_dir: "./kokoro-en-v0_19/espeak-ng-data".into(),
            length_scale: 1.0,
            ..Default::default()
        };
        TextToSpeech::Kokoro {
            tts: KokoroTts::new(config),
            voice_id,
            sample_rate: 24000,
        }
    }

    pub fn create(&mut self, text: &str) -> Vec<f32> {
        let (result, sr) = match self {
            TextToSpeech::Matcha {
                tts,
                voice_id,
                sample_rate,
            } => (tts.create(text, *voice_id as i32, 1.0), *sample_rate),
            TextToSpeech::Kitten {
                tts,
                voice_id,
                sample_rate,
            } => (tts.create(text, *voice_id as i32, 1.0), *sample_rate),
            TextToSpeech::Kokoro {
                tts,
                voice_id,
                sample_rate,
            } => (tts.create(text, *voice_id as i32, 1.0), *sample_rate),
        };
        if let Ok(audio) = result {
            assert_eq!(
                audio.sample_rate, sr,
                "Model was set up with wrong sample rate!"
            );
            audio.samples
        } else {
            Vec::new()
        }
    }

    pub fn sample_rate(&self) -> u32 {
        match self {
            TextToSpeech::Matcha { sample_rate, .. } => *sample_rate,
            TextToSpeech::Kitten { sample_rate, .. } => *sample_rate,
            TextToSpeech::Kokoro { sample_rate, .. } => *sample_rate,
        }
    }
}
