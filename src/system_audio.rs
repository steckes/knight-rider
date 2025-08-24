use std::sync::{atomic::AtomicBool, Arc};

use ringbuf::traits::{Consumer as _, Observer};
use ringbuf::traits::{Producer as _, Split};
use ringbuf::{storage::Heap, wrap::caching::Caching, HeapRb, SharedRb};
use rtaudio::{
    Api, Buffers, DeviceParams, RtAudioError, SampleFormat, StreamHandle, StreamInfo,
    StreamOptions, StreamStatus,
};
use rubato::{FftFixedIn, FftFixedOut, ResampleError, Resampler, ResamplerConstructionError};

type Producer = Caching<Arc<SharedRb<Heap<f32>>>, true, false>;
type Consumer = Caching<Arc<SharedRb<Heap<f32>>>, false, true>;

#[derive(thiserror::Error, Debug)]
pub enum SystemAudioError {
    #[error("RtAudio error: {0}")]
    RtAudio(#[from] RtAudioError),
    #[error("Output Device not found")]
    OutputDeviceNotFound,
    #[error("Output Device not found")]
    InputDeviceNotFound,
    #[error("Resample construction error: {0}")]
    ResamplerConstruction(#[from] ResamplerConstructionError),
    #[error("Resample error: {0}")]
    Resample(#[from] ResampleError),
}

pub struct AudioConfig {
    pub output_device: Option<String>,
    pub input_device: Option<String>,
    pub system_sample_rate: u32,
    pub vad_sample_rate: u32,
    pub generated_speech_sample_rate: u32,
    pub num_frames: usize,
}

pub struct SystemAudio {
    stream_handle: StreamHandle,
    input_consumer: Consumer,
    output_producer: Producer,
    ready_to_receive: Arc<AtomicBool>,
}

impl SystemAudio {
    pub fn new(config: AudioConfig) -> Result<Self, SystemAudioError> {
        // The system host (alsa, coreaudio, wasapi, ...)
        let host = rtaudio::Host::new(Api::Unspecified)?;

        // Search the wanted output device or take the default
        let output_device = if let Some(device_name) = &config.output_device {
            let output_device = host
                .iter_output_devices()
                .find(|d| d.name == device_name.as_str());
            output_device.ok_or(SystemAudioError::OutputDeviceNotFound)?
        } else {
            host.default_output_device()?
        };

        // Search the wanted input device or take the default
        let input_device = if let Some(device_name) = &config.input_device {
            let input_device = host
                .iter_input_devices()
                .find(|d| d.name == device_name.as_str());
            input_device.ok_or(SystemAudioError::InputDeviceNotFound)?
        } else {
            host.default_input_device()?
        };

        // Construct the resampler that resamples the audio input to the VAD sample rate
        let mut input_resampler = FftFixedIn::new(
            config.system_sample_rate as usize,
            config.vad_sample_rate as usize,
            config.num_frames,
            1,
            1,
        )?;
        let mut resampled_input = input_resampler.output_buffer_allocate(true);

        // Construct the resampler that resamples the generated speech to the system sample rate
        let mut output_resampler = FftFixedOut::new(
            config.generated_speech_sample_rate as usize,
            config.system_sample_rate as usize,
            config.num_frames,
            1,
            1,
        )?;
        let mut output_speech = output_resampler.input_buffer_allocate(true);

        // The ringbuffer that sends the audio from the system to the background AI process
        let rb = HeapRb::<f32>::new(600 * config.num_frames);
        let (mut input_producer, input_consumer) = rb.split();

        // The ringbuffer that sends the audio from the background ai process back to the system
        let rb = HeapRb::<f32>::new(600 * config.num_frames);
        let (output_producer, mut output_consumer) = rb.split();

        let mut stream_handle = host
            .open_stream(
                Some(DeviceParams {
                    device_id: output_device.id,
                    num_channels: 1,
                    first_channel: 0,
                }),
                Some(DeviceParams {
                    device_id: input_device.id,
                    num_channels: 1,
                    first_channel: 0,
                }),
                SampleFormat::Float32,
                config.system_sample_rate,
                config.num_frames as u32,
                StreamOptions::default(),
                |error| eprintln!("Error in RtAudio Stream: {}", error),
            )
            .map_err(|e| e.1)?;

        // A variable so the ai process can indicate if it is ready to receive more input
        let ready_to_receive = Arc::new(AtomicBool::new(true));

        let ready_to_receive_clone = ready_to_receive.clone();

        stream_handle.start(
            move |buffers: Buffers<'_>, _info: &StreamInfo, _status: StreamStatus| {
                if let Buffers::Float32 { output, input } = buffers {
                    let available_samples = output_consumer.occupied_len();

                    let ready_to_receive =
                        ready_to_receive_clone.load(std::sync::atomic::Ordering::Relaxed);

                    // either KITT or you are speaking
                    if available_samples > 0 {
                        // receive KITTs voice
                        let resampler_input_frames_next =
                            available_samples.min(output_resampler.input_frames_next());
                        let num_popped = output_consumer
                            .pop_slice(&mut output_speech[0][..resampler_input_frames_next]);
                        // pad with 0 if less samples were received
                        if num_popped < output_speech[0].len() {
                            output_speech[0][num_popped..].fill(0.0);
                        }
                        if output_resampler
                            .process_into_buffer(&output_speech, &mut [output], None)
                            .is_err()
                        {
                            eprintln!("Output resampling did not suceed, output nothing.")
                        }
                    } else if ready_to_receive {
                        output.fill(0.0);
                        // send your voice
                        let result = input_resampler.process_into_buffer(
                            &[&input],
                            &mut resampled_input,
                            None,
                        );
                        if let Ok((_, num_samples_generated)) = result {
                            input_producer.push_slice(&resampled_input[0][..num_samples_generated]);
                        } else {
                            eprintln!("Input resampling did not suceed, send nothing to KITT.")
                        }
                    }
                }
            },
        )?;

        Ok(SystemAudio {
            stream_handle,
            input_consumer,
            output_producer,
            ready_to_receive,
        })
    }

    pub fn num_samples_available(&self) -> usize {
        self.input_consumer.occupied_len()
    }

    pub fn receive_audio(&mut self, num_samples: usize) -> Vec<f32> {
        self.input_consumer.pop_iter().take(num_samples).collect()
    }

    pub fn send_audio(&mut self, data: &[f32]) {
        self.output_producer.push_slice(data);
    }

    pub fn set_ready_to_receive(&self, ready: bool) {
        self.ready_to_receive
            .store(ready, std::sync::atomic::Ordering::Relaxed);
    }
}

impl Drop for SystemAudio {
    fn drop(&mut self) {
        self.stream_handle.stop();
    }
}

#[allow(unused)]
pub fn list_device_names() {
    match rtaudio::Host::new(Api::Unspecified) {
        Ok(rt) => {
            println!("Available Output Devices:");
            for device_info in rt.iter_output_devices() {
                println!("\"{}\"", device_info.name);
            }
            println!();
            println!("Available Input Devices:");
            for device_info in rt.iter_input_devices() {
                println!("\"{}\"", device_info.name);
            }
        }
        Err(e) => {
            eprintln!("{}", e);
        }
    }
    println!();
}
