use ort::{
    session::{builder::GraphOptimizationLevel, Session, SessionInputValue},
    value::Tensor,
};

use std::{
    collections::HashMap,
    io::{self, Write},
    path::Path,
};

use tokenizers::Tokenizer;

// Model configuration - adjust these based on your model's config
const NUM_KEY_VALUE_HEADS: usize = 1; // config.num_key_value_heads (from model inspection)
const HEAD_DIM: usize = 256; // config.head_dim
const NUM_HIDDEN_LAYERS: usize = 26; // config.num_hidden_layers (from model inspection: 0-25)
const EOS_TOKEN_ID: i64 = 106; // <end_of_turn> token
const MAX_NEW_TOKENS: i32 = 1024;

pub struct Gemma {
    session: Session,
    tokenizer: Tokenizer,
}

impl Gemma {
    pub fn new() -> ort::Result<Self> {
        let model_path = "./model_q4.onnx";
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path)?;

        // Load tokenizer
        let tokenizer_path = "./tokenizer.json";
        let tokenizer = Tokenizer::from_file(Path::new(tokenizer_path))
            .map_err(|e| ort::Error::new(format!("Failed to load tokenizer: {}", e)))?;

        Ok(Self { session, tokenizer })
    }

    pub fn chat(&mut self, prompt: &str) -> ort::Result<String> {
        // Create chat messages (simulating the chat template)

        let system_msg = "Please answer in exactly one short sentence.";
        let user_msg = prompt;

        // For Gemma, the chat template typically looks like:
        let prompt = format!(
            "<start_of_turn>system\n{system_msg}<end_of_turn>\n<start_of_turn>user\n{user_msg}<end_of_turn>\n<start_of_turn>model\n",
        );

        let mut stdout = io::stdout();

        // Tokenize the prompt
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| ort::Error::new(format!("Tokenization failed: {}", e)))?;

        let mut input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();

        let batch_size = 1;

        // Initialize past key values - will be populated after first inference
        let mut past_key_values: HashMap<String, Vec<f32>> = HashMap::new();
        let mut cache_seq_len = 0usize;

        stdout.flush().unwrap();

        let mut generated_tokens = Vec::new();

        // Generation loop
        for step in 0..MAX_NEW_TOKENS {
            // Prepare input tensors
            let (current_input_ids, current_seq_len) = if step == 0 {
                // First step: use full prompt
                (input_ids.clone(), input_ids.len())
            } else {
                // Subsequent steps: only use the last generated token
                (vec![input_ids[input_ids.len() - 1]], 1)
            };

            // Create input_ids tensor
            let input_ids_tensor = Tensor::from_array((
                vec![batch_size as i64, current_seq_len as i64],
                current_input_ids,
            ))?;

            // Create position_ids for the current tokens

            let position_ids: Vec<i64> = if step == 0 {
                (0..current_seq_len as i64).collect()
            } else {
                vec![(cache_seq_len + current_seq_len - 1) as i64]
            };

            let position_ids_tensor = Tensor::from_array((
                vec![batch_size as i64, position_ids.len() as i64],
                position_ids,
            ))?;

            // Build inputs

            let mut inputs_vec: Vec<(String, SessionInputValue)> = vec![
                ("input_ids".to_owned(), input_ids_tensor.into()),
                ("position_ids".to_owned(), position_ids_tensor.into()),
            ];

            // Add past key values - model expects them even on first step

            for layer in 0..NUM_HIDDEN_LAYERS {
                let key_name = format!("past_key_values.{}.key", layer);
                let value_name = format!("past_key_values.{}.value", layer);

                if step == 0 {
                    // First step: try creating minimal valid cache tensors
                    // Some models expect at least 1 in all dimensions
                    let empty_data = vec![0.0f32; batch_size * NUM_KEY_VALUE_HEADS * 1 * HEAD_DIM];
                    let empty_key_tensor = Tensor::from_array((
                        vec![
                            batch_size as i64,
                            NUM_KEY_VALUE_HEADS as i64,
                            1i64, // Use 1 instead of 0 to avoid invalid dimension
                            HEAD_DIM as i64,
                        ],
                        empty_data.clone(),
                    ))?;

                    let empty_value_tensor = Tensor::from_array((
                        vec![
                            batch_size as i64,
                            NUM_KEY_VALUE_HEADS as i64,
                            1i64, // Use 1 instead of 0 to avoid invalid dimension
                            HEAD_DIM as i64,
                        ],
                        empty_data,
                    ))?;

                    inputs_vec.push((key_name, empty_key_tensor.into()));

                    inputs_vec.push((value_name, empty_value_tensor.into()));
                } else if let (Some(key_data), Some(value_data)) = (
                    past_key_values.get(&key_name),
                    past_key_values.get(&value_name),
                ) {
                    // Subsequent steps: use actual cache data
                    let expected_size = batch_size * NUM_KEY_VALUE_HEADS * cache_seq_len * HEAD_DIM;

                    if key_data.len() == expected_size
                        && value_data.len() == expected_size
                        && cache_seq_len > 0
                    {
                        let key_tensor = Tensor::from_array((
                            vec![
                                batch_size as i64,
                                NUM_KEY_VALUE_HEADS as i64,
                                cache_seq_len as i64,
                                HEAD_DIM as i64,
                            ],
                            key_data.clone(),
                        ))?;

                        let value_tensor = Tensor::from_array((
                            vec![
                                batch_size as i64,
                                NUM_KEY_VALUE_HEADS as i64,
                                cache_seq_len as i64,
                                HEAD_DIM as i64,
                            ],
                            value_data.clone(),
                        ))?;

                        inputs_vec.push((key_name, key_tensor.into()));

                        inputs_vec.push((value_name, value_tensor.into()));
                    } else {
                        // Fall back to minimal cache if there's an issue

                        let empty_data =
                            vec![0.0f32; batch_size * NUM_KEY_VALUE_HEADS * 1 * HEAD_DIM];

                        let empty_key_tensor = Tensor::from_array((
                            vec![
                                batch_size as i64,
                                NUM_KEY_VALUE_HEADS as i64,
                                1i64,
                                HEAD_DIM as i64,
                            ],
                            empty_data.clone(),
                        ))?;

                        let empty_value_tensor = Tensor::from_array((
                            vec![
                                batch_size as i64,
                                NUM_KEY_VALUE_HEADS as i64,
                                1i64,
                                HEAD_DIM as i64,
                            ],
                            empty_data,
                        ))?;

                        inputs_vec.push((key_name, empty_key_tensor.into()));
                        inputs_vec.push((value_name, empty_value_tensor.into()));
                    }
                }
            }

            // Run inference
            let outputs = self.session.run(inputs_vec)?;

            // Extract logits
            let logits_output = outputs
                .get("logits")
                .ok_or_else(|| ort::Error::new("No logits output found".to_string()))?;

            let (logits_dims, logits_data) = logits_output.try_extract_tensor::<f32>()?;

            // Get the logits for the last token
            let vocab_size = logits_dims[logits_dims.len() - 1] as usize;

            let last_token_logits = if current_seq_len > 1 {
                // Multiple tokens: get the last token's logits
                let seq_len_idx = logits_dims[1] as usize - 1;

                &logits_data[seq_len_idx * vocab_size..(seq_len_idx + 1) * vocab_size]
            } else {
                // Single token: use all logits
                &logits_data[..vocab_size]
            };

            // Simple argmax sampling

            let next_token_id = last_token_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0 as i64;

            // Update past key values for next iteration
            cache_seq_len = 0; // Reset and recalculate

            for layer in 0..NUM_HIDDEN_LAYERS {
                let present_key_name = format!("present.{}.key", layer);

                let present_value_name = format!("present.{}.value", layer);

                if let (Some(present_key), Some(present_value)) = (
                    outputs.get(&present_key_name),
                    outputs.get(&present_value_name),
                ) {
                    let (key_dims, key_data) = present_key.try_extract_tensor::<f32>()?;

                    let (_value_dims, value_data) = present_value.try_extract_tensor::<f32>()?;

                    // Calculate cache sequence length from the key tensor shape
                    if layer == 0 {
                        cache_seq_len = key_dims[2] as usize;
                    }

                    let past_key_name = format!("past_key_values.{}.key", layer);

                    let past_value_name = format!("past_key_values.{}.value", layer);

                    past_key_values.insert(past_key_name, key_data.to_vec());

                    past_key_values.insert(past_value_name, value_data.to_vec());
                }
            }

            // Add the generated token
            input_ids.push(next_token_id);
            generated_tokens.push(next_token_id);

            // Check for EOS token
            if next_token_id == EOS_TOKEN_ID {
                break;
            }

            // Decode and print the token (streaming)
            if let Ok(token_str) = self.tokenizer.decode(&[next_token_id as u32], true) {
                print!("{}", token_str);
                stdout.flush().unwrap();
            }
        }

        println!("\n");

        // Return the full generated response
        if let Ok(full_response) = self.tokenizer.decode(
            &generated_tokens
                .iter()
                .map(|&id| id as u32)
                .collect::<Vec<_>>(),
            true,
        ) {
            Ok(full_response)
        } else {
            Ok(String::new())
        }
    }
}
