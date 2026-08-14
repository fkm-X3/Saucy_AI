use burn::{
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig},
    prelude::*,
};

#[derive(Config, Debug)]
pub struct LlmConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub max_seq_len: usize,
}

#[derive(Module, Debug)]
pub struct LargeLanguageModel<B: Backend> {
    token_embedding: Embedding<B>,
    position_embedding: Embedding<B>,
    output_layer: Linear<B>,
}

impl LlmConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> LargeLanguageModel<B> {
        LargeLanguageModel {
            token_embedding: EmbeddingConfig::new(self.vocab_size, self.d_model).init(device),
            position_embedding: EmbeddingConfig::new(self.max_seq_len, self.d_model).init(device),
            output_layer: LinearConfig::new(self.d_model, self.vocab_size).init(device),
        }
    }
}

impl<B: Backend> LargeLanguageModel<B> {
    pub fn forward(&self, input_tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch_size, seq_len] = input_tokens.dims();
        let device = &input_tokens.device();

        // Compute Embeddings
        let x = self.token_embedding.forward(input_tokens);

        // Generate and Add Positional Encodings
        let positions = Tensor::<B, 1, Int>::arange(0..seq_len as i64, device)
            .unsqueeze()
            .repeat_dim(0, batch_size);

        let pos_emb = self.position_embedding.forward(positions);

        let h = x + pos_emb;

        // Project Hidden States back to Vocabulary Probs
        self.output_layer.forward(h)
    }
}