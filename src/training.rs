use burn::nn::loss::CrossEntropyLossConfig;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::train::{ClassificationOutput, InferenceStep, TrainOutput, TrainStep};

use crate::data::TextBatch;
use crate::model::LargeLanguageModel;

impl<B: Backend> LargeLanguageModel<B> {
    /// Runs the forward pass and computes cross-entropy loss over every
    /// token position, packaged as a `ClassificationOutput` so it can be
    /// used with Burn's training metrics (it implements `ItemLazy`, which a
    /// bare `Tensor` does not).
    pub fn forward_loss(
        &self,
        inputs: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
    ) -> ClassificationOutput<B> {
        let outputs = self.forward(inputs);
        let [batch_size, seq_len, vocab_size] = outputs.dims();

        // Flatten sequence dimensions to calculate Cross Entropy
        let outputs = outputs.reshape([batch_size * seq_len, vocab_size]);
        let targets = targets.reshape([batch_size * seq_len]);

        let loss = CrossEntropyLossConfig::new()
            .init(&outputs.device())
            .forward(outputs.clone(), targets.clone());

        ClassificationOutput::new(loss, outputs, targets)
    }
}

impl<B: AutodiffBackend> TrainStep for LargeLanguageModel<B> {
    type Input = TextBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: TextBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_loss(batch.inputs, batch.targets);

        // Clone loss before backward so ownership isn't moved
        let grads = item.loss.clone().backward();

        TrainOutput::new(self, grads, item)
    }
}

impl<B: Backend> InferenceStep for LargeLanguageModel<B> {
    type Input = TextBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: TextBatch<B>) -> ClassificationOutput<B> {
        self.forward_loss(batch.inputs, batch.targets)
    }
}