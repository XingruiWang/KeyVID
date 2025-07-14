# Let's see how to retrieve time steps for a model
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModelForCTC
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_dataset
import datasets
import torch

# import model, feature extractor, tokenizer
# model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h")
# tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
# feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")


processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# load first sample of English common_voice
dataset = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="train", streaming=True, trust_remote_code=True)
dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate=16_000))

dataset_iter = iter(dataset)


for _ in range(10):
    sample_1 = next(dataset_iter)
    sample_2 = next(dataset_iter)
    sample = [sample_1, sample_2]
    audio = [s["audio"]["array"] for s in sample]


    # forward sample through model to get greedily predicted transcription ids

    # input_values = feature_extractor(sample["audio"]["array"], return_tensors="pt").input_values
    input_values = processor(audio, return_tensors="pt", padding=True).input_values

    # plot audio signal
    import matplotlib.pyplot as plt
    plt.clf()
    plt.plot(input_values[0].numpy())
    plt.savefig("audio.png")

    # torch.Size([1, 89856]     
    logits = model(input_values).logits[0]
    print(logits.shape, input_values.shape)
    
    import ipdb; ipdb.set_trace()

    pred_ids = torch.argmax(logits, axis=-1) 

    # retrieve word stamps (analogous commands for `output_char_offsets`)
    outputs = processor.decode(pred_ids, output_word_offsets=True)
    # compute `time_offset` in seconds as product of downsampling ratio and sampling_rate
    time_offset = model.config.inputs_to_logits_ratio / processor.feature_extractor.sampling_rate
    # 320 / 16_000 = 0.02
    
    word_offsets = [
        {
            "word": d["word"],
            "start_time": round(d["start_offset"] * time_offset, 2),
            "end_time": round(d["end_offset"] * time_offset, 2),
        }
        for d in outputs.word_offsets
    ]
    # compare word offsets with audio `en_train_0/common_voice_en_19121553.mp3` online on the dataset viewer:
    # https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0/viewer/en
    print(word_offsets[:3])
    [{'word': 'THE', 'start_time': 0.7, 'end_time': 0.78}, {'word': 'TRICK', 'start_time': 0.88, 'end_time': 1.08}, {'word': 'APPEARS', 'start_time': 1.2, 'end_time': 1.64}]