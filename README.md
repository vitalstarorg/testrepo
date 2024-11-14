#### Inference

`Inference` encapsulates the entire transformer process, from transforming a prompt to making final predictions. This encapsulation is crucial to safeguard the mathematical validity and accuracy when experimenting with different transformer architectures using the same model, such as skipping, repeating, or swapping layers.

Below we show both the original encoder-decoder transformer and decorder only GPT2 architectures for comparison. In GPT2, the positioning of the layer normalization is different.

<table align="center" style="border: none;">
  <tr>
    <td align="center">
      <img src="https://raw.githubusercontent.com/vitalstarorg/projector/refs/heads/main/nbs/ModalNet-21.png" width="300"><br>
    </td>
    <td align="center">
      <img src="https://raw.githubusercontent.com/vitalstarorg/projector/refs/heads/main/nbs/gpt2-architecture.png" width="300"><br>
    </td>
  </tr>
  <tr>
    <td align="center">
      <b>Original Transformer</b><br>
      <b>Encoder-Decoder Architecture</b>
    </td>
    <td align="center">
      <b>GPT2</b><br>
      <b>Decoder Only Architecture</b>
    </td>
  </tr>
</table>

The following code represents this transformer in Python:

```python
infer = self.model.inference().prompt("Alan Turing theorized that the")

infer.wte().wpe()
for layer in range(infer.nlayer()):
    infer.lnorm1(layer).attn(layer).sum()
    infer.lnorm2(layer).ffn(layer).sum()
infer.fnorm()                # final normalization
infer.logits()               # output 50257-dimensional logits
infer.argmax()               # next likely predicted token ids
infer.generation()           # next likely predicted tokens

# Alternatively, we can express the transformer using Smallscript
infer.ssrun("""self wte | wpe
       | lnorm1: 0 | attn: 0 | sum | lnorm2: 0 | ffn: 0 | sum
       | layer: 1 | layer: 2 | layer: 3
       | layer: 4 | layer: 5 | layer: 6 | layer: 7
       | layer: 8 | layer: 9 | layer: 10 | layer: 11
       | fnorm | logits""")  # output 50257-dimensional logits
```

