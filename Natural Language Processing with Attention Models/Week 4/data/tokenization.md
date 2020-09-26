## Tokenization Instructions

We made every effort to preserve the original text utterances in MultiWOZ 2.0, while also preserving the slot indices for the `span_info` in the End-to-End Multi-Domain Dialog Challenge Track (Task 1) of DSTC8.
While some of the tokenization required to achieve these indices did not seem correct,
to the best of our knowledge, the following tokenization should produce indices largely consistent 
with the original span annotations in the DSTC8 challenge.

```
text = re.sub("/", " / ", text)
text = re.sub("\-", " \- ", text)
text = re.sub("Im", "I\'m", text)
text = re.sub("im", "i\'m", text)
text = re.sub("theres", "there's", text)
text = re.sub("dont", "don't", text)
text = re.sub("whats", "what's", text)
text = re.sub("[0-9]:[0-9]+\. ", "[0-9]:[0-9]+ \. ", text)
text = re.sub("[a-z]\.[A-Z]", "[a-z]\. [A-Z]", text)
text = re.sub("\t:[0-9]+", "\t: [0-9]+", text)
tokens = word_tokenize(text)
```
