
# Load model directly
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import os

load_dotenv()  # take environment variables

hf_token = os.getenv("HF_TOKEN")
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNA_bert_6", trust_remote_code=True, token=hf_token)
model = AutoModel.from_pretrained("zhihan1996/DNA_bert_6", trust_remote_code=True, token=hf_token)


dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
inputs = tokenizer(dna, return_tensors='pt',)
res = model(**inputs)

print(res)