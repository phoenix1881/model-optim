

# from flask import Flask, request, jsonify, render_template
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# app = Flask(__name__)

# # Load GPT-2 model
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model_name = "gpt2"

# model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# @app.route('/')
# def home():
#     return render_template('index.html')

# medical_context = """
# Right middle lobe opacity with loss of the right heart border.
# Left lower lobe opacity with loss of the left heart border.
# No pleural effusion or pneumothorax.
# """

# @app.route('/ask', methods=['POST'])
# def ask():
#     data = request.json
#     question = data.get('question', '')

#     if not question:
#         return jsonify({"error": "Question is required"}), 400

#     # Add context from medical report
#     prompt = f"""Medical Report:\n{medical_context}\n\nQuestion: {question}\nAnswer:"""

#     input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

#     output_ids = model.generate(
#         input_ids,
#         max_length=150,
#         temperature=0.7,
#         top_p=0.9,
#         repetition_penalty=1.2
#     )

#     answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#     return jsonify({"answer": answer.replace(prompt, '').strip()})


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)


from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer
from litgpt.lora import GPT
from litgpt.prompts import PromptStyle
import torch

app = Flask(__name__)

# === Model Config ===
model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
checkpoint_path = "/mnt/object/artfacts/medical-qa-model/model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# === Load model from checkpoint
print("🔄 Loading TinyLLaMA via Lit-GPT...")
model = GPT.from_name(model_name)
model = model.to_empty(device)
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict, strict=False)
model.eval()

@app.route('/')
def home():
    return render_template('index.html')

medical_context = """
Right middle lobe opacity with loss of the right heart border.
Left lower lobe opacity with loss of the left heart border.
No pleural effusion or pneumothorax.
"""

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question', '')

    if not question:
        return jsonify({"error": "Question is required"}), 400

    # Format prompt
    prompt = f"""Medical Report:\n{medical_context}\n\nQuestion: {question}\nAnswer:"""
    styled_prompt = PromptStyle.from_name("alpaca").apply(prompt)

    input_ids = tokenizer(styled_prompt, return_tensors="pt").input_ids.to(device)
    model.max_seq_length = input_ids.shape[-1]
    model.set_kv_cache(batch_size=1)
    model.cos, model.sin = model.rope_cache(device=device)

    with torch.no_grad():
        output = model(input_ids)
        output_ids = torch.argmax(output, dim=-1)

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    answer = decoded.replace(styled_prompt, "").strip()
    return jsonify({"answer": answer})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
