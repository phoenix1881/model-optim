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

# @app.route('/ask', methods=['POST'])
# def ask():
#     data = request.json
#     symptoms = data.get('symptoms', '')
#     question = data.get('question', '')

#     if not symptoms or not question:
#         return jsonify({"error": "Both symptoms and question are required"}), 400

#     # Format the prompt
#     prompt = f"Symptoms: {symptoms}\nQuestion: {question}\nAnswer:"

#     # Tokenize input
#     input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

#     # Generate response
#     output_ids = model.generate(
#         input_ids,
#         max_length=100,
#         temperature=0.7,
#         top_p=0.9,
#         repetition_penalty=1.2
#     )

#     answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)

#     # return jsonify({"answer": answer.replace(prompt, '').strip()})
#     return jsonify({"answer": "Take an antibiotic like citrizen and rest."})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)


from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Load GPT-2 model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "gpt2"

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

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

    # Add context from medical report
    prompt = f"""Medical Report:\n{medical_context}\n\nQuestion: {question}\nAnswer:"""

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    output_ids = model.generate(
        input_ids,
        max_length=150,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2
    )

    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return jsonify({"answer": answer.replace(prompt, '').strip()})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
