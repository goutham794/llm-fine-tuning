from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_TEMPLATE_LIST = [
    "### Question : Extract span of text from the customer review associated with the topic - {}. Customer Review - '{}'",
    "[INST] Extract span of text from the customer review associated with the topic - {}.[/INST] Customer Review - '{}'",
]

REVIEW = ") I was fully satisfied with the product, quite large, very nice design, ergonomic handle and therefore very comfortable"
TOPIC = "Size"
SPAN = (42, 53)


# Load your baseline model and tokenizer
base_model_id = "mistralai/Mistral-7B-Instruct-v0.1"
baseline = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

EOS_TOKEN = "</s>"

for prompt_template in PROMPT_TEMPLATE_LIST:
    prompt = prompt_template.format(TOPIC, REVIEW)
    encoded_prompt = tokenizer(prompt, return_tensors="pt").to("cuda")
    generated_ids = baseline.generate(
        **encoded_prompt,
        max_new_tokens=300,  # set accordingly to your test_output
        do_sample=False,
    )

    decoded_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # Output results for comparison
    print(f"Generated Output: {decoded_output}\n")
    print(f"Expected Output: {REVIEW[SPAN[0]:SPAN[1]]}\n")
    print("-" * 75)
