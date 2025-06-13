from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch

app = FastAPI()

# Load DialoGPT
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

chat_history_ids = None  # For conversation memory

faq_data = {
    "What are your business hours?": "Our business hours are 24/7.",
    "What is your company name?": "We are Tech — a leading provider of software solutions.",
    "Where are you located?": "We are located at Gairidhara, Naxal.",
    "Do I need an account to make a purchase?": "You can checkout as a guest, but creating an account offers additional benefits.",
    "How do I place an order?": "Simply browse products, add to cart, and proceed to checkout.",
    "What payment methods do you accept?": "We accept credit/debit cards, eSewa, and Khalti.",
    "What are your shipping options?": "We offer express and standard shipping across Nepal.",
    "How can I track my order?": "After placing an order, you'll receive a tracking link via email.",
    "What is your return policy?": "We accept returns within 7 days of delivery in original condition.",
    "How can I create an account?": "Click on 'Sign Up' and fill in the required information to register.",
    "Tell me about your company?": "We build intelligent, customer-focused digital products.",
    "Can I change or cancel my order?": "Yes, within 24 hours of placing it. Contact support.",
    "Do you offer technical support?": "Yes, we provide 24/7 technical support.",
    "Are the products you sell authentic?": "Absolutely, we only deal with genuine and verified products.",
    "Do you provide custom software solutions?": "Yes, we specialize in building custom software tailored to your needs.",
    "Do you offer discounts or promotions?": "Check our website banner or subscribe to our newsletter for offers.",
    "How do I find the right size or fit?": "Refer to our size guide available on each product page.",
    "Can I get a free consultation?": "Yes, we offer a free initial consultation session."
}


class UserInput(BaseModel):
    message: str


def get_faq_response(user_input: str):
    questions = list(faq_data.keys())
    vectorizer = TfidfVectorizer().fit_transform([user_input] + questions)
    similarities = cosine_similarity(vectorizer[0:1], vectorizer).flatten()

    best_match_idx = similarities[1:].argmax()
    if similarities[best_match_idx + 1] > 0.6:  # Threshold
        return faq_data[questions[best_match_idx]]
    return None


@app.post("/chat")
def chat(user_input: UserInput):
    global chat_history_ids

    # 1. Try matching with FAQ
    faq_response = get_faq_response(user_input.message)
    if faq_response:
        return {"response": faq_response}

    # 2. Fallback to DialoGPT
    input_ids = tokenizer.encode(user_input.message + tokenizer.eos_token, return_tensors="pt")
    bot_input_ids = torch.cat([chat_history_ids, input_ids], dim=-1) if chat_history_ids is not None else input_ids

    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )

    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return {"response": response}
