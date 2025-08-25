import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import streamlit as st
from textwrap import wrap

# Set page configuration
st.set_page_config(
    page_title="BioGPT Clinical Intervention Advisor",
    page_icon="🏥",
    layout="wide"
)

@st.cache_resource(show_spinner=False)
def load_model():
    """Load the model and tokenizer with caching to avoid reloading on every interaction"""
    try:
        # Path to fine-tuned model
        model_path = "/content/BioGPT_weights"

        if not os.path.exists(model_path):
            st.error(f"Model not found at {model_path}. Please run training first.")
            return None, None

        # Loading the tokenizer from the fine-tuned model
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Loading the base model with the correct vocabulary size
        base_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/biogpt",
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Resizing token embeddings to match the fine-tuned tokenizer
        base_model.resize_token_embeddings(len(tokenizer))

        # Loading the PEFT adapter
        model = PeftModel.from_pretrained(
            base_model,
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Setting padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return model, tokenizer
        
    except Exception as e:
        st.error(f"Model loading failed with error: {e}")
        return None, None

def main():
    st.title("🏥 BioGPT Clinical Intervention Advisor")
    st.markdown("""
    This tool provides specific, actionable interventions based on TBI management guidelines.
    Enter a clinical snapshot below to get recommendations.
    """)
    
    # Load model (with caching)
    with st.spinner("Loading model (this may take a moment)..."):
        model, tokenizer = load_model()
    
    if model is None or tokenizer is None:
        st.stop()
    
    # Example inputs
    example_inputs = [
        "ICP 25 mmHg, PbtO2 14 mmHg, CPP 68 mmHg",
        "GCS 6, ICP 30 mmHg, MAP 90 mmHg",
        "Temperature 39.2°C, HR 110, BP 85/50 mmHg"
    ]
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Clinical Input")
        
        # Use example button
        example_clicked = st.button("Use Example Input")
        
        # Text input for clinical data
        default_text = example_inputs[0] if example_clicked else ""
        user_input = st.text_area(
            "Enter the clinical snapshot:",
            value=default_text,
            height=100,
            placeholder="e.g., 'ICP 25 mmHg, PbtO2 14 mmHg, CPP 68 mmHg'"
        )
        
        # Generate button
        generate_btn = st.button("Generate Recommendations", type="primary")
    
    with col2:
        st.subheader("Quick Examples")
        for example in example_inputs:
            st.caption(f"• {example}")
    
    # Display results when generate button is clicked
    if generate_btn and user_input:
        with st.spinner("Generating recommendations..."):
            try:
                # Creating a structured prompt
                prompt = f"""### Clinical Scenario:
You are a neurointensivist managing a severe TBI patient in the ICU.

### Current Status:
{user_input}

### Task:
Provide 1 specific, actionable interventions based on TBI management guidelines. Prioritize by urgency. Please write in a more natural language format.

### Recommended Interventions:
"""

                # Tokenizing input
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(model.device)

                # Generating response
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=200,
                        temperature=0.3,
                        top_p=0.9,
                        repetition_penalty=1.2,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )

                # Decoding response
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Extracting only the new content (after the prompt)
                response = generated_text[len(prompt):].strip()

                # Display results
                st.subheader("Recommended Interventions")
                st.markdown("---")
                
                # Format the response with better styling
                st.info(response)
                
                # Show raw prompt and response in expander for debugging
                with st.expander("Debug: View prompt and full response"):
                    st.text("Prompt sent to model:")
                    st.text(prompt)
                    st.text("Full model response:")
                    st.text(generated_text)
                    
            except Exception as e:
                st.error(f"An error occurred during generation: {e}")

    elif generate_btn and not user_input:
        st.warning("Please enter a clinical snapshot first.")

    # Add footer
    st.markdown("---")
    st.caption("Note: This tool is for educational purposes only. Always follow your institution's protocols and consult with senior clinicians when making patient care decisions.")

if __name__ == "__main__":
    main()