from fastai.vision.all import *
import gradio as gr

model = load_learner("models/cat_classify.pkl")


# Option 1: Return probabilities as text
def predict_text(img):
    pred_class, _, probabilities = model.predict(img)
    
    labels = {False: 'Not a Cat', True: 'Cat'}
    # Create a nice readable string for output
    result = ""
    for l, p in zip(model.dls.vocab, probabilities):
        result += f"{labels[l]}: {p:.4f}\n"
    result += f"\nPrediction: {labels[pred_class]}"
    return result


# Option 2: Return probabilities as dict (Gradio bar chart)
def predict_chart(img):
    _, _, probs = model.predict(img)
    labels = ["Not a Cat", "Cat"]
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

# Build Gradio interface
iface = gr.Interface(
    fn=predict_chart,          # two outputs: text + chart
    inputs=gr.Image(type="pil"),               # input is an image
    outputs=gr.Label(),        # show both text + chart
    title="Cat vs Not Cat Classifier",
    description="Upload an image and see whether it's a cat or not"
)

iface.launch()