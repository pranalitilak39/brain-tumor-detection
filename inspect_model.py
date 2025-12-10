from tensorflow.keras.models import load_model

model = load_model("brain_tumor_cnn_full.h5")

with open("model_summary.txt", "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

print("\n\nLAYER NAMES:")
layer_names = []
for layer in model.layers:
    layer_names.append(f"{layer.name} → {layer.__class__.__name__}")
    print(layer.name, " → ", layer.__class__.__name__)

with open("layer_names.txt", "w") as f:
    f.write("\n".join(layer_names))
