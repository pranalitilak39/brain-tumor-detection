from tensorflow.keras.models import load_model

model = load_model("brain_tumor_cnn_full.h5")

model.summary()

print("\n\nLAYER NAMES:")
for layer in model.layers:
    print(layer.name, " → ", layer.__class__.__name__)
