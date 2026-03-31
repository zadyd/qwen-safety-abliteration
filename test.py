from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "Qwen3.5-2B",
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

print("=== Top-level named_children ===")
for name, child in model.named_children():
    print(f"  model.{name}: {type(child).__name__}")

# 尝试所有可能的 layers 路径
print("\n=== 尝试查找 layers ===")

paths_to_try = [
    ("model", "model", "layers"),   # model.model.layers
    ("model", "layers"),            # model.layers
    ("language_model", "layers"),   # language_model.layers
    ("model",),                     # model 本身是 ModuleList
]

layers = None
found_path = None
for path in paths_to_try:
    try:
        if len(path) == 3:
            obj = getattr(model, path[0])
            obj = getattr(obj, path[1])
            obj = getattr(obj, path[2])
        elif len(path) == 2:
            obj = getattr(model, path[0])
            obj = getattr(obj, path[1])
        else:
            obj = getattr(model, path[0])
        if isinstance(obj, (list, torch.nn.ModuleList)):
            layers = obj
            found_path = ".".join(path)
            print(f"  [OK] model.{found_path} -> len={len(layers)}, type={type(layers[0]).__name__}")
            break
        else:
            print(f"  [SKIP] model.{'.'.join(path)} exists but is {type(obj).__name__}, not a list/ModuleList")
    except AttributeError as e:
        print(f"  [MISS] model.{'.'.join(path)} -> AttributeError: {e}")

if layers is None:
    print("\n  无法找到 layers! 需要进一步探索...")
    print("\n=== 继续探索 model.model 的内部结构 ===")
    try:
        inner = model.model
        for name, child in inner.named_children():
            print(f"  model.model.{name}: {type(child).__name__}")
        # 尝试 model.model 本身
        if isinstance(inner, (list, torch.nn.ModuleList)):
            print(f"  model.model IS the layers list! len={len(inner)}")
    except AttributeError:
        print("  model.model 不存在")

else:
    print(f"\n=== layers 数量和类型 ===")
    print(f"  len(layers) = {len(layers)}")
    print(f"  layers[0] type = {type(layers[0]).__name__}")

    # 测试 hook 能否捕获输出
    print("\n=== 测试 Hook 捕获 ===")
    activation_holder = {}

    def test_hook(module, input, output):
        if isinstance(output, tuple):
            print(f"    [TUPLE] len={len(output)}")
            print(f"    [TUPLE] output[0].shape={output[0].shape}")
            activation_holder['captured'] = output[0].detach().clone()
        else:
            print(f"    [NON-TUPLE] type={type(output).__name__}")
            if hasattr(output, 'shape'):
                print(f"    [NON-TUPLE] shape={output.shape}")
            activation_holder['captured'] = output.detach().clone()

    handle = layers[0].register_forward_hook(test_hook)

    tokenizer = AutoTokenizer.from_pretrained("Qwen3.5-2B", trust_remote_code=True)
    text = "Hello world"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        _ = model(**inputs)

    handle.remove()
    print(f"\n  Hook captured: {'YES' if 'captured' in activation_holder else 'NO'}")
    if 'captured' in activation_holder:
        print(f"  Shape: {activation_holder['captured'].shape}")
