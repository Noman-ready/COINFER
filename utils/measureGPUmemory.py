import torch
from thop import profile
from torchvision import transforms
from PIL import Image


def measure_segment(model, input_tensor, name):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    device = next(model.parameters()).device
    input_on_device = input_tensor.to(device)

    input_bytes = input_on_device.element_size() * input_on_device.nelement()
    print(f"[{name}] 输入张量大小: {input_bytes / 1024 ** 2:.2f} MB")

    model.eval()
    with torch.no_grad():
        output = model(input_on_device)

    peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 2
    output_bytes = output.element_size() * output.nelement()

    print(f"[{name}] 显存峰值: {peak_memory:.2f} MB")
    print(f"[{name}] 输出张量大小: {output_bytes / 1024 ** 2:.2f} MB")

    output_cpu = output.cpu()
    del output
    torch.cuda.empty_cache()
    return output_cpu
