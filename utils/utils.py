import torch

def create_ts_model (model, save_path):

    # model = ResNetModel()
    model.eval()
    sample_input = torch.randn(1,3,192,640)
    # model(sample_input)
    ts_model = torch.jit.trace_module(model, {"forward": sample_input})
    torch.jit.save(ts_model, save_path)
    ts_model = torch.jit.load(save_path)
    output1 = model(sample_input)
    output2 = ts_model(sample_input)


    # multiple outputs
    if(isinstance(output1, tuple)) or isinstance(output1, list):
        check = [torch.all(out1.isclose(out2)) for out1, out2 in zip(output1, output2)]
        check = all(check)
    # normal ouput
    else: 
        check = torch.all(model(sample_input).isclose(ts_model(sample_input))).bool()
    
    if check:
        print("conversion successful!")
    else:
        print("conversion failure!")