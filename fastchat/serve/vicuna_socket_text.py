# server.py
import socket
import argparse
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from fastchat.conversation import conv_templates, SeparatorStyle
from fastchat.serve.vicuna import generate_stream

def main(args):
    model_name = args.model_name
    num_gpus = args.num_gpus

    # Model
    if args.device == "cuda":
        kwargs = {"torch_dtype": torch.float16}
        if num_gpus == "auto":
            kwargs["device_map"] = "auto"
        else:
            num_gpus = int(num_gpus)
            if num_gpus != 1:
                kwargs.update({
                    "device_map": "auto",
                    "max_memory": {i: "13GiB" for i in range(num_gpus)},
                })
    elif args.device == "cpu":
        kwargs = {}
    else:
        raise ValueError(f"Invalid device: {args.device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if args.wbits > 0:
        from fastchat.serve.load_gptq_model import load_quantized

        print("Loading GPTQ quantized model...")
        model = load_quantized(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name,
            low_cpu_mem_usage=True, **kwargs)

    if args.device == "cuda" and num_gpus == 1:
        model.cuda()

    # Create a server socket and listen for connections
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', 11245))
    server_socket.listen(1)
    print("Server is ready to accept a connection...")

    client_socket, client_address = server_socket.accept()
    print(f"Connection from {client_address} has been established!")

    # Chat
    conv = conv_templates[args.conv_template].copy()
    while True:
        try:
            inp = client_socket.recv(1024).decode("utf-8").strip()
            if not inp:
                print("exit...")
                break
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        
        params = {
            "model": model_name,
            "prompt": prompt,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "stop": conv.sep if conv.sep_style == SeparatorStyle.SINGLE else conv.sep2,
        }

        print(f"{conv.roles[1]}: ", end="", flush=True)

        pre = 0
        for outputs in generate_stream(tokenizer, model, params, args.device):
            outputs = outputs[len(prompt) + 1:].strip()
            outputs = outputs.split(" ")
            now = len(outputs)
            if now - 1 > pre:
                token = " ".join(outputs[pre:now-1])
                client_socket.sendall(token.encode())  # Send token to the client
                pre = now - 1
        final_token = " ".join(outputs[pre:])
        client_socket.sendall((final_token+"").encode())  # Send the final token to the client
        conv.messages[-1][-1] = " ".join(outputs)

    client_socket.close()
    server_socket.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="anon8231489123/vicuna-13b-GPTQ-4bit-128g")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--conv-template", type=str, default="v1")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--wbits", type=int, default = 4)
    parser.add_argument("--groupsize", type=int, default = 128)
    args = parser.parse_args()
    main(args)
