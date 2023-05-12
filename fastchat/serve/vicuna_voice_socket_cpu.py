# server.py
import socketio
import argparse
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from fastchat.conversation import conv_templates, SeparatorStyle
from fastchat.serve.vicuna import generate_stream

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import whisper
import wave
from eventlet import sleep

sio = socketio.Server(cors_allowed_origins='*')
app = socketio.WSGIApp(sio)

whisper_model = whisper.load_model("base")

# Function to save the received audio file
def save_audio_file(audio_data, file_path):
    with open(file_path, 'wb') as audio_file:
        audio_file.write(audio_data)
    print("done saving")

@sio.event
def connect(sid, environ):
    print("connect ", sid)

@sio.event
def disconnect(sid):
    print('disconnect ', sid)

from eventlet import spawn

@sio.event
def audio(sid, audio_data):
    spawn(handle_audio, sid, audio_data)

def handle_audio(sid, audio_data):
    audio_file_path = 'received_audio.mp3'
    save_audio_file(audio_data, audio_file_path)
    # Transcribe the audio using Whisper Jax
    text = whisper_model.transcribe(audio_file_path)["text"]
    print(str(text))
    # Use the transcribed text as input
    inp = str(text)
    if not inp:
        print("exit...")
        sio.emit('exit', room=sid)
        return

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
            print(token)
            sio.emit('message', token, room=sid)  # Send token to the client
            sleep(0.01)  # Add a small delay to allow the emit to be sent
            pre = now - 1
    final_token = " ".join(outputs[pre:])
    sio.emit('message', final_token, room=sid)  # Send the final token to the client
    sleep(0.01)  # Add a small delay to allow the emit to be sent
    conv.messages[-1][-1] = " ".join(outputs)



def main(args):
    global model_name
    global conv
    global tokenizer
    global model

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

    # Chat
    conv = conv_templates[args.conv_template].copy()

    if __name__ == '__main__':
        import eventlet.wsgi
        import eventlet
        eventlet.wsgi.server(eventlet.listen(('localhost', 11245)), app)


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
