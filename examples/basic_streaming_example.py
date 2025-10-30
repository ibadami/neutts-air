import os
import torch
import numpy as np
from neuttsair.neutts import NeuTTSAir
import pyaudio
from queue import Queue
from threading import Thread


def audio_producer(tts, input_text, ref_codes, ref_text, audio_queue):
    """Thread that generates audio chunks and puts them in the queue."""
    try:
        for chunk in tts.infer_stream(input_text, ref_codes, ref_text):
            audio = (chunk * 32767).astype(np.int16)
            audio_queue.put(audio)
    except Exception as e:
        print(f"Error in producer: {e}")
    finally:
        audio_queue.put(None)  # Signal end of stream


def audio_consumer(audio_queue, p, stream):
    """Thread that consumes audio chunks from the queue and plays them."""
    try:
        while True:
            audio = audio_queue.get()
            if audio is None:  # End of stream signal
                break
            stream.write(audio.tobytes())
            audio_queue.task_done()
    except Exception as e:
        print(f"Error in consumer: {e}")


def main(input_text, ref_codes_path, ref_text, backbone):
    assert backbone in ["neuphonic/neutts-air-q4-gguf", "neuphonic/neutts-air-q8-gguf"], "Must be a GGUF ckpt as streaming is only currently supported by llama-cpp."
    
    # Initialize NeuTTSAir with the desired model and codec
    tts = NeuTTSAir(
        backbone_repo=backbone,
        backbone_device="cpu",
        codec_repo="neuphonic/neucodec-onnx-decoder",
        codec_device="cpu"
    )

    # Check if ref_text is a path if it is read it if not just return string
    if ref_text and os.path.exists(ref_text):
        with open(ref_text, "r") as f:
            ref_text = f.read().strip()

    if ref_codes_path and os.path.exists(ref_codes_path):
        ref_codes = torch.load(ref_codes_path)

    print(f"Generating audio for input text: {input_text}")
    
    # Create a queue for audio chunks
    audio_queue = Queue(maxsize=3)  # Buffer up to 3 chunks ahead
    
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=24_000,
        output=True
    )
    
    # Start producer thread (decoding)
    producer_thread = Thread(
        target=audio_producer,
        args=(tts, input_text, ref_codes, ref_text, audio_queue)
    )
    producer_thread.start()
    
    # Start consumer thread (playback)
    consumer_thread = Thread(
        target=audio_consumer,
        args=(audio_queue, p, stream)
    )
    consumer_thread.start()
    
    print("Streaming with parallel decoding and playback...")
    
    # Wait for both threads to complete
    producer_thread.join()
    consumer_thread.join()
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    print("Streaming complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NeuTTSAir Example")
    parser.add_argument(
        "--input_text", 
        type=str, 
        required=True, 
        help="Input text to be converted to speech"
    )
    parser.add_argument(
        "--ref_codes", 
        type=str, 
        default="./samples/dave.pt", 
        help="Path to pre-encoded reference audio"
    )
    parser.add_argument(
        "--ref_text",
        type=str,
        default="./samples/dave.txt", 
        help="Reference text corresponding to the reference audio",
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="output.wav", 
        help="Path to save the output audio"
    )
    parser.add_argument(
        "--backbone", 
        type=str, 
        default="neuphonic/neutts-air-q8-gguf",
        help="Huggingface repo containing the backbone checkpoint. Must be GGUF."
    )
    args = parser.parse_args()
    main(
        input_text=args.input_text,
        ref_codes_path=args.ref_codes,
        ref_text=args.ref_text,
        backbone=args.backbone,
    )
