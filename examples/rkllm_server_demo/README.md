# RKLLM-Server Demo
## Before Run
Before running the demo, you need to prepare the following files:
- The transformed RKLLM model file in board.
- check the IP address of the board with 'ifconfig' command.
  
## RKLLM-Server-Flask Demo
### Build
You can run the demo with the only command:
```bash
# Usage: ./build_rkllm_server_flask.sh --workshop [RKLLM-Server Working Path] --model_path [Absolute Path of Converted RKLLM Model on Board] --platform [Target Platform: rk3588/rk3576] [--lora_model_path [Lora Model Path]] [--prompt_cache_path [Prompt Cache File Path]]
./build_rkllm_server_flask.sh --workshop /user/data --model_path /user/data/model.rkllm --platform rk3588
```
### Access with API 
After building the RKLLM-Server-Flask, You can use ‘chat_api_flask.py’ to access the RKLLM-Server-Flask and get the answer of RKLLM models.

Attention: you should check the IP address of the board with 'ifconfig' command and replace the IP address in the ‘chat_api_flask.py’.

## RKLLM-Server-Gradio Demo
### Build
You can run the demo with the only command:
```bash
# Usage: ./build_rkllm_server_gradio.sh --workshop [RKLLM-Server Working Path] --model_path [Absolute Path of Converted RKLLM Model on Board] --platform [Target Platform: rk3588/rk3576] [--lora_model_path [Lora Model Path]] [--prompt_cache_path [Prompt Cache File Path]]
./build_rkllm_server_gradio.sh --workshop /user/data --model_path /user/data/model.rkllm --platform rk3588
```
### Access the Server
After running the demo, You can access the RKLLM-Server-Gradio with two ways:
1. Just Start your browser and access the URL: ‘http://[board_ip]:8080/’. You can chat with the RKLLM models in visual interface.
2. Use the 'chat_api_gradio.py'(you need fix the IP address in the code previously) and get the answer of RKLLM models.

## OpenAI-Compatible Multimodal Service (Vision + LLM)

This repo now provides a systemd-ready multimodal API service for models such as Qwen3-VL.

### What it provides

1. OpenAI-compatible endpoint: `/v1/chat/completions`
2. Supports mixed user content: text + image_url/data-url(base64)
3. Uses RKNN vision encoder (`.rknn`) + RKLLM language model (`.rkllm`)
4. Can run as a persistent `systemctl` service

### 1. Prepare models on board

1. Place your converted models on the board, for example:

```bash
/data/models/qwen3-vl-4b-instruct_w8a8_rk3588.rkllm
/data/models/qwen3-vl-4b_vision_rk3588.rknn
```

### 2. Install and start service on board

Run on board:

```bash
cd examples/rkllm_server_demo
sudo bash install_vl_openai_service.sh \
	--rkllm_model /data/models/qwen3-vl-4b-instruct_w8a8_rk3588.rkllm \
	--vision_model /data/models/qwen3-vl-4b_vision_rk3588.rknn \
	--platform rk3588 \
	--model_name qwen3-vl-4b-rk3588 \
	--port 8080
```

This script will:

1. Install dependencies
2. Compile multimodal encoder binaries on board
3. Install service files and runtime libs to `/opt/rkllm-vl-openai`
4. Register and start `rkllm-vl-openai.service`

### 3. Manage service

```bash
sudo systemctl status rkllm-vl-openai.service
sudo systemctl restart rkllm-vl-openai.service
sudo journalctl -u rkllm-vl-openai.service -f
```

### 4. API request example

```bash
curl http://<board_ip>:8080/v1/chat/completions \
	-H "Content-Type: application/json" \
	-d '{
		"model": "qwen3-vl-4b-rk3588",
		"stream": false,
		"messages": [
			{
				"role": "user",
				"content": [
					{"type": "text", "text": "Describe this image."},
					{"type": "image_url", "image_url": {"url": "https://example.com/demo.jpg"}}
				]
			}
		]
	}'
```
   