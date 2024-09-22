from .nodes.ez_audio_node import EZLoadModelNode,EZGenerateAudioNode


NODE_CLASS_MAPPINGS = {
    "EZLoadModelNode": EZLoadModelNode,
    "EZGenerateAudioNode":EZGenerateAudioNode, 
}

# dict = { "key":value }

NODE_DISPLAY_NAME_MAPPINGS = {
    "EZGenerateAudioNode": "EZ Generate Audio",
    "EZLoadModelNode":"EZ Load Model", 
}

# web ui的节点功能
# WEB_DIRECTORY = "./web"
