import dataclasses
from enum import auto, Enum
from typing import List, Tuple


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    CHATML = auto()
    QWEN2 = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False

    def get_prompt(self):
        messages = self.messages
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0].replace("<image>", "").strip()
            if 'mmtag' in self.version:
                messages[0] = (init_role, init_msg)
                messages.insert(0, (self.roles[0], "<Image><image></Image>"))
                messages.insert(1, (self.roles[1], "Received."))
            else:
                messages[0] = (init_role, "<image>\n" + init_msg)

        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += message + seps[i % 2]
                else:
                    ret += ""
        elif self.sep_style == SeparatorStyle.CHATML:
            ret = "" if self.system == "" else self.system + self.sep + "\n"
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, images, _ = message
                        message = "<image>" * len(images) + message
                    ret += role + "\n" + message + self.sep + "\n"
                else:
                    ret += role + "\n"
        # elif self.sep_style == SeparatorStyle.QWEN2:
        #     seps = [self.sep, self.sep2]
        #     ret = self.system + seps[0]
        #     for i, (role, message) in enumerate(messages):
        #         if message:
        #             if type(message) is tuple:
        #                 message, _, _ = message
        #             ret += role + ": " + message + seps[i % 2]
        #         else:
        #             ret += role + ":"
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version)


# Used during pretraining
conv_plain = Conversation(
    system="",
    roles=("", ""),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.PLAIN,
    sep="\n",
)

# Should be used for Vicuna
conv_v1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

# Should be used for Phi-3-mini models
conv_phi3_instruct = Conversation(
    system="""<|system|>\nYou are a helpful AI assistant.""",
    roles=("\n<|user|>\n", "\n<|assistant|>\n"),
    version="phi3",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|end|>",
)

# from https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/internvl/conversation.py
# internlm2-chat
conv_qwen2_cap = Conversation(
    system="""<|im_start|>system\nYou are a helpful AI assistant.""",
    roles=('<|im_start|>user\n', '<|im_start|>assistant\n'),
    version="qwen2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep='<|im_end|>',
)
# https://github.com/haotian-liu/LLaVA/pull/1573/files and 
# https://github.com/haotian-liu/LLaVA/issues/1153#issuecomment-1956308533
# conv_qwen2_cap = Conversation(
#     system="A chat between a curious user and an artificial intelligence assistant. "
#     "The assistant gives helpful, detailed, and polite answers to the user's questions.",
#     roles=("USER", "ASSISTANT"),
#     version="qwen_v2",
#     messages=(),
#     offset=0,
#     sep_style=SeparatorStyle.QWEN2,
#     sep=" ",
#     sep2="<|endoftext|>",
# )

# Should be used for LLaMA-3 models
conv_llama3 = Conversation(
    system="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.""",
    roles=("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    version="llama3",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|eot_id|>",
)

default_conversation = conv_qwen2_cap
conv_templates = {
    "default": conv_phi3_instruct,
    "plain": conv_plain,
    "v1": conv_v1,
    "phi3_instruct": conv_phi3_instruct,
    "qwen2_cap": conv_qwen2_cap,
}

if __name__ == "__main__":
    print(default_conversation.get_prompt())
