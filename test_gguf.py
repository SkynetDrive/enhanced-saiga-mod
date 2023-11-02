import logging

DEFAULT_MESSAGE_TEMPLATE = "<s>{role}\n{content}</s>\n"
DEFAULT_SYSTEM_PROMPT = "Ты — PavelGPT, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."


class Conversation:
    def __init__(
            self,
            message_template=DEFAULT_MESSAGE_TEMPLATE,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            start_token_id=2,
            # Bot token may be a list or single int
            bot_token_id=10093,  # yarn_mistral_7b_128k
            # bot_token_id=46787,  # rugpt35_13b
            # int (amount of questions a