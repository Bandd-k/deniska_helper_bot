import config
import openai


# setup openai
openai.api_key = config.openai_api_key
if config.openai_api_base is not None:
    openai.api_base = config.openai_api_base


OPENAI_COMPLETION_OPTIONS = {
    "temperature": 0.7,
    "max_tokens": 2096,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "request_timeout": 150.0,
}


class ChatGPT:
    def __init__(self, model="gpt-4o"):
        assert (
            model in config.models["available_text_models"]
            or model in config.models["available_premium_models"]
        ), f"Unknown model: {model}. Use settings to choose new one"
        self.model = model

    async def send_message(self, message, dialog_messages=[], chat_mode="assistant"):
        if chat_mode not in config.chat_modes.keys():
            raise ValueError(f"Chat mode {chat_mode} is not supported")

        n_dialog_messages_before = len(dialog_messages)
        answer = None
        while answer is None:
            try:
                if (
                    self.model in config.models["available_text_models"]
                    or self.model in config.models["available_premium_models"]
                ):
                    messages = self._generate_prompt_messages(
                        message, dialog_messages, chat_mode
                    )
                    r = await openai.ChatCompletion.acreate(
                        model=self.model, messages=messages, **OPENAI_COMPLETION_OPTIONS
                    )
                    answer = r.choices[0].message["content"]
                else:
                    raise ValueError(f"Unknown model: {self.model}")

                answer = self._postprocess_answer(answer)
                n_input_tokens, n_output_tokens = (
                    r.usage.prompt_tokens,
                    r.usage.completion_tokens,
                )
            except openai.error.InvalidRequestError as e:  # too many tokens
                if len(dialog_messages) == 0:
                    raise ValueError(
                        "Dialog messages is reduced to zero, but still has too many tokens to make completion"
                    ) from e

                # forget first message in dialog_messages
                dialog_messages = dialog_messages[1:]

        n_first_dialog_messages_removed = n_dialog_messages_before - len(
            dialog_messages
        )

        return (
            answer,
            (n_input_tokens, n_output_tokens),
            n_first_dialog_messages_removed,
        )

    def _generate_prompt_messages(self, message, dialog_messages, chat_mode):
        prompt = config.chat_modes[chat_mode]["prompt_start"]

        messages = [{"role": "system", "content": prompt}]
        for dialog_message in dialog_messages:
            messages.append({"role": "user", "content": dialog_message["user"]})
            messages.append({"role": "assistant", "content": dialog_message["bot"]})
        messages.append({"role": "user", "content": message})

        return messages

    def _postprocess_answer(self, answer):
        answer = answer.strip()
        return answer


async def transcribe_audio(audio_file) -> str:
    r = await openai.Audio.atranscribe("whisper-1", audio_file)
    return r["text"] or ""


async def generate_images(prompt, n_images=4, size="512x512"):
    r = await openai.Image.acreate(prompt=prompt, n=n_images, size=size)
    image_urls = [item.url for item in r.data]
    return image_urls


async def is_content_acceptable(prompt):
    r = await openai.Moderation.acreate(input=prompt)
    return not all(r.results[0].categories.values())
