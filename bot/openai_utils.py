import config
import openai
from openai import AsyncOpenAI, AsyncAzureOpenAI


azureclient_sweden = AsyncAzureOpenAI(
    api_key=config.azure_openai_sweden_api_key,
    base_url="https://chattyswedencentral.openai.azure.com/openai/",
    api_version="2025-01-01-preview",
)

azureclient_4o_transcribe = AsyncAzureOpenAI(
    api_key=config.azure_openai_eastus2_api_key,
    base_url=config.azure_openai_endpoint_eastus2_4o_transcribe,
    api_version="2025-03-01-preview",
)


no_system_message_models = ["gpt-5-nano", "gpt-5-mini", "gpt-5"]


class ChatGPT:
    async def send_message(
        self, message, model="gpt-5-nano", dialog_messages=[], chat_mode="assistant"
    ):
        if chat_mode not in config.chat_modes.keys():
            raise ValueError(f"Chat mode {chat_mode} is not supported")

        n_dialog_messages_before = len(dialog_messages)

        reasoning_effort = "minimal"
        if model == "gpt-5-mini-thinking":
            model = "gpt-5-mini"
            reasoning_effort = "medium"

        OPENAI_COMPLETION_OPTIONS = {
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "timeout": 20.0,
            "reasoning_effort": reasoning_effort,
            "verbosity": "low",
        }

        answer = None
        while answer is None:
            try:
                if (
                    model in config.models["available_text_models"]
                    or model in config.models["available_premium_models"]
                ):
                    messages = self._generate_prompt_messages(
                        message, dialog_messages, chat_mode
                    )
                    r = await azureclient_sweden.chat.completions.create(
                        model=model, messages=messages, **OPENAI_COMPLETION_OPTIONS
                    )
                    answer = r.choices[0].message.content
                else:
                    raise ValueError(f"Unknown model: {model}")

                answer = self._postprocess_answer(answer)
                n_input_tokens, n_output_tokens = (
                    r.usage.prompt_tokens,
                    r.usage.completion_tokens,
                )
            except openai.BadRequestError as e:  # too many tokens
                if len(dialog_messages) == 0:
                    raise e

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
        messages = []

        prompt = config.chat_modes[chat_mode]["prompt_start"]
        messages = [{"role": "developer", "content": prompt}]

        for dialog_message in dialog_messages:
            messages.append({"role": "user", "content": dialog_message["user"]})
            messages.append({"role": "assistant", "content": dialog_message["bot"]})
        messages.append({"role": "user", "content": message})

        return messages

    def _postprocess_answer(self, answer):
        answer = answer.strip()
        return answer


async def transcribe_audio_4o_azure(audio_file) -> str:
    r = await azureclient_4o_transcribe.audio.transcriptions.create(
        model="gpt-4o-transcribe",
        file=audio_file,
    )
    return r.text.strip() or ""


async def generate_images(prompt, n_images=4, size="512x512"):
    r = await azureclient_4o_transcribe.images.generate(
        prompt=prompt, n=n_images, size=size
    )
    image_urls = [item.url for item in r.data]
    return image_urls
