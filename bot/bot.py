import io
import logging
import asyncio
import traceback
import html
import json
from datetime import datetime
import openai

import telegram
from telegram import (
    Update,
    User,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    BotCommand,
)
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackContext,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    AIORateLimiter,
    filters,
)
from telegram.constants import ParseMode, ChatAction

import config
import database
import openai_utils


# setup
db = database.Database()
logger = logging.getLogger(__name__)

user_semaphores = {}
user_tasks = {}

HELP_MESSAGE = """Commands:
⚪ /retry – Regenerate last bot answer
⚪ /new – Start new dialog
⚪ /mode – Select chat mode
⚪ /settings – Show settings
⚪ /balance – Show balance
⚪ /help – Show help

🎨 Generate images from text prompts in <b>👩‍🎨 Artist</b> /mode
🎤 You can send <b>Voice Messages</b> instead of text
"""

# HELP_GROUP_CHAT_MESSAGE = """You can add bot to any <b>group chat</b> to help and entertain its participants!

# Instructions (see <b>video</b> below):
# 1. Add the bot to the group chat
# 2. Make it an <b>admin</b>, so that it can see messages (all other rights can be restricted)
# 3. You're awesome!

# To get a reply from the bot in the chat – @ <b>tag</b> it or <b>reply</b> to its message.
# For example: "{bot_username} write a poem about Telegram"
# """


def split_text_into_chunks(text, chunk_size):
    for i in range(0, len(text), chunk_size):
        yield text[i : i + chunk_size]


async def check_channel_subscription(update: Update, context: CallbackContext) -> bool:
    """Check if user is subscribed to the required channel"""
    try:
        user_id = update.effective_user.id
        chat_member = await context.bot.get_chat_member("@Deniska_zver", user_id)
        return chat_member.status in ["member", "administrator", "creator"]
    except telegram.error.TelegramError:
        return False


async def unauthorized_handler(update: Update, context: CallbackContext):
    """Handle unauthorized users by asking them to subscribe"""
    text = (
        "🔒 Для использования этого бота сначала подпишитесь на @Deniska_zver!\n\n"
        "После подписки повторите свой запрос."
    )

    keyboard = [
        [
            InlineKeyboardButton(
                "Подписаться на @Deniska_zver", url="https://t.me/Deniska_zver"
            )
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        text, reply_markup=reply_markup, parse_mode=ParseMode.HTML
    )


async def force_channel_subscription_if_needed(
    update: Update, context: CallbackContext
) -> bool:
    if not await check_channel_subscription(update, context):
        username = update.effective_user.username
        if username in config.no_sub_usernames:
            return True
        await unauthorized_handler(update, context)
        return False
    return True


async def register_user_if_not_exists(
    update: Update, context: CallbackContext, user: User
):
    if not db.check_if_user_exists(user.id):
        db.add_new_user(
            user.id,
            update.message.chat_id,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name,
        )
        db.start_new_dialog(user.id)

    if user.id not in user_semaphores:
        user_semaphores[user.id] = asyncio.Semaphore(1)


async def is_bot_mentioned(update: Update, context: CallbackContext):
    try:
        message = update.message

        if message.chat.type == "private":
            return True

        if message.text is not None and ("@" + context.bot.username) in message.text:
            return True

        if message.reply_to_message is not None:
            if message.reply_to_message.from_user.id == context.bot.id:
                return True
    except:
        return True
    else:
        return False


async def start_handle(update: Update, context: CallbackContext):
    if not await force_channel_subscription_if_needed(update, context):
        return
    await register_user_if_not_exists(update, context, update.message.from_user)
    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())
    db.start_new_dialog(user_id)

    reply_text = "Hi! I'm <b>ChatGPT</b> bot implemented with OpenAI API 🤖\n\n"
    reply_text += HELP_MESSAGE

    await update.message.reply_text(reply_text, parse_mode=ParseMode.HTML)
    await show_chat_modes_handle(update, context)


async def help_handle(update: Update, context: CallbackContext):
    if not await force_channel_subscription_if_needed(update, context):
        return
    await register_user_if_not_exists(update, context, update.message.from_user)
    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())
    await update.message.reply_text(HELP_MESSAGE, parse_mode=ParseMode.HTML)


# async def help_group_chat_handle(update: Update, context: CallbackContext):
#     await register_user_if_not_exists(update, context, update.message.from_user)
#     user_id = update.message.from_user.id
#     db.set_user_attribute(user_id, "last_interaction", datetime.now())

#     text = HELP_GROUP_CHAT_MESSAGE.format(bot_username="@" + context.bot.username)

#     await update.message.reply_text(text, parse_mode=ParseMode.HTML)
#     await update.message.reply_video(config.help_group_chat_video_path)


async def retry_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context):
        return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    dialog_messages = db.get_dialog_messages(user_id, dialog_id=None)
    if len(dialog_messages) == 0:
        await update.message.reply_text("No message to retry 🤷‍♂️")
        return

    last_dialog_message = dialog_messages.pop()
    db.set_dialog_messages(
        user_id, dialog_messages, dialog_id=None
    )  # last message was removed from the context

    await message_handle(
        update,
        context,
        message=last_dialog_message["user"],
        use_new_dialog_timeout=False,
    )


async def message_handle(
    update: Update, context: CallbackContext, message=None, use_new_dialog_timeout=True
):
    # check if bot was mentioned (for group chats)
    if not await is_bot_mentioned(update, context):
        return

    # check if message is edited
    if update.edited_message is not None:
        await edited_message_handle(update, context)
        return

    # Register user and check subscription
    if not await force_channel_subscription_if_needed(update, context):
        return

    await register_user_if_not_exists(update, context, update.message.from_user)

    if await is_previous_message_not_answered_yet(update, context):
        return

    _message = message or update.message.text

    # remove bot mention (in group chats)
    if update.message.chat.type != "private":
        _message = _message.replace("@" + context.bot.username, "").strip()

    user_id = update.message.from_user.id
    chat_mode = db.get_user_attribute(user_id, "current_chat_mode")

    if chat_mode == "artist":
        await generate_image_handle(update, context, message=message)
        return

    async def message_handle_fn():
        # new dialog timeout
        if use_new_dialog_timeout:
            if (
                datetime.now() - db.get_user_attribute(user_id, "last_interaction")
            ).seconds > config.new_dialog_timeout and len(
                db.get_dialog_messages(user_id)
            ) > 0:
                db.start_new_dialog(user_id)
                await update.message.reply_text(
                    f"Starting new dialog due to timeout (<b>{config.chat_modes[chat_mode]['name']}</b> mode) ✅",
                    parse_mode=ParseMode.HTML,
                )
        db.set_user_attribute(user_id, "last_interaction", datetime.now())

        # in case of CancelledError
        n_input_tokens, n_output_tokens = 0, 0
        current_model = db.get_user_attribute(user_id, "current_model")

        try:
            # send typing action
            await update.message.chat.send_action(action="typing")

            if _message is None or len(_message) == 0:
                await update.message.reply_text(
                    "🥲 You sent <b>empty message</b>. Please, try again!",
                    parse_mode=ParseMode.HTML,
                )
                return

            dialog_messages = db.get_dialog_messages(user_id, dialog_id=None)
            parse_mode = (
                ParseMode.MARKDOWN
                if current_model in config.models["reasoning_models"]
                else {"html": ParseMode.HTML, "markdown": ParseMode.MARKDOWN}[
                    config.chat_modes[chat_mode]["parse_mode"]
                ]
            )

            chatgpt_instance = openai_utils.ChatGPT(model=current_model)
            (
                answer,
                (n_input_tokens, n_output_tokens),
                n_first_dialog_messages_removed,
            ) = await chatgpt_instance.send_message(
                _message, dialog_messages=dialog_messages, chat_mode=chat_mode
            )

            answer = answer[:4096]  # telegram message limit

            try:
                await update.message.reply_text(answer, parse_mode=parse_mode)
            except telegram.error.BadRequest:
                await update.message.reply_text(answer)

            # update user data
            new_dialog_message = {
                "user": _message,
                "bot": answer,
                "date": datetime.now(),
            }
            db.set_dialog_messages(
                user_id,
                db.get_dialog_messages(user_id, dialog_id=None) + [new_dialog_message],
                dialog_id=None,
            )

            db.update_n_used_tokens(
                user_id, current_model, n_input_tokens, n_output_tokens
            )

        except asyncio.CancelledError:
            # note: intermediate token updates only work when enable_message_streaming=True (config.yml)
            db.update_n_used_tokens(
                user_id, current_model, n_input_tokens, n_output_tokens
            )
            raise

        except Exception as e:
            error_text = f"Something went wrong during completion. Reason: {e}"
            logger.error(error_text)
            await update.message.reply_text(error_text)
            return

        # send message if some messages were removed from the context
        if n_first_dialog_messages_removed > 0:
            if n_first_dialog_messages_removed == 1:
                text = "✍️ <i>Note:</i> Your current dialog is too long, so your <b>first message</b> was removed from the context.\n Send /new command to start new dialog"
            else:
                text = f"✍️ <i>Note:</i> Your current dialog is too long, so <b>{n_first_dialog_messages_removed} first messages</b> were removed from the context.\n Send /new command to start new dialog"
            await update.message.reply_text(text, parse_mode=ParseMode.HTML)

    async with user_semaphores[user_id]:
        task = asyncio.create_task(message_handle_fn())
        user_tasks[user_id] = task

        try:
            await task
        except asyncio.CancelledError:
            await update.message.reply_text("✅ Canceled", parse_mode=ParseMode.HTML)
        else:
            pass
        finally:
            if user_id in user_tasks:
                del user_tasks[user_id]


async def is_previous_message_not_answered_yet(
    update: Update, context: CallbackContext
):
    await register_user_if_not_exists(update, context, update.message.from_user)

    user_id = update.message.from_user.id
    if user_semaphores[user_id].locked():
        text = "⏳ Please <b>wait</b> for a reply to the previous message\n"
        text += "Or you can /cancel it"
        await update.message.reply_text(
            text, reply_to_message_id=update.message.id, parse_mode=ParseMode.HTML
        )
        return True
    else:
        return False


async def voice_message_handle(update: Update, context: CallbackContext):
    # check if bot was mentioned (for group chats)
    if not await is_bot_mentioned(update, context):
        return

    if not await force_channel_subscription_if_needed(update, context):
        return

    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context):
        return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    voice = update.message.voice
    voice_file = await context.bot.get_file(voice.file_id)

    # store file in memory, not on disk
    buf = io.BytesIO()
    await voice_file.download_to_memory(buf)
    buf.name = "voice.oga"  # file extension is required
    buf.seek(0)  # move cursor to the beginning of the buffer

    transcribed_text = await openai_utils.transcribe_audio(buf)
    text = f"🎤: <i>{transcribed_text}</i>"
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)

    # update n_transcribed_seconds
    db.set_user_attribute(
        user_id,
        "n_transcribed_seconds",
        voice.duration + db.get_user_attribute(user_id, "n_transcribed_seconds"),
    )

    await message_handle(update, context, message=transcribed_text)


async def generate_image_handle(update: Update, context: CallbackContext, message=None):
    if not await force_channel_subscription_if_needed(update, context):
        return
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context):
        return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    await update.message.chat.send_action(action="upload_photo")

    message = message or update.message.text

    try:
        image_urls = await openai_utils.generate_images(
            message, n_images=config.return_n_generated_images, size=config.image_size
        )
    except openai.error.InvalidRequestError as e:
        if str(e).startswith(
            "Your request was rejected as a result of our safety system"
        ):
            text = "🥲 Your request <b>doesn't comply</b> with OpenAI's usage policies.\nWhat did you write there, huh?"
            await update.message.reply_text(text, parse_mode=ParseMode.HTML)
            return
        else:
            raise

    # token usage
    db.set_user_attribute(
        user_id,
        "n_generated_images",
        config.return_n_generated_images
        + db.get_user_attribute(user_id, "n_generated_images"),
    )

    for i, image_url in enumerate(image_urls):
        await update.message.chat.send_action(action="upload_photo")
        await update.message.reply_photo(image_url, parse_mode=ParseMode.HTML)


async def new_dialog_handle(update: Update, context: CallbackContext):
    if not await force_channel_subscription_if_needed(update, context):
        return
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context):
        return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    db.start_new_dialog(user_id)
    await update.message.reply_text("Starting new dialog ✅")

    chat_mode = db.get_user_attribute(user_id, "current_chat_mode")
    await update.message.reply_text(
        f"{config.chat_modes[chat_mode]['welcome_message']}", parse_mode=ParseMode.HTML
    )


async def cancel_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    if user_id in user_tasks:
        task = user_tasks[user_id]
        task.cancel()
    else:
        await update.message.reply_text(
            "<i>Nothing to cancel...</i>", parse_mode=ParseMode.HTML
        )


def get_chat_mode_menu(page_index: int):
    n_chat_modes_per_page = config.n_chat_modes_per_page
    text = f"Select <b>chat mode</b> ({len(config.chat_modes)} modes available):"

    # buttons
    chat_mode_keys = list(config.chat_modes.keys())
    page_chat_mode_keys = chat_mode_keys[
        page_index * n_chat_modes_per_page : (page_index + 1) * n_chat_modes_per_page
    ]

    keyboard = []
    for chat_mode_key in page_chat_mode_keys:
        name = config.chat_modes[chat_mode_key]["name"]
        keyboard.append(
            [InlineKeyboardButton(name, callback_data=f"set_chat_mode|{chat_mode_key}")]
        )

    # pagination
    if len(chat_mode_keys) > n_chat_modes_per_page:
        is_first_page = page_index == 0
        is_last_page = (page_index + 1) * n_chat_modes_per_page >= len(chat_mode_keys)

        if is_first_page:
            keyboard.append(
                [
                    InlineKeyboardButton(
                        "»", callback_data=f"show_chat_modes|{page_index + 1}"
                    )
                ]
            )
        elif is_last_page:
            keyboard.append(
                [
                    InlineKeyboardButton(
                        "«", callback_data=f"show_chat_modes|{page_index - 1}"
                    ),
                ]
            )
        else:
            keyboard.append(
                [
                    InlineKeyboardButton(
                        "«", callback_data=f"show_chat_modes|{page_index - 1}"
                    ),
                    InlineKeyboardButton(
                        "»", callback_data=f"show_chat_modes|{page_index + 1}"
                    ),
                ]
            )

    reply_markup = InlineKeyboardMarkup(keyboard)

    return text, reply_markup


async def show_chat_modes_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context):
        return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    text, reply_markup = get_chat_mode_menu(0)
    await update.message.reply_text(
        text, reply_markup=reply_markup, parse_mode=ParseMode.HTML
    )


async def show_chat_modes_callback_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(
        update.callback_query, context, update.callback_query.from_user
    )
    if await is_previous_message_not_answered_yet(update.callback_query, context):
        return

    user_id = update.callback_query.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    query = update.callback_query
    await query.answer()

    page_index = int(query.data.split("|")[1])
    if page_index < 0:
        return

    text, reply_markup = get_chat_mode_menu(page_index)
    try:
        await query.edit_message_text(
            text, reply_markup=reply_markup, parse_mode=ParseMode.HTML
        )
    except telegram.error.BadRequest as e:
        if str(e).startswith("Message is not modified"):
            pass


async def set_chat_mode_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(
        update.callback_query, context, update.callback_query.from_user
    )
    user_id = update.callback_query.from_user.id

    query = update.callback_query
    await query.answer()

    chat_mode = query.data.split("|")[1]

    db.set_user_attribute(user_id, "current_chat_mode", chat_mode)
    db.start_new_dialog(user_id)

    await context.bot.send_message(
        update.callback_query.message.chat.id,
        f"{config.chat_modes[chat_mode]['welcome_message']}",
        parse_mode=ParseMode.HTML,
    )


def get_settings_menu(user_id: int):
    current_model = db.get_user_attribute(user_id, "current_model")
    text = config.models["info"][current_model]["description"]

    text += "\n\n"
    score_dict = config.models["info"][current_model]["scores"]
    for score_key, score_value in score_dict.items():
        text += "🟢" * score_value + "⚪️" * (5 - score_value) + f" – {score_key}\n\n"

    text += "\nSelect <b>model</b>:"

    # buttons to choose models
    buttons = []
    models_list = config.models["available_text_models"]

    # Add premium models if user is premium
    username = db.get_user_attribute(user_id, "username")
    if username in config.premium_usernames:
        models_list = models_list + config.models["available_premium_models"]

    for model_key in models_list:
        title = config.models["info"][model_key]["name"]
        if model_key == current_model:
            title = "✅ " + title

        buttons.append(
            InlineKeyboardButton(title, callback_data=f"set_settings|{model_key}")
        )
    reply_markup = InlineKeyboardMarkup([buttons])

    return text, reply_markup


async def settings_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context):
        return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    text, reply_markup = get_settings_menu(user_id)
    await update.message.reply_text(
        text, reply_markup=reply_markup, parse_mode=ParseMode.HTML
    )


async def set_settings_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(
        update.callback_query, context, update.callback_query.from_user
    )
    user_id = update.callback_query.from_user.id

    query = update.callback_query
    await query.answer()

    _, model_key = query.data.split("|")
    db.set_user_attribute(user_id, "current_model", model_key)
    db.start_new_dialog(user_id)

    text, reply_markup = get_settings_menu(user_id)
    try:
        await query.edit_message_text(
            text, reply_markup=reply_markup, parse_mode=ParseMode.HTML
        )
    except telegram.error.BadRequest as e:
        if str(e).startswith("Message is not modified"):
            pass


async def show_balance_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    # count total usage statistics
    total_n_spent_dollars = 0
    total_n_used_tokens = 0

    n_used_tokens_dict = db.get_user_attribute(user_id, "n_used_tokens")
    n_generated_images = db.get_user_attribute(user_id, "n_generated_images")
    n_transcribed_seconds = db.get_user_attribute(user_id, "n_transcribed_seconds")

    details_text = "🏷️ Details:\n"
    for model_key in sorted(n_used_tokens_dict.keys()):
        n_input_tokens, n_output_tokens = (
            n_used_tokens_dict[model_key]["n_input_tokens"],
            n_used_tokens_dict[model_key]["n_output_tokens"],
        )
        total_n_used_tokens += n_input_tokens + n_output_tokens

        n_input_spent_dollars = config.models["info"][model_key][
            "price_per_1000_input_tokens"
        ] * (n_input_tokens / 1000)
        n_output_spent_dollars = config.models["info"][model_key][
            "price_per_1000_output_tokens"
        ] * (n_output_tokens / 1000)
        total_n_spent_dollars += n_input_spent_dollars + n_output_spent_dollars

        details_text += f"- {model_key}: <b>{n_input_spent_dollars + n_output_spent_dollars:.03f}$</b> / <b>{n_input_tokens + n_output_tokens} tokens</b>\n"

    # image generation
    image_generation_n_spent_dollars = (
        config.models["info"]["dalle-2"]["price_per_1_image"] * n_generated_images
    )
    if n_generated_images != 0:
        details_text += f"- DALL·E 2 (image generation): <b>{image_generation_n_spent_dollars:.03f}$</b> / <b>{n_generated_images} generated images</b>\n"

    total_n_spent_dollars += image_generation_n_spent_dollars

    # voice recognition
    voice_recognition_n_spent_dollars = config.models["info"]["whisper"][
        "price_per_1_min"
    ] * (n_transcribed_seconds / 60)
    if n_transcribed_seconds != 0:
        details_text += f"- Whisper (voice recognition): <b>{voice_recognition_n_spent_dollars:.03f}$</b> / <b>{n_transcribed_seconds:.01f} seconds</b>\n"

    total_n_spent_dollars += voice_recognition_n_spent_dollars

    text = f"You spent <b>{total_n_spent_dollars:.03f}$</b>\n"
    text += f"You used <b>{total_n_used_tokens}</b> tokens\n\n"
    text += details_text

    await update.message.reply_text(text, parse_mode=ParseMode.HTML)


async def edited_message_handle(update: Update, context: CallbackContext):
    if update.edited_message.chat.type == "private":
        text = "🥲 Unfortunately, message <b>editing</b> is not supported"
        await update.edited_message.reply_text(text, parse_mode=ParseMode.HTML)


async def error_handle(update: Update, context: CallbackContext) -> None:
    logger.error(msg="Exception while handling an update:", exc_info=context.error)

    try:
        # collect error message
        tb_list = traceback.format_exception(
            None, context.error, context.error.__traceback__
        )
        tb_string = "".join(tb_list)
        update_str = update.to_dict() if isinstance(update, Update) else str(update)
        message = (
            f"An exception was raised while handling an update\n"
            f"<pre>update = {html.escape(json.dumps(update_str, indent=2, ensure_ascii=False))}"
            "</pre>\n\n"
            f"<pre>{html.escape(tb_string)}</pre>"
        )

        # split text into multiple messages due to 4096 character limit
        for message_chunk in split_text_into_chunks(message, 4096):
            try:
                await context.bot.send_message(
                    update.effective_chat.id, message_chunk, parse_mode=ParseMode.HTML
                )
            except telegram.error.BadRequest:
                # answer has invalid characters, so we send it without parse_mode
                await context.bot.send_message(update.effective_chat.id, message_chunk)
    except:
        await context.bot.send_message(
            update.effective_chat.id, "Some error in error handler"
        )


async def post_init(application: Application):
    await application.bot.set_my_commands(
        [
            BotCommand("/new", "Start new dialog"),
            BotCommand("/mode", "Select chat mode"),
            BotCommand("/retry", "Re-generate response for previous query"),
            BotCommand("/balance", "Show balance"),
            BotCommand("/settings", "Show settings"),
            BotCommand("/help", "Show help message"),
        ]
    )


def run_bot() -> None:
    application = (
        ApplicationBuilder()
        .token(config.telegram_token)
        .concurrent_updates(True)
        .rate_limiter(AIORateLimiter(max_retries=5))
        .http_version("1.1")
        .get_updates_http_version("1.1")
        .post_init(post_init)
        .build()
    )

    # add handlers
    user_filter = filters.ALL
    if len(config.allowed_telegram_usernames) > 0:
        usernames = [x for x in config.allowed_telegram_usernames if isinstance(x, str)]
        any_ids = [x for x in config.allowed_telegram_usernames if isinstance(x, int)]
        user_ids = [x for x in any_ids if x > 0]
        group_ids = [x for x in any_ids if x < 0]
        user_filter = (
            filters.User(username=usernames)
            | filters.User(user_id=user_ids)
            | filters.Chat(chat_id=group_ids)
        )

    application.add_handler(CommandHandler("start", start_handle, filters=user_filter))
    application.add_handler(CommandHandler("help", help_handle, filters=user_filter))
    # application.add_handler(
    #     CommandHandler("help_group_chat", help_group_chat_handle, filters=user_filter)
    # )

    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND & user_filter, message_handle)
    )
    application.add_handler(CommandHandler("retry", retry_handle, filters=user_filter))
    application.add_handler(
        CommandHandler("new", new_dialog_handle, filters=user_filter)
    )
    application.add_handler(
        CommandHandler("cancel", cancel_handle, filters=user_filter)
    )

    application.add_handler(
        MessageHandler(filters.VOICE & user_filter, voice_message_handle)
    )

    application.add_handler(
        CommandHandler("mode", show_chat_modes_handle, filters=user_filter)
    )
    application.add_handler(
        CallbackQueryHandler(
            show_chat_modes_callback_handle, pattern="^show_chat_modes"
        )
    )
    application.add_handler(
        CallbackQueryHandler(set_chat_mode_handle, pattern="^set_chat_mode")
    )

    application.add_handler(
        CommandHandler("settings", settings_handle, filters=user_filter)
    )
    application.add_handler(
        CallbackQueryHandler(set_settings_handle, pattern="^set_settings")
    )

    application.add_handler(
        CommandHandler("balance", show_balance_handle, filters=user_filter)
    )

    application.add_error_handler(error_handle)

    # start the bot
    application.run_polling()


if __name__ == "__main__":
    run_bot()
