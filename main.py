from typing import Final, Dict
from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup, LabeledPrice, PreCheckoutQuery
)
from telegram.ext import (
    CommandHandler, CallbackQueryHandler, Application,
    ContextTypes, MessageHandler, filters, PreCheckoutQueryHandler
)
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

from formula import ShortTermPredictor

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TOKEN: Final = "7464820680:AAHeYmWzf88-7KDs8BiJF8liG6sQDeN31Zc"
BOT_USERNAME: Final = "@signal_trading_ai_bot"

active_predictions: Dict[int, Dict] = {}
user_choices: Dict[int, Dict] = {}
user_subscription_status: Dict[int, bool] = {}

class PredictionManager:
    def __init__(self):
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

    async def run_prediction(self, crypto: str, minutes: int) -> tuple:
        try:
            predictor = ShortTermPredictor(crypto, minutes)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.thread_pool, predictor.train_model)
            prediction_result = await loop.run_in_executor(
                self.thread_pool,
                predictor.predict_next_movement
            )
            return prediction_result
        except Exception as e:
            logger.error(f"Prediction error for {crypto}: {str(e)}")
            raise

prediction_manager = PredictionManager()

free_usage: Dict[int, bool] = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id
    if user_id in free_usage:
        keyboard = [
            [InlineKeyboardButton("⭐ Subscribe - 250 Stars", callback_data="subscribe")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(
            "You have used your free trial.\n"
            "Subscribe for 250 Stars to continue using the bot.",
            reply_markup=reply_markup
        )
    else:
        free_usage[user_id] = True
        await send_crypto_options(update)

async def handle_subscription(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    prices = [LabeledPrice("Bot Subscription", 250)]  # 250 Stars
    await context.bot.send_invoice(
        chat_id=query.from_user.id,
        title="Crypto Signal Bot Subscription",
        description="Full access to predictions",
        payload="{}",
        provider_token="",
        currency="XTR",  # Changed to STARS
        prices=prices
    )

async def pre_checkout_query_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query: PreCheckoutQuery = update.pre_checkout_query
    await query.answer(ok=True)

async def successful_payment(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id
    free_usage[user_id] = False
    await update.message.reply_text(
        "⭐ Thank you for subscribing! You can now use the bot without restrictions."
    )

async def send_crypto_options(update: Update) -> None:
    keyboard = [
        [
            InlineKeyboardButton("₿ BTC-USD", callback_data="BTC-USD"),
            InlineKeyboardButton("Ξ ETH-USD", callback_data="ETH-USD")
        ],
        [
            InlineKeyboardButton("◎ SOL-USD", callback_data="SOL-USD"),
            InlineKeyboardButton("✧ XRP-USD", callback_data="XRP-USD")
        ],
        [
            InlineKeyboardButton("◎ DOGE-USD", callback_data="DOGE-USD"),
            InlineKeyboardButton("✧ DOT-USD", callback_data="DOT-USD")
        ],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "🚀 Welcome to Crypto Signal Bot!\n\nI can help you predict short-term cryptocurrency price movements using machine learning.\n\nPlease select a cryptocurrency to begin:",
        reply_markup=reply_markup
    )

async def crypto_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id

    if not user_subscription_status.get(user_id, False) and user_id in active_predictions:
        await query.edit_message_text(
            "❌ You have already used your free prediction.\n\n"
            "Type /subscribe to unlock unlimited access for 250 Stars."
        )
        return

    user_choices[user_id] = {"crypto": query.data}
    keyboard = [
        [
            InlineKeyboardButton("⚡ 1 Minute", callback_data="1"),
        ],
        [
            InlineKeyboardButton("🕐 5 Minutes", callback_data="5"),
            InlineKeyboardButton("🕑 15 Minutes", callback_data="15")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(
        f"Selected: {query.data}\nChoose prediction timeframe:",
        reply_markup=reply_markup
    )

async def time_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id

    if user_id not in user_choices:
        await query.edit_message_text("Session expired. Please start over with /start")
        return

    crypto = user_choices[user_id]["crypto"]
    minutes_ahead = int(query.data)
    user_choices[user_id]["time"] = minutes_ahead

    await query.edit_message_text(
        f"🔄 Analyzing {crypto} data...\n"
        "This may take a moment while I train the prediction model."
    )

    try:
        prediction, probabilities, current_price, current_time, confidence_level = (
            await prediction_manager.run_prediction(crypto, minutes_ahead)
        )

        direction = "⬆️ UP" if prediction == 1 else "⬇️ DOWN"
        probability = probabilities[1] * 100 if prediction == 1 else probabilities[0] * 100
        confidence_emoji = "🟢" if confidence_level == "High" else "🟡"

        result_message = (
            f"🎯 Prediction for {crypto}\n\n"
            f"⏰ Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"💰 Current Price: ${float(current_price):,.2f}\n"
            f"📈 Direction: {direction}\n"
            f"🎲 Probability: {probability:.1f}%\n"
            f"{confidence_emoji} Confidence: {confidence_level}\n"
            f"⏱ Timeframe: Next {minutes_ahead} minute(s)\n\n"
            "Type /start to make a new prediction"
        )

        active_predictions[user_id] = {
            "crypto": crypto,
            "timestamp": current_time,
            "prediction": prediction,
            "price": current_price
        }

        await query.edit_message_text(result_message)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        await query.edit_message_text("❌ An error occurred. Please try again later.")

def main() -> None:
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(crypto_selection, pattern="^(BTC-USD|ETH-USD|SOL-USD|XRP-USD|DOGE-USD|DOT-USD)$"))
    app.add_handler(CallbackQueryHandler(time_selection, pattern="^(1|5|15)$"))
    app.add_handler(CallbackQueryHandler(handle_subscription, pattern="^subscribe$"))
    app.add_handler(PreCheckoutQueryHandler(pre_checkout_query_handler))
    app.add_handler(MessageHandler(filters.SUCCESSFUL_PAYMENT, successful_payment))

    logger.info("🚀 Bot is starting...")
    app.run_polling()

if __name__ == "__main__":
    main()